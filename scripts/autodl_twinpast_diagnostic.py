"""Run the frozen, annotation-guided TwinPast development diagnostic."""
import argparse
import csv
import hashlib
import json
import shutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ARTIFACT = "HERMES_twinpast_diagnostic_evidence.zip"
ARMS = ("question_only", "recent_four", "history_a", "history_b")
ORDERS = ("canonical", "reversed")
LETTERS = "ABCDEFGH"
HOSTS = ("huggingface.co", "hf-mirror.com")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fetch(bundle, relative, dest, expected_sha, size=None, reuse=None):
    import requests

    if dest.is_file() and digest(dest) == expected_sha:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(".part")
    if reuse and reuse.is_file() and digest(reuse) == expected_sha:
        shutil.copyfile(reuse, temporary)
        temporary.replace(dest)
        print(f"MEDIA_REUSED {relative}", flush=True)
        return
    cap = size if size is not None else 2 * 1024**2
    remaining_attempts = [(host, direct) for direct in (False, True) for host in HOSTS]
    for host, direct in remaining_attempts:
        url = f"https://{host}/datasets/{bundle['manifest']['dataset']}/resolve/{bundle['manifest']['revision']}/{relative}"
        try:
            with requests.Session() as session:
                session.trust_env = not direct
                with session.get(url, stream=True, timeout=(8, 15)) as response, temporary.open("wb") as stream:
                    response.raise_for_status()
                    start, last_report, count = time.monotonic(), time.monotonic(), 0
                    for block in response.iter_content(128 * 1024):
                        count += len(block)
                        now = time.monotonic()
                        if count > cap or now - start > 120:
                            raise ValueError("Download exceeded size or elapsed-time bound")
                        stream.write(block)
                        if now - last_report >= 8:
                            print(f"MEDIA_PROGRESS {relative} {count / 1024**2:.1f} MiB", flush=True)
                            last_report = now
            if (size is not None and count != size) or digest(temporary) != expected_sha:
                raise ValueError("Downloaded size or SHA-256 does not match")
            temporary.replace(dest)
            print(f"MEDIA_READY {relative} {count / 1024**2:.1f} MiB", flush=True)
            return
        except (requests.RequestException, OSError, ValueError) as exc:
            print(f"MEDIA_RETRY {relative} host={host} direct={direct} reason={type(exc).__name__}", flush=True)
    raise RuntimeError(f"Could not acquire {relative}. Verified completed files remain available for rerun.")


def cases(bundle):
    plans = {row["pair_id"]: row for row in bundle["manifest"]["selected_pairs"]}
    visual = {row["pair_id"]: row for row in bundle["visual"]}
    media = {row["pair_id"]: row for row in bundle["media"]}
    assert len(plans) == len(visual) == len(media) == 12 and plans.keys() == visual.keys() == media.keys()
    selected = []
    for pair_id, plan in plans.items():
        if visual[pair_id]["status"] == "clear_contrast":
            checked = media[pair_id]
            assert checked["status"] == "checked" and checked["task"] == plan["task"]
            assert checked["grid_s"] == sorted(set(checked["grid_s"])) and len(checked["grid_s"]) == 6
            assert checked["grid_s"][-4:] == plan["latest_four_grid_s"]
            assert checked["grid_s"][-1] < plan["query_time_s"]
            for branch in "ab":
                assert all(checked["decoded"][branch]["early_samples_inside"])
            assert checked["decoded"]["a"]["decoded_rgb_sha256"][-4:] == checked["decoded"]["b"]["decoded_rgb_sha256"][-4:]
            selected.append((plan, checked, visual[pair_id]))
    assert len(selected) == 9
    return selected


def annotation_rows(bundle, raw):
    assert sha(raw) == bundle["manifest"]["annotation_sha256"], "Pinned annotations changed"
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    result = {row["pair_id"]: row for row in rows}
    assert len(rows) == len(result) == bundle["manifest"]["dataset_pairs"]
    for plan, media, visual in cases(bundle):
        row = result[plan["pair_id"]]
        assert all(row[field] == plan[field] for field in ("video_a", "video_b", "task"))
        assert row["t_query"] == plan["query_time_s"] and row["question"].strip()
        choices = row["options"]
        assert 2 <= len(choices) <= len(LETTERS) and all(c.strip() for c in choices)
        assert len({c.strip().casefold() for c in choices}) == len(choices)
        assert row["answer_a"] != row["answer_b"]
        for branch in "ab":
            assert row[f"answer_{branch}"] in LETTERS[:len(choices)]
            assert row[f"answer_{branch}"] == visual[f"observed_option_{branch}"]
            seconds = media["decoded"][branch]["frame_start_s"]
            assert all(t < row["t_query"] for t in seconds)
            assert all(any(lo - 1e-6 <= t <= hi + 1e-6
                           for lo, hi in row["evidence_intervals"][branch.upper()]) for t in seconds[:2])
    return result


def decode(path, recorded):
    from decord import VideoReader

    reader = VideoReader(str(path), num_threads=1)
    frames = reader.get_batch(recorded["source_frames"]).asnumpy()
    assert [sha(frame.tobytes()) for frame in frames] == recorded["decoded_rgb_sha256"], (
        f"Decoded frames differ from the frozen audit for {path.name}")
    return frames


def features(model, frames):
    import torch

    chw = torch.from_numpy(frames).permute(0, 3, 1, 2)
    pixels = model.processor.video_processor(chw, return_tensors="pt").pixel_values_videos
    output = model.get_video_features(pixels.to(model.device, model.dtype))
    assert output.shape[0] == 1 and output.shape[1] % len(frames) == 0
    return output


def score(model, system, prefix, option_ids, visual, cached=False):
    import torch

    lm = model.language_model
    embed = lm.get_input_embeddings()
    question = embed(prefix)
    left = system if visual is None else torch.cat((system, visual), dim=1)
    if cached:
        initial = lm(inputs_embeds=left, use_cache=True, return_dict=True)
        output = lm(inputs_embeds=question, past_key_values=initial.past_key_values, use_cache=True, return_dict=True)
    else:
        output = lm(inputs_embeds=torch.cat((left, question), dim=1), use_cache=False, return_dict=True)
    result = output.logits[0, -1, option_ids].float().detach().cpu()
    assert torch.isfinite(result).all()
    return result, left.shape[1] + question.shape[1]


def make_prompt(model, question, choices, order, with_video):
    import torch

    perm = list(range(len(choices)))
    if order == "reversed":
        perm.reverse()
    letters = LETTERS[:len(choices)]
    context = "Answer the question about the video." if with_video else "Answer the question."
    options = "\n".join(f"({letter}) {choices[i]}" for letter, i in zip(letters, perm, strict=True))
    text = model.get_prompt(f"{context}\nQuestion: {question}\nOptions:\n{options}\n"
                            "Reply with only the letter of the best option.") + "Answer letter:\n"
    tokenizer = model.processor.tokenizer
    ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = []
    for letter in letters:
        suffix = tokenizer.encode(letter, add_special_tokens=False)
        assert len(suffix) == 1 and tokenizer.encode(text + letter, add_special_tokens=False) == ids + suffix
        tokens.extend(suffix)
    assert len(set(tokens)) == len(tokens)
    return torch.tensor([ids], device=model.device), tokens, perm, sha(text.encode("utf-8"))


def evaluate_pair(model, plan, media, row, paths, preflight=False):
    import numpy as np
    import torch

    frames = {branch: decode(paths[branch], media["decoded"][branch]) for branch in "ab"}
    assert np.array_equal(frames["a"][2:], frames["b"][2:])
    recent = features(model, frames["a"][2:])
    a = features(model, frames["a"][:2])
    b = features(model, frames["b"][:2])
    assert a.shape == b.shape and a.shape[1] * 2 == recent.shape[1]
    visuals = {"question_only": None, "recent_four": recent,
               "history_a": torch.cat((a, recent), dim=1), "history_b": torch.cat((b, recent), dim=1)}
    lm = model.language_model
    system = lm.get_input_embeddings()(torch.as_tensor(model.init_prompt_ids, device=model.device))
    results, checks = [], []
    for order in ORDERS:
        prompt = make_prompt(model, row["question"], row["options"], order, True)
        for arm in ARMS:
            ids, option_ids, perm, prompt_sha = prompt if arm != "question_only" else make_prompt(
                model, row["question"], row["options"], order, False)
            started = time.perf_counter()
            output, count = score(model, system, ids, option_ids, visuals[arm])
            elapsed = time.perf_counter() - started
            if preflight:
                replay, replay_count = score(model, system, ids, option_ids, visuals[arm])
                torch.testing.assert_close(output, replay, atol=0, rtol=0)
                assert replay_count == count
                if arm == "history_a":
                    cached, _ = score(model, system, ids, option_ids, visuals[arm], cached=True)
                    torch.testing.assert_close(output, cached, atol=0.025, rtol=0.005)
                    assert output.argmax().item() == cached.argmax().item(), "Native-cache top option differs"
                    checks.append(dict(order=order, native_cache_max_abs=float((output - cached).abs().max())))
            values = [0.0] * len(perm)
            for idx, value in zip(perm, output.tolist(), strict=True):
                values[idx] = value
            # Canonical tie-breaking prevents label order from deciding exact ties.
            pred = LETTERS[max(range(len(values)), key=values.__getitem__)]
            recent_sha = media["decoded"]["a"]["decoded_rgb_sha256"][-4:] if arm != "question_only" else []
            earlier_sha = media["decoded"][arm[-1]]["decoded_rgb_sha256"][:2] if arm.startswith("history") else []
            results.append(dict(pair_id=plan["pair_id"], task=plan["task"], order=order, arm=arm, prediction=pred,
                                gold_a=row["answer_a"], gold_b=row["answer_b"], scores=values, permutation=perm,
                                label_token_ids=option_ids, prompt_sha256=prompt_sha, lm_tokens=count,
                                earlier_frame_sha256=earlier_sha, recent_frame_sha256=recent_sha,
                                forward_s=elapsed))
    return results, checks


def summarize(blocks):
    pairs = []
    for block in blocks:
        records = {(r["order"], r["arm"]): r for r in block}
        pair = dict(pair_id=block[0]["pair_id"], task=block[0]["task"])
        for order in ORDERS:
            a, b = (records[order, arm] for arm in ("history_a", "history_b"))
            assert a["lm_tokens"] == b["lm_tokens"] and a["prompt_sha256"] == b["prompt_sha256"]
            assert a["recent_frame_sha256"] == b["recent_frame_sha256"]
            ai, bi = (LETTERS.index(a[f"gold_{branch}"]) for branch in "ab")
            ma = a["scores"][ai] - a["scores"][bi]
            mb = b["scores"][ai] - b["scores"][bi]
            wins = int(a["prediction"] == a["gold_a"]) + int(b["prediction"] == b["gold_b"])
            reversal = a["prediction"] != b["prediction"]
            pair[order] = dict(correct_branches=wins, correct_reversal=wins == 2, unchanged=not reversal,
                               wrong_reversal=reversal and wins != 2, margin_a=ma, margin_b=mb,
                               directional_change=ma - mb)
            for arm in ("question_only", "recent_four"):
                prediction = records[order, arm]["prediction"]
                pair[order][f"{arm}_correct_branches"] = int(prediction in (a["gold_a"], a["gold_b"]))
        pair["both_orders_correct"] = all(pair[order]["correct_reversal"] for order in ORDERS)
        pair["predictions_order_stable"] = {
            arm: records["canonical", arm]["prediction"] == records["reversed", arm]["prediction"] for arm in ARMS
        }
        pairs.append(pair)
    orders = {order: dict(
        both_correct=sum(row[order]["correct_reversal"] for row in pairs),
        correct_branches=sum(row[order]["correct_branches"] for row in pairs),
        question_only_correct_branches=sum(row[order]["question_only_correct_branches"] for row in pairs),
        recent_four_correct_branches=sum(row[order]["recent_four_correct_branches"] for row in pairs),
        wrong_reversals=sum(row[order]["wrong_reversal"] for row in pairs),
        unchanged=sum(row[order]["unchanged"] for row in pairs),
        median_directional_change=statistics.median(row[order]["directional_change"] for row in pairs))
        for order in ORDERS} if pairs else {}
    robust = sum(row["both_orders_correct"] for row in pairs)
    numeric_gate = len(pairs) == 9 and robust >= 5 and all(x["median_directional_change"] > 0 for x in orders.values())
    return dict(pairs=len(pairs), robust_correct_pairs=robust, order_summary=orders, per_pair=pairs,
                numeric_development_gate=numeric_gate,
                advancement="pending source-group verification" if numeric_gate else
                ("stop on this backbone if diagnostic validity holds" if len(pairs) == 9 and robust == 0
                 else "review diagnostic outcomes before advancing"),
                source_groups="The pinned annotations do not identify original source groups. Verify independently.")


def package(work):
    artifact = work / ARTIFACT
    temporary = work / (ARTIFACT + ".tmp")
    files = [path for name in ("exports", "checkpoints", "provenance", "pinned_source", "logs")
             for path in (work / name).rglob("*") if path.is_file() and path.suffix != ".tmp"]
    files += [path for path in (work / "launch.json", work / "completion.json") if path.exists()]
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        entries = []
        for path in sorted(files):
            raw = path.read_bytes()
            relative = path.relative_to(work).as_posix()
            archive.writestr(relative, raw)
            entries.append(dict(path=relative, bytes=len(raw), sha256=sha(raw)))
        archive.writestr("evidence_manifest.json", json.dumps(entries, indent=2))
    temporary.replace(artifact)
    print(f"EVIDENCE_ZIP={artifact}", flush=True)


def execute(args):
    import torch
    from inference.llavaov_hermes import load_model

    work = args.work
    bundle_path = work / "provenance" / "bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    selection = cases(bundle)
    dataset = work / "media" / "data.jsonl"
    fetch(bundle, "data.jsonl", dataset, bundle["manifest"]["annotation_sha256"])
    rows = annotation_rows(bundle, dataset.read_bytes())
    atomic(work / "provenance" / "selected_annotations.json", [rows[plan["pair_id"]] for plan, _, _ in selection])
    needed = sum(meta["bytes"] for plan, media, _ in selection
                 for branch, meta in media["decoded"].items()
                 if not (work / "media" / plan[f"video_{branch}"]).is_file()
                 or (work / "media" / plan[f"video_{branch}"]).stat().st_size != meta["bytes"])
    assert shutil.disk_usage(work).free >= needed + 256 * 1024**2, "Free data-disk space before running"
    locations = {}
    print(f"MEDIA_PLAN 18 pinned clips, up to {needed / 1024**2:.1f} MiB not present", flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = []
        for plan, media, _ in selection:
            locations[plan["pair_id"]] = {}
            for branch, meta in media["decoded"].items():
                relative = plan[f"video_{branch}"]
                dest = work / "media" / relative
                reuse = args.media_reuse / Path(relative).relative_to("videos")
                pending.append(pool.submit(fetch, bundle, relative, dest, meta["sha256"], meta["bytes"], reuse))
                locations[plan["pair_id"]][branch] = dest
        for download in pending:
            download.result()
    settings = dict(bundle_sha256=digest(bundle_path), runner_sha256=digest(Path(__file__)),
                    model_name=args.model.name, torch_version=torch.__version__,
                    gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                    driver=torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") else None)
    identity = sha(json.dumps(settings, sort_keys=True).encode())
    checkpoint_dir = work / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    atomic(work / "exports" / "run_config.json", settings)
    print("LOADING_PINNED_MODEL", args.model.name, flush=True)
    model, _ = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=0.5)
    assert model.device.type == "cuda"
    blocks = []
    with torch.inference_mode():
        for index, (plan, media, _) in enumerate(selection):
            pair_id = plan["pair_id"]
            checkpoint = checkpoint_dir / f"{pair_id}.json"
            if index and checkpoint.is_file():
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                assert saved["identity"] == identity and saved["pair_id"] == pair_id, "Use a new WORK for changed settings"
                records = saved["records"]
                assert len(records) == 8 and {(r["order"], r["arm"]) for r in records} == {
                    (order, arm) for order in ORDERS for arm in ARMS}
                assert all(r["pair_id"] == pair_id for r in records)
                print(f"PAIR_REUSED {index + 1}/9 {pair_id}", flush=True)
            else:
                print(f"PAIR_RUNNING {index + 1}/9 {pair_id}", flush=True)
                records, checks = evaluate_pair(model, plan, media, rows[pair_id], locations[pair_id], preflight=index == 0)
                if index == 0:
                    if checkpoint.exists():
                        assert json.loads(checkpoint.read_text(encoding="utf-8"))["identity"] == identity, (
                            "Use a new WORK for changed settings")
                    atomic(work / "exports" / "preflight.json", dict(status="pass", pair_id=pair_id, checks=checks))
                    print("TWINPAST_PREFLIGHT_PASS", flush=True)
                atomic(checkpoint, dict(identity=identity, pair_id=pair_id, records=records))
                print(f"PAIR_DONE {index + 1}/9 {pair_id}", flush=True)
            blocks.append(records)
    summary = summarize(blocks)
    atomic(work / "exports" / "summary.json", summary)
    fields = ("pair_id", "task", "order", "arm", "prediction", "gold_a", "gold_b", "scores", "permutation",
              "label_token_ids", "prompt_sha256", "lm_tokens", "earlier_frame_sha256", "recent_frame_sha256", "forward_s")
    with (work / "exports" / "scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()}
                         for block in blocks for row in block)
    atomic(work / "completion.json", dict(status="complete", pairs=len(blocks), scores=sum(map(len, blocks))))
    print(f"TWINPAST_DIAGNOSTIC_COMPLETE pairs={len(blocks)} scores={sum(map(len, blocks))}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--media-reuse", type=Path, default=Path("/root/autodl-tmp/twinpast-development"))
    args = parser.parse_args()
    from filelock import FileLock

    with FileLock(str(args.work / "run.lock"), timeout=0):
        (args.work / "exports").mkdir(exist_ok=True)
        atomic(args.work / "completion.json", dict(status="running"))
        try:
            execute(args)
        except BaseException as exc:
            atomic(args.work / "completion.json", dict(status="failed", error_type=type(exc).__name__, message=str(exc)))
            raise
        finally:
            package(args.work)


if __name__ == "__main__":
    main()
