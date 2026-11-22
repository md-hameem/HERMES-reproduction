"""Test whether earlier evidence alone supports the frozen TwinPast answers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import twinpast_base as base

ARTIFACT = "HERMES_twinpast_history_only_evidence.zip"
DENSE_COUNT = 6


def dense_indices(seconds, intervals, earlier, cutoff):
    """Keep the two original frames and add four spread across valid native timestamps."""
    import numpy as np

    mask = np.zeros(len(seconds), dtype=bool)
    for start, stop in intervals:
        mask |= (seconds >= start - 1e-6) & (seconds <= stop + 1e-6)
    eligible = np.flatnonzero(mask & (seconds < cutoff)).tolist()
    assert sorted(set(earlier)) == earlier and len(earlier) == 2
    assert set(earlier) <= set(eligible), "Frozen history frame outside eligible annotated evidence"
    if len(eligible) < DENSE_COUNT:
        raise ValueError(f"Only {len(eligible)} distinct annotated frames before recent context")
    chosen = set(earlier)
    for k in range(DENSE_COUNT - len(earlier)):
        target = k * (len(eligible) - 1) / (DENSE_COUNT - len(earlier) - 1)
        pos = min((i for i, ix in enumerate(eligible) if ix not in chosen), key=lambda i: abs(i - target))
        chosen.add(eligible[pos])
    return sorted(chosen)


def prepare_dense(paths, annotated, media):
    import numpy as np
    from decord import VideoReader

    result = {}
    cutoff = min(annotated["t_query"], *(media["decoded"][b]["frame_start_s"][2] for b in "ab"))
    for branch in "ab":
        reader = VideoReader(str(paths[branch]), num_threads=1)
        seconds = np.asarray(reader.get_frame_timestamp(list(range(len(reader)))), dtype=np.float64)
        seconds -= seconds[0, 0]
        seconds = seconds[:, 0]
        indices = dense_indices(seconds, annotated["evidence_intervals"][branch.upper()],
                                media["decoded"][branch]["source_frames"][:2], cutoff)
        frames = reader.get_batch(indices).asnumpy()
        result[branch] = dict(frames=frames, indices=indices, seconds=seconds[indices].tolist(),
                              cutoff_s=cutoff, hashes=[base.sha(frame.tobytes()) for frame in frames])
    return result


def contact_sheet(path, pair_id, condition, windows):
    from PIL import Image, ImageDraw

    cols = len(windows["a"]["frames"])
    canvas = Image.new("RGB", (320 * cols, 450), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), f"{pair_id} | {condition} | Native frames resized to fit", fill="black")
    for row, branch in enumerate("ab"):
        for col, (frame, seconds) in enumerate(zip(windows[branch]["frames"], windows[branch]["seconds"], strict=True)):
            x, y = col * 320, 23 + row * 210
            picture = Image.fromarray(frame)
            picture.thumbnail((320, 180))
            canvas.paste(picture, (x, y))
            draw.text((x + 4, y + 182), f"{branch.upper()} | {seconds:.3f} s", fill="black")
    path.parent.mkdir(exist_ok=True)
    canvas.save(path, quality=90)


def evaluate(model, annotated, windows, reference, preflight=False):
    import torch

    system = model.language_model.get_input_embeddings()(
        torch.as_tensor(model.init_prompt_ids, device=model.device))
    visual = {branch: base.features(model, window["frames"]) for branch, window in windows.items()}
    assert visual["a"].shape == visual["b"].shape
    rows, checks = [], []
    for order in base.ORDERS:
        ids, option_ids, perm, prompt_sha = base.make_prompt(
            model, annotated["question"], annotated["options"], order, True)
        for branch in "ab":
            old = reference[order, branch]
            assert old["prompt_sha256"] == prompt_sha and old["label_token_ids"] == option_ids
            start = time.perf_counter()
            output, length = base.score(model, system, ids, option_ids, visual[branch])
            elapsed = time.perf_counter() - start
            if preflight:
                repeat, _ = base.score(model, system, ids, option_ids, visual[branch])
                torch.testing.assert_close(output, repeat, rtol=0, atol=0)
                if branch == "a":
                    cached, _ = base.score(model, system, ids, option_ids, visual[branch], cached=True)
                    torch.testing.assert_close(output, cached, rtol=0.005, atol=0.025)
                    assert output.argmax().item() == cached.argmax().item()
                    checks.append(dict(order=order, max_native_cache_diff=float((output - cached).abs().max())))
            values = [0.0] * len(perm)
            for i, value in zip(perm, output.tolist(), strict=True):
                values[i] = value
            prediction = base.LETTERS[max(range(len(values)), key=values.__getitem__)]
            gold = annotated[f"answer_{branch}"]
            window = windows[branch]
            rows.append(dict(order=order, branch=branch, prediction=prediction, gold=gold, scores=values,
                             prompt_sha256=prompt_sha, lm_tokens=length, frame_indices=window["indices"],
                             frame_seconds=window["seconds"], frame_hashes=window["hashes"],
                             prior_prediction=old["prediction"], prior_scores=old["scores"], forward_s=elapsed))
        assert rows[-1]["lm_tokens"] == rows[-2]["lm_tokens"]
    return rows, checks


def outcome(rows, annotated):
    wins = {}
    ai, bi = (base.LETTERS.index(annotated[f"answer_{b}"]) for b in "ab")
    for order in base.ORDERS:
        a, b = (next(r for r in rows if r["order"] == order and r["branch"] == branch) for branch in "ab")
        margin = lambda x: x[ai] - x[bi]
        wins[order] = dict(branches_correct=int(a["prediction"] == a["gold"]) + int(b["prediction"] == b["gold"]),
                           directional_change=margin(a["scores"]) - margin(b["scores"]))
    return dict(orders=wins, robust=all(row["branches_correct"] == 2 for row in wins.values()))


def acquire(args, bundle, selection):
    paths = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        downloads = []
        for plan, media, _ in selection:
            pair_paths = paths[plan["pair_id"]] = {}
            for branch, check in media["decoded"].items():
                relative = Path(plan[f"video_{branch}"])
                reused = args.previous / "media" / relative
                if reused.is_file() and base.digest(reused) == check["sha256"]:
                    pair_paths[branch] = reused
                    print(f"MEDIA_REUSED {relative.as_posix()}", flush=True)
                else:
                    dest = args.work / "media" / relative
                    downloads.append(pool.submit(base.fetch, bundle, relative.as_posix(), dest, check["sha256"],
                                                 check["bytes"], args.media_reuse / relative.relative_to("videos")))
                    pair_paths[branch] = dest
        for download in downloads:
            download.result()
    return paths


def execute(args):
    import torch
    from inference.llavaov_hermes import load_model

    work = args.work
    provenance = work / "provenance"
    bundle = json.loads((provenance / "bundle.json").read_bytes())
    prior = json.loads((provenance / "prior.json").read_bytes())
    selection = base.cases(bundle)
    annotations = {r["pair_id"]: r for r in prior["annotations"]}
    assert annotations.keys() == {plan["pair_id"] for plan, _, _ in selection}
    paths = acquire(args, bundle, selection)
    print("LOADING_PINNED_MODEL", args.model.name, flush=True)
    model, _ = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=0.5)
    assert model.device.type == "cuda"
    config = dict(model_revision=args.model.name, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  cuda=torch.version.cuda, provenance_hashes={p.name: base.digest(p) for p in sorted(provenance.iterdir())})
    ident = base.sha(json.dumps(config, sort_keys=True).encode())
    exports = work / "exports"
    checkpoints = work / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    base.atomic(exports / "run_config.json", config)
    reports = []
    with torch.inference_mode():
        for index, (plan, media, screened) in enumerate(selection):
            pair_id = plan["pair_id"]
            annotated = annotations[pair_id]
            assert all(annotated[f"answer_{b}"] == screened[f"observed_option_{b}"] for b in "ab")
            old = {(r["order"], r["arm"][-1]): r for r in prior["pairs"][pair_id]}
            checkpoint = checkpoints / f"{pair_id}.json"
            expected = json.loads(checkpoint.read_bytes()) if checkpoint.is_file() else None
            if expected is not None:
                assert expected["identity"] == ident and expected["pair_id"] == pair_id
            if index and expected is not None:
                reports.append(expected["summary"])
                print(f"PAIR_REUSED {index + 1}/9 {pair_id}", flush=True)
                continue
            two = {}
            for branch in "ab":
                record = media["decoded"][branch]
                assert all(r["earlier_frame_sha256"] == record["decoded_rgb_sha256"][:2] for (_, b), r in old.items() if b == branch)
                frames = base.decode(paths[pair_id][branch], record)
                two[branch] = dict(frames=frames[:2], indices=record["source_frames"][:2],
                                   seconds=record["frame_start_s"][:2], hashes=record["decoded_rgb_sha256"][:2])
            rows, checks = evaluate(model, annotated, two, old, preflight=index == 0)
            if index == 0:
                base.atomic(exports / "preflight.json", dict(status="pass", checks=checks))
                print("TWINPAST_HISTORY_ONLY_PREFLIGHT_PASS", flush=True)
            comparison = outcome(rows, annotated)
            summary = dict(pair_id=pair_id, task=plan["task"], two_history_only=comparison,
                           prior_history_with_recent_robust=all(r["prediction"] == r[f"gold_{b}"] for (_, b), r in old.items()))
            dense_rows = []
            if plan["task"] == "HSR":
                contact_sheet(exports / "review" / f"{pair_id}_two.jpg", pair_id, "Two earlier frames", two)
                if not comparison["robust"]:
                    try:
                        dense = prepare_dense(paths[pair_id], annotated, media)
                    except ValueError as exc:
                        summary["dense_skip"] = str(exc)
                    else:
                        dense_rows, _ = evaluate(model, annotated, dense, old)
                        summary["dense_history_only"] = outcome(dense_rows, annotated)
                        contact_sheet(exports / "review" / f"{pair_id}_dense.jpg", pair_id, "Six earlier frames", dense)
            base.atomic(checkpoint, dict(pair_id=pair_id, identity=ident, summary=summary,
                                        history_two=rows, history_dense=dense_rows))
            reports.append(summary)
            print(f"PAIR_DONE {index + 1}/9 {pair_id} two_order_robust={comparison['robust']}", flush=True)
    dense = [item["dense_history_only"] for item in reports if "dense_history_only" in item]
    summary = dict(pairs=len(reports), two_history_only_robust=sum(r["two_history_only"]["robust"] for r in reports),
                   restored_from_original=sum(r["two_history_only"]["robust"] and not r["prior_history_with_recent_robust"]
                                              for r in reports),
                   lost_from_original=sum(not r["two_history_only"]["robust"] and r["prior_history_with_recent_robust"]
                                          for r in reports),
                   dense_pairs=len(dense), dense_robust=sum(r["robust"] for r in dense), per_pair=reports)
    base.atomic(exports / "summary.json", summary)
    base.atomic(work / "completion.json", dict(status="complete", pairs=len(reports), dense_pairs=len(dense)))
    print("TWINPAST_HISTORY_ONLY_COMPLETE", json.dumps({k: v for k, v in summary.items() if k != "per_pair"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "model", "media-reuse", "previous"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    from filelock import FileLock

    with FileLock(str(args.work / "run.lock"), timeout=0):
        (args.work / "exports").mkdir(exist_ok=True)
        base.ARTIFACT = ARTIFACT
        base.atomic(args.work / "completion.json", dict(status="running"))
        try:
            execute(args)
        except BaseException as exc:
            base.atomic(args.work / "completion.json", dict(status="failed", error_type=type(exc).__name__, message=str(exc)))
            raise
        finally:
            base.package(args.work)


if __name__ == "__main__":
    main()
