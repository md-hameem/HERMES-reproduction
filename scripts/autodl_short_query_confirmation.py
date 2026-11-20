"""Run the frozen clip-disjoint StreamingBench comparison."""
import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote
from zipfile import BadZipFile, ZipFile

try:
    from provenance import short_base as q
    from provenance import selection_base as selection
except ModuleNotFoundError as exc:
    if exc.name != "provenance":
        raise
    import autodl_short_query as q
    import autodl_construction_selection as selection

s, d, p = q.s, q.d, q.p
ARMS = ("compact_recent4", "latest_two_short_readout", "all_history_short_readout")
ARTIFACT = "HERMES_short_query_confirmation_evidence.zip"
SOURCE_PATHS = (*q.SOURCE_PATHS, "provenance/short_base.py", "provenance/selection_base.py")


def acquire_media(shards, work):
    """Read only listed ZIP members, check CRC and hash each realized input."""
    import requests

    output = work / "videos"
    output.mkdir(exist_ok=True)
    receipts = work / "provenance/media"
    receipts.mkdir(parents=True, exist_ok=True)
    assets = []
    for shard in shards:
        expected = [c["sample_id"] for c in shard["clips"]]
        receipt = receipts / f"{shard['shard']}.json"
        old = json.loads(receipt.read_text()) if receipt.exists() else {}
        valid = {}
        for clip in old.get("videos", []):
            path = output / f"sample_{clip['sample_id']}_real.mp4"
            if (clip["sample_id"] in expected and path.is_file() and path.stat().st_size == clip["bytes"]
                    and p.digest(path) == clip["sha256"]):
                valid[clip["sample_id"]] = dict(clip, path=str(path))
        received = old.get("transferred_bytes", 0)
        if set(valid) != set(expected):
            for host in ("huggingface.co", "hf-mirror.com"):
                url = f"https://{host}/datasets/mjuicem/StreamingBench/resolve/main/{quote(shard['archive'])}?download=true"
                try:
                    with requests.Session() as session:
                        accessor = p.HTTPRangeFile(url, shard["archive_bytes"], session)
                        try:
                            with ZipFile(accessor) as archive:
                                for clip in shard["clips"]:
                                    sid = clip["sample_id"]
                                    if sid in valid:
                                        continue
                                    item, = p.extract_members(archive, [dict(video_path=clip["video_basename"])], output)
                                    valid[sid] = item
                                    p.atomic_json(receipt, dict(shard=shard["shard"], archive=shard["archive"],
                                        transferred_bytes=received + accessor.bytes_received, videos=list(valid.values())))
                        finally:
                            received += accessor.bytes_received
                    break
                except (OSError, ValueError, BadZipFile, p.RangeUnsupported, requests.RequestException) as exc:
                    if host == "hf-mirror.com":
                        raise RuntimeError("Selected-member download failed on both endpoints") from exc
                    print(f"MEDIA_RANGE_RETRY {shard['shard']} {type(exc).__name__}", flush=True)
        if set(valid) != set(expected):
            raise ValueError("Frozen media set incomplete")
        entries = [valid[i] for i in expected]
        p.atomic_json(receipt, dict(shard=shard["shard"], archive=shard["archive"],
                                   transferred_bytes=received, videos=entries))
        assets.extend(entries)
    p.atomic_json(work / "asset_manifest.json", dict(mode="range_members_and_sha256", videos=assets))
    return {a["sample_id"]: a for a in assets}


def videos_for_manifest(manifest, repo):
    path = repo / manifest["annotation_path"]
    if p.digest(path) != manifest["annotation_sha256"]:
        raise ValueError("Annotation digest differs from the frozen manifest")
    annotation = {p.sample_id(v): v for v in json.loads(path.read_text(encoding="utf-8"))}
    videos = []
    for shard in manifest["shards"]:
        for clip in shard["clips"]:
            sid = clip["sample_id"]
            video = dict(annotation[sid], sample_id=sid)
            if (video["video_id"] != clip["video_id"] or len(video["conversations"]) != clip["question_count"]
                    or Path(video["video_path"]).name != clip["video_basename"]):
                raise ValueError("Frozen clip identity differs")
            if any(len(qa["choices"]) > 8 or qa["choices"].count(qa["answer"]) != 1 for qa in video["conversations"]):
                raise ValueError(f"Ambiguous multiple-choice gold for sample {sid}")
            videos.append(video)
    if len(videos) != 30 or len({v["sample_id"] for v in videos}) != 30:
        raise ValueError("Invalid confirmation video count")
    if sum(len(v["conversations"]) for v in videos) != 150:
        raise ValueError("Invalid confirmation question count")
    return videos


def plan_sample_grid(frame_timestamps, video):
    """Sample the latest frame at or before each 2-second grid point."""
    import math
    import numpy as np

    frame_timestamps = np.asarray(frame_timestamps, dtype=np.float64)
    if frame_timestamps.ndim != 2 or frame_timestamps.shape[1] != 2 or len(frame_timestamps) == 0:
        raise ValueError("Invalid source timestamps")
    relative = frame_timestamps - frame_timestamps[0, 0]
    if not np.isfinite(relative).all() or np.any(np.diff(relative[:, 0]) < 0):
        raise ValueError("Nonmonotone source timestamps")
    end = max(float(qa["end_time"]) for qa in video["conversations"])
    grid = np.arange(math.ceil(end * p.FPS)) / p.FPS
    if not len(grid) or grid[-1] > relative[-1, 1] + 1e-6:
        raise ValueError("Frozen query extends beyond the sampled source")
    indices = np.searchsorted(relative[:, 0], grid, side="right") - 1
    if indices[0] < 0 or np.any(np.diff(indices) <= 0):
        raise ValueError("Source cannot represent the frozen grid without repeated frames")
    return [dict(sampled_index=i, source_frame=int(j), start_s=float(relative[j, 0]),
                 end_s=float(relative[j, 1])) for i, j in enumerate(indices)]


def prepare_confirmation_features(model, video, work, identity):
    """Encode only the union of question windows while preserving chunk boundaries."""
    import math
    import torch
    from decord import VideoReader

    sid = video["sample_id"]
    directory = work / "features" / f"sample_{sid}"
    directory.mkdir(parents=True, exist_ok=True)
    marker, frame_path = directory / "complete.json", directory / "sampled_timestamps.csv"
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved["identity"] == identity and p.digest(frame_path) == saved["frame_times_sha256"]:
            d.verify_feature_files(directory, saved)
            print(f"FEATURE_CACHE reuse sample={sid}", flush=True)
            return directory, saved
        raise d.RunConflict("Existing confirmation feature cache has a different identity")
    reader = VideoReader(video["video_path"], num_threads=1)
    times = plan_sample_grid(reader.get_frame_timestamp(list(range(len(reader)))), video)
    p.write_csv(frame_path, times)
    needed = set()
    for qa in video["conversations"]:
        end = math.ceil(float(qa["end_time"]) * p.FPS)
        needed.update(range(max(0, end - 20), end))
    chunks = []
    for original in p.chunk_plan(video, len(times)):
        group = [i for i in range(original["start"], original["stop"]) if i in needed]
        if not group:
            continue
        if group != list(range(group[0], group[-1] + 1)):
            raise ValueError("Non-contiguous selection inside an original chunk")
        p.report_progress("confirmation_features", sample_id=sid, sampled_start=group[0], sampled_stop=group[-1]+1)
        frames = torch.from_numpy(reader.get_batch([times[i]["source_frame"] for i in group]).asnumpy())
        pixels = model.processor.video_processor(frames.permute(0, 3, 1, 2), return_tensors="pt").pixel_values_videos
        features = model.get_video_features(pixels.to(model.device, model.dtype)).detach().cpu().contiguous()
        target = directory / f"chunk_{len(chunks):04d}.pt"
        temp = target.with_suffix(".tmp")
        torch.save(features, temp)
        os.replace(temp, target)
        chunks.append(dict(qa_index=original["qa_index"], start=group[0], stop=group[-1]+1, tokens=features.shape[1],
                           sha256=p.digest(target), file=target.name))
    saved = dict(identity=identity, sample_id=sid, chunks=chunks, frame_times_sha256=p.digest(frame_path),
                 sampling="0.5 FPS grid with source frame at or before grid timestamp")
    d.verify_feature_files(directory, saved)
    p.atomic_json(marker, saved)
    return directory, saved


def construct_block(model, analyzer, bank, meta, video, qi, history, recent, warmup=False):
    p.sync()
    before = time.perf_counter()
    p.reset(model)
    s.ingest(model, bank, history + recent, meta)
    original = s.snapshot(model)
    start = original["prompt"]
    logical_start = start + 196 * len(history)
    prefix = s.prefix_state(original, logical_start)
    p.sync()
    prep_time = time.perf_counter() - before
    qa = video["conversations"][qi]
    prompt = analyzer.format_mcqa_prompt(qa["question"], qa["choices"])
    sid = video["sample_id"]
    order = list(ARMS)
    rng = random.Random(p.SEED + 31 * sid + qi)
    rng.shuffle(order)
    rows = {}
    for arm in order:
        before = time.perf_counter()
        if arm == ARMS[0]:
            p.reset(model)
            groups = s.ingest(model, bank, recent, meta)
            encoded, stop = s.snapshot(model), start
            chosen = []
        elif not history:
            encoded, groups, stop, chosen = original, s.c.input_groups(recent, "original", meta), start, []
        else:
            chosen = list(range(max(0, len(history) - 2) if arm == ARMS[1] else 0, len(history)))
            encoded, groups, stop = selection.construction(model, bank, meta, prefix, recent, chosen, logical_start)
        p.sync()
        construction_time = time.perf_counter() - before
        before = time.perf_counter()
        state = q.make_short(encoded, start, stop) if stop > start else encoded
        p.sync()
        extraction_time = time.perf_counter() - before if stop > start else 0.0
        if warmup:
            q.position_trace(model, state, prompt, (0, 0))
        answer, answer_time, trace = q.position_trace(model, state, prompt, (0, 0))
        choice = analyzer.extract_characters_regex(answer)
        row = d.question_identity(video, qi, arm)
        row.update(pred_choice=choice, pred_answer=answer.replace("\n", ""), correct=int(choice == row["gold_choice"]),
                   selected_history_indices=[history[i] for i in chosen], recent_indices=recent,
                   recent_groups=groups, query_kv_bytes=q.cache_bytes(state), position_trace=trace,
                   answer_s=answer_time, construction_s=construction_time, extraction_s=extraction_time)
        rows[arm] = row
        model.clear_cache()
    if len({r["query_kv_bytes"] for r in rows.values()}) != 1:
        raise ValueError("Answer K/V budgets differ")
    expected_start = logical_start + 196 * len(recent)
    if any(rows[a]["position_trace"][0]["logical_start"] != expected_start for a in ARMS[1:]):
        raise ValueError("Contextual readout changed recent positions")
    if not history and len({r["pred_choice"] for r in rows.values()}) != 1:
        raise ValueError("Empty-history control differs")
    return [rows[a] for a in ARMS], dict(history_indices=history, recent_indices=recent,
                                      answer_order=order, shared_reference_preparation_s=prep_time)


def paired_interval(deltas, seed=2024, draws=10000):
    """Question-weighted point estimate and video-resampled percentile interval."""
    rng = random.Random(seed)
    videos = list(deltas.values())
    if not videos:
        raise ValueError("No paired videos")
    wins = [(sum(v), len(v)) for v in videos]
    values = []
    for _ in range(draws):
        choices = rng.choices(wins, k=len(wins))
        values.append(100 * sum(x[0] for x in choices) / sum(x[1] for x in choices))
    values.sort()
    total, count = (sum(x[i] for x in wins) for i in (0, 1))
    loo = [100 * (total - x) / (count - n) for x, n in wins if count > n]
    return dict(difference_pp=100 * total / count, ci_low_pp=values[int(.025 * draws)],
                ci_high_pp=values[min(draws - 1, int(.975 * draws))],
                leave_one_video_min_pp=min(loo) if loo else None, leave_one_video_max_pp=max(loo) if loo else None)


def export(rows, work, *, complete):
    p.write_csv(work / "exports/answers.csv", rows)
    groups, summary = {}, []
    for r in rows:
        groups.setdefault((r["sample_id"], r["qa_index"]), {})[r["method"]] = r
    for arm in ARMS:
        subset = [g[arm] for g in groups.values()]
        summary.append(dict(arm=arm, answers=len(subset), correct=sum(r["correct"] for r in subset),
                            median_query_kv_bytes=statistics.median(r["query_kv_bytes"] for r in subset),
                            median_construction_s=statistics.median(r["construction_s"] for r in subset),
                            median_extraction_s=statistics.median(r["extraction_s"] for r in subset),
                            median_answer_s=statistics.median(r["answer_s"] for r in subset)))
    p.write_csv(work / "exports/summary.csv", summary)
    lines = ["# Frozen short query-cache comparison", "", "| Arm | Correct | Answers | Median query KV bytes |",
             "|---|---:|---:|---:|"]
    lines.extend(f"| {r['arm']} | {r['correct']} | {r['answers']} | {r['median_query_kv_bytes']} |" for r in summary)
    lines.extend(["", "Clip-disjoint within StreamingBench. Original source overlap is unverified."])
    if complete:
        deltas = defaultdict(list)
        for (sid, _), g in groups.items():
            deltas[sid].append(g[ARMS[2]]["correct"] - g[ARMS[0]]["correct"])
        effects = paired_interval(deltas)
        values = [v for group in deltas.values() for v in group]
        p.atomic_json(work / "exports/paired_primary.json", dict(
            candidate=ARMS[2], baseline=ARMS[0], wins=values.count(1), losses=values.count(-1),
            videos=len(deltas), questions=len(values), **effects))
        lines.extend(["", f"Primary paired difference: {effects['difference_pp']:.2f} pp "
                      f"(95% video bootstrap {effects['ci_low_pp']:.2f}, {effects['ci_high_pp']:.2f})."])
    (work / "exports/report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(args):
    import torch
    import transformers

    if (args.model.name != p.MODEL_SHA or torch.__version__ != "2.10.0+cu128"
            or transformers.__version__ != "4.45.0.dev0" or not torch.cuda.is_available()):
        raise ValueError("Pinned model/runtime required")
    path = args.work / "reference.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    videos = videos_for_manifest(manifest, args.repo)
    config = dict(manifest_sha256=p.digest(path), sources={n: p.digest(args.work/n) for n in SOURCE_PATHS},
                  model=p.MODEL_SHA, torch=torch.__version__, gpu=torch.cuda.get_device_name(0), arms=ARMS)
    fingerprint = d.fingerprint(config)
    config_path = args.work / "exports/run_config.json"
    if config_path.exists() and json.loads(config_path.read_text())["config_hash"] != fingerprint:
        raise d.RunConflict("Confirmation configuration changed; choose a new WORK")
    p.atomic_json(config_path, dict(**config, config_hash=fingerprint))
    assets = acquire_media(manifest["shards"], args.work)
    for video in videos:
        video["video_path"] = assets[video["sample_id"]]["path"]
    sys.path.insert(0, str(args.repo))
    from inference.llavaov_hermes import load_model
    from video_qa.hermes_vqa import HermesVQA

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(p.SEED)
    p.report_progress("loading_confirmation_model")
    model, processor = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=p.FPS)
    model.eval()
    analyzer = HermesVQA(anno=[], save_dir=str(args.work), sample_fps=p.FPS, qa_model=model, qa_processor=processor)
    rows, started, warmed = [], time.perf_counter(), False
    with torch.inference_mode():
        for i, video in enumerate(videos):
            if i and args.max_hours and time.perf_counter() - started > args.max_hours * 3600:
                break
            sid = video["sample_id"]
            source_sha = assets[sid]["sha256"]
            directory, meta = prepare_confirmation_features(
                model, video, args.work, d.fingerprint(dict(config=fingerprint, source=source_sha)))
            video["timestamps"] = p.read_csv(directory / "sampled_timestamps.csv")
            p.atomic_json(args.work / "provenance" / f"features_{sid}.json", dict(metadata=meta, timestamps=video["timestamps"]))
            bank = s.c.FeatureBank(directory, meta)
            if i == 0:
                p.preflight(model, video, args.work)
            for qi in range(len(video["conversations"])):
                h, r = s.split_frames(video, qi)
                key = d.fingerprint(dict(config=fingerprint, feature=d.fingerprint(meta), sample=sid, qa=qi, h=h, r=r))
                dest = args.work / "checkpoints" / f"sample_{sid}_q{qi}.json"
                if dest.exists():
                    saved = json.loads(dest.read_text())
                    if saved["key"] != key or d.fingerprint(saved["rows"]) != saved["rows_sha256"]:
                        raise d.RunConflict("Confirmation checkpoint differs")
                    block = saved["rows"]
                else:
                    block, prep = construct_block(model, analyzer, bank, meta, video, qi, h, r, warmup=not warmed)
                    warmed = True
                    p.atomic_json(args.work / "preparation" / dest.name, prep)
                    p.atomic_json(dest, dict(key=key, rows=block, rows_sha256=d.fingerprint(block)))
                rows.extend(block)
            del bank
            export(rows, args.work, complete=False)
            print(f"VIDEO_BLOCK_DONE {i+1}/30 questions={len(rows)//3}/150", flush=True)
    status = "complete" if len(rows) == 450 else "partial_time_budget"
    p.atomic_json(args.work / "exports/completion.json", dict(status=status, main_answers=len(rows),
                  main_questions=len(rows)//3, elapsed_post_model_s=time.perf_counter()-started, config_hash=fingerprint))
    export(rows, args.work, complete=status == "complete")
    p.report_progress(status, main_questions=len(rows)//3)
    package(args.work, status)
    print(f"SHORT_QUERY_CONFIRMATION_{status.upper()}", flush=True)


def package(work, status):
    previous = s.ARTIFACT
    try:
        s.ARTIFACT = ARTIFACT
        s.package(work, status)
    finally:
        s.ARTIFACT = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ("repo", "work", "model", "reuse"):
        parser.add_argument("--" + n, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--max-hours", type=float, default=2.0)
    args = parser.parse_args()
    d.validate_workspaces(args.work, args.reuse)
    if args.max_hours < 0:
        parser.error("Invalid soft budget")
    with p.ProgressMonitor(args.work):
        try:
            execute(args)
        except d.RunConflict:
            raise
        except Exception as exc:
            p.atomic_json(args.work / "exports/failure.json", dict(type=type(exc).__name__, detail=str(exc)[-1600:]))
            p.atomic_json(args.work / "exports/completion.json", dict(status="failed"))
            package(args.work, "failed_partial")
            raise


if __name__ == "__main__":
    main()
