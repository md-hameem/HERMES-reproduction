"""Evaluate target/context verification under a fixed four-frame answer budget."""
import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote
from zipfile import BadZipFile, ZipFile

try:
    from provenance import confirmation_base as base
except ModuleNotFoundError as exc:
    if exc.name != "provenance":
        raise
    import autodl_short_query_confirmation as base

p, s, d, q = base.p, base.s, base.d, base.q
ARTIFACT = "HERMES_target_binding_evidence.zip"
SOURCE_PATHS = (*base.SOURCE_PATHS, "provenance/confirmation_base.py")
ARMS = ("recent", "previous", "global_relevance", "target_only", "mean", "conjunction")
PROBES = {
    "relevance": "Do these frames visibly contain information relevant to answering the question?",
    "target": "Is the specific subject named or referred to in the question visually identifiable in these frames?",
    "binding": "Do these frames show the requested action, measurement, or condition for that exact subject? "
               "Choose No if it is shown only for a different subject or setting.",
}
ARM_PROBES = {"recent": (), "previous": (), "global_relevance": ("relevance",), "target_only": ("target",),
              "mean": ("target", "binding"), "conjunction": ("target", "binding")}
TEXT_TASK = "Text-Rich Understanding"


def frozen_videos(manifest, repo):
    raw = (repo / manifest["annotation_path"]).read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(raw).hexdigest() != manifest["annotation_sha256"]:
        raise ValueError("Annotation digest differs from the frozen manifest")
    source = {p.sample_id(v): v for v in json.loads(raw)}
    videos = []
    for shard in manifest["shards"]:
        for clip in shard["clips"]:
            sid = clip["sample_id"]
            video = dict(source[sid], sample_id=sid, shard=shard["shard"])
            questions = video["conversations"]
            if (video["video_id"] != clip["video_id"] or Path(video["video_path"]).name != clip["video_basename"]
                    or len(questions) != clip["question_count"]
                    or sum(d.prefix_end(video, i) >= 8 for i in range(len(questions))) != clip["primary_count"]
                    or any(not 2 <= len(a["choices"]) <= 8 or a["choices"].count(a["answer"]) != 1 for a in questions)):
                raise ValueError(f"Frozen identity differs for sample {sid}")
            videos.append(video)
    ids = {v["sample_id"] for v in videos}
    if len(videos) != manifest["clips"] or len(ids) != len(videos) or ids.intersection(manifest["excluded_diagnostic_sample_ids"]):
        raise ValueError("Frozen cohort has missing or reused clips")
    return videos


def check_media_space(shards, work, reserve=4 * 1024**3):
    """Read ZIP indexes to budget selected members plus a work-space reserve."""
    import requests

    remaining = 0
    for shard in shards:
        receipt = work / "provenance" / "media" / f"{shard['shard']}.json"
        metadata = json.loads(receipt.read_text()) if receipt.exists() else {}
        wanted = {c["sample_id"] for c in shard["clips"]}
        for clip in metadata.get("videos", []):
            path = work / "videos" / f"sample_{clip['sample_id']}_real.mp4"
            if path.is_file() and path.stat().st_size == clip["bytes"]:
                wanted.discard(clip["sample_id"])
        if not wanted:
            continue
        for host in ("huggingface.co", "hf-mirror.com"):
            url = f"https://{host}/datasets/mjuicem/StreamingBench/resolve/main/{quote(shard['archive'])}?download=true"
            try:
                with requests.Session() as session:
                    accessor = p.HTTPRangeFile(url, shard["archive_bytes"], session)
                    with ZipFile(accessor) as archive:
                        members = defaultdict(list)
                        for item in archive.infolist():
                            if item.is_dir() or Path(item.filename).name.startswith("._"):
                                continue
                            match = re.search(r"sample_(\d+)", item.filename)
                            if match and int(match[1]) in wanted and item.filename.lower().endswith(".mp4"):
                                members[int(match[1])].append(item.file_size)
                        if set(members) != wanted or any(len(sizes) != 1 for sizes in members.values()):
                            raise ValueError(f"Missing or ambiguous video names in {shard['shard']}")
                        remaining += sum(sizes[0] for sizes in members.values())
                break
            except (OSError, BadZipFile, p.RangeUnsupported, requests.RequestException) as exc:
                if host == "hf-mirror.com":
                    raise RuntimeError(f"Cannot read ZIP index for {shard['shard']}") from exc
                print(f"MEDIA_INDEX_RETRY {shard['shard']} {type(exc).__name__}", flush=True)
    free = shutil.disk_usage(work).free
    print(f"MEDIA_PLAN remaining_GiB={remaining / 1024**3:.2f} free_GiB={free / 1024**3:.2f}", flush=True)
    if remaining + reserve > free:
        raise RuntimeError("Insufficient free space for selected video members and the 4 GiB work-space reserve")


def candidate_windows(history, recent):
    """Return [newest, ...] complete disjoint windows. Keep short starts as recent-only."""
    if not recent:
        raise ValueError("No causal frame at this question time")
    return [recent] + [history[end-4:end] for end in range(len(history), 3, -4)]


def select(scores):
    """Ties go to the newest window. All selectors share the candidate pool."""
    if len(scores) == 1:
        return dict.fromkeys(ARMS, 0)
    if not scores or any(not math.isfinite(row[k]) for row in scores for k in PROBES):
        raise ValueError("Each selectable window requires finite scores")
    relevance = [row["relevance"] for row in scores]
    target = [row["target"] for row in scores]
    mean = [(row["target"] + row["binding"]) / 2 for row in scores]
    conjunction = [min(row["target"], row["binding"]) for row in scores]
    return dict(zip(ARMS, (0, 1, relevance.index(max(relevance)), target.index(max(target)),
                           mean.index(max(mean)), conjunction.index(max(conjunction))), strict=True))


def probe_window(model, state, question, criterion):
    """Average signed Yes-minus-No logit margins after swapping the answer letters."""
    tokenizer = model.processor.tokenizer
    label_ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in ("A", "B")]
    if any(len(ids) != 1 for ids in label_ids) or label_ids[0] == label_ids[1]:
        raise ValueError("A/B must map to distinct single tokens")
    margins = []
    p.sync()
    started = time.perf_counter()
    for yes_first in (True, False):
        first, second = ("Yes", "No") if yes_first else ("No", "Yes")
        text = model.get_prompt(
            f"Question about the video: {question}\nVerification: {PROBES[criterion]}\n"
            f"Options:\n(A) {first}\n(B) {second}\nReply with only the letter A or B.") + "Answer letter:\n"
        ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        prefix_ids = ids[0].tolist()
        for letter, suffix in zip(("A", "B"), label_ids, strict=True):
            if tokenizer(text + letter).input_ids != prefix_ids + suffix:
                raise ValueError(f"Tokenizer merges the probe continuation {letter}")
        logits, _, _ = s.probe(model, state, ids, 0, 0, steps=0)
        a, b = logits[0][0, -1, [label_ids[0][0], label_ids[1][0]]].float().tolist()
        margins.append(a - b if yes_first else b - a)
        s.assert_state(model, state)
    p.sync()
    if not all(math.isfinite(value) for value in margins):
        raise ValueError("Nonfinite probe margin")
    return statistics.mean(margins), margins, time.perf_counter() - started


def evaluate_question(model, analyzer, bank, meta, video, qi):
    qa = video["conversations"][qi]
    history, recent = s.split_frames(video, qi)
    windows = candidate_windows(history, recent)
    order = list(range(len(windows)))
    random.Random(p.SEED + 31 * video["sample_id"] + qi).shuffle(order)
    scores, candidates = [{} for _ in windows], [{} for _ in windows]
    for i in order:
        indices = windows[i]
        p.sync()
        started = time.perf_counter()
        p.reset(model)
        groups = s.ingest(model, bank, indices, meta)
        state = s.snapshot(model)
        d.assert_fresh_cache(model, len(indices))
        p.sync()
        construction = time.perf_counter() - started
        raw, probe_times = {}, {}
        if len(windows) > 1:
            for criterion in PROBES:
                scores[i][criterion], raw[criterion], probe_times[criterion] = probe_window(
                    model, state, qa["question"], criterion)
        prompt = analyzer.format_mcqa_prompt(qa["question"], qa["choices"])
        answer, answer_time, trace = q.position_trace(model, state, prompt, (0, 0))
        candidates[i] = dict(window=i, indices=indices, groups=groups, scores=scores[i], order_margins=raw,
                             query_kv_bytes=q.cache_bytes(state), position_trace=trace,
                             construction_s=construction, probe_s=probe_times, answer_s=answer_time,
                             pred_choice=analyzer.extract_characters_regex(answer), pred_answer=answer)
        del state
        model.clear_cache()
    chosen = select(scores)
    if len({(c["query_kv_bytes"], c["position_trace"][0]["logical_start"]) for c in candidates}) != 1:
        raise ValueError("Candidate answer cache sizes or starting positions differ")
    identity = d.question_identity(video, qi, "window")
    for c in candidates:
        c["correct"] = int(c["pred_choice"] == identity["gold_choice"])
    rows = []
    for arm in ARMS:
        i = chosen[arm]
        c = candidates[i]
        scanned = candidates if ARM_PROBES[arm] and len(windows) > 1 else [c]
        component_s = (sum(x["construction_s"] for x in scanned)
                       + sum(x["probe_s"].get(probe, 0.0) for x in scanned for probe in ARM_PROBES[arm])
                       + c["answer_s"])
        rows.append(dict(identity, method=arm, shard=video["shard"], eligible=len(windows) > 1,
                         window=i, indices=c["indices"], pred_choice=c["pred_choice"], correct=c["correct"],
                         query_kv_bytes=c["query_kv_bytes"], component_s=component_s))
    return dict(rows=rows, windows=candidates, execution_order=order, oracle_correct=max(c["correct"] for c in candidates))


def compare(blocks, candidate, baseline):
    deltas = defaultdict(list)
    wins = losses = switches = 0
    for block in blocks:
        rows = {r["method"]: r for r in block["rows"]}
        a, b = rows[candidate], rows[baseline]
        delta = a["correct"] - b["correct"]
        deltas[a["sample_id"]].append(delta)
        wins += delta == 1
        losses += delta == -1
        switches += a["window"] != b["window"]
    return dict(candidate=candidate, baseline=baseline, wins=wins, losses=losses, window_changes=switches,
                questions=len(blocks), videos=len(deltas), **base.paired_interval(deltas))


def write_blind_review(blocks, work):
    """Queue differing selections with selector and correctness kept in a separate key."""
    rng = random.Random(p.SEED)
    review, key = [], []
    for block in blocks:
        rows = {r["method"]: r for r in block["rows"]}
        a, b = rows["conjunction"], rows["mean"]
        if a["window"] == b["window"]:
            continue
        pair = [a, b]
        rng.shuffle(pair)
        rid = f"{a['sample_id']}-q{a['qa_index']}"
        review.append(dict(review_id=rid, sample_id=a["sample_id"], qa_index=a["qa_index"], question=a["question"],
                           query_time_s=a["query_time_s"], left_indices=pair[0]["indices"], right_indices=pair[1]["indices"],
                           left_support="", right_support="", notes=""))
        key.append(dict(review_id=rid, left_method=pair[0]["method"], right_method=pair[1]["method"],
                        left_correct=pair[0]["correct"], right_correct=pair[1]["correct"]))
    if review:
        p.write_csv(work / "exports/blind_review.csv", review)
        p.write_csv(work / "exports/blind_review_key.csv", key)


def export(blocks, work, complete):
    rows = [row for block in blocks for row in block["rows"]]
    p.write_csv(work / "exports/answers.csv", rows)
    primary = [block for block in blocks if block["rows"][0]["eligible"]]
    lines = ["# Target binding under a fixed answer budget", "",
             "Within StreamingBench. Original source overlap is unverified.", "",
             "| Cohort | Selector | Correct / questions | Median component ms | p95 component ms |",
             "|---|---|---:|---:|---:|"]
    summary = []
    cohorts = {"causal_two_or_more_windows": primary, "all_questions": blocks,
               "text_rich_two_or_more_windows": [b for b in primary if b["rows"][0]["task"] == TEXT_TASK]}
    for name, cohort in cohorts.items():
        if not cohort:
            continue
        for i, arm in enumerate(ARMS):
            subset = [b["rows"][i] for b in cohort]
            times = sorted(r["component_s"] * 1000 for r in subset)
            item = dict(cohort=name, arm=arm, questions=len(subset), correct=sum(r["correct"] for r in subset),
                        older_selected=sum(r["window"] > 0 for r in subset), median_component_ms=statistics.median(times),
                        p95_component_ms=times[math.ceil(.95 * len(times)) - 1],
                        oracle_correct=sum(b["oracle_correct"] for b in cohort))
            summary.append(item)
            lines.append(f"| {name} | {arm} | {item['correct']} / {item['questions']} | "
                         f"{item['median_component_ms']:.1f} | {item['p95_component_ms']:.1f} |")
    p.write_csv(work / "exports/summary.csv", summary)
    lines.extend(["", "Component times sum serialized cache builds, required probes, and one answer from the shared replay.",
                  "They exclude frame decoding, feature extraction and stream maintenance. The oracle is diagnostic only."])
    if complete:
        effects = [compare(primary, "conjunction", baseline) for baseline in
                   ("mean", "recent", "global_relevance", "target_only", "previous")]
        p.atomic_json(work / "exports/paired.json", dict(primary=effects[0], secondary=effects[1:]))
        for result in effects:
            lines.append(f"\nConjunction vs {result['baseline']}: {result['difference_pp']:+.2f} pp, "
                         f"95% video bootstrap [{result['ci_low_pp']:+.2f}, {result['ci_high_pp']:+.2f}], "
                         f"{result['wins']} wins and {result['losses']} losses.")
        write_blind_review(primary, work)
    (work / "exports/report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(args):
    import torch
    import transformers

    if (args.model.name != p.MODEL_SHA or torch.__version__ != "2.10.0+cu128"
            or transformers.__version__ != "4.45.0.dev0" or not torch.cuda.is_available()):
        raise ValueError("Pinned GPU model/runtime required")
    path = args.work / "reference.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    videos = frozen_videos(manifest, args.repo)
    config = dict(manifest_sha256=p.digest(path), sources={n: p.digest(args.work / n) for n in SOURCE_PATHS},
                  model=p.MODEL_SHA, torch=torch.__version__, gpu=torch.cuda.get_device_name(0), arms=ARMS)
    fingerprint = d.fingerprint(config)
    config_path = args.work / "exports/run_config.json"
    if config_path.exists() and json.loads(config_path.read_text())["config_hash"] != fingerprint:
        raise d.RunConflict("Configuration changed. Preserve this run and use a new WORK directory.")
    p.atomic_json(config_path, dict(**config, config_hash=fingerprint))
    p.report_progress("indexing_selected_media")
    check_media_space(manifest["shards"], args.work)
    assets = base.acquire_media(manifest["shards"], args.work)
    for video in videos:
        video["video_path"] = assets[video["sample_id"]]["path"]
    sys.path.insert(0, str(args.repo))
    from inference.llavaov_hermes import load_model
    from video_qa.hermes_vqa import HermesVQA

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(p.SEED)
    p.report_progress("loading_target_binding_model")
    model, processor = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=p.FPS)
    model.eval()
    analyzer = HermesVQA(anno=[], save_dir=str(args.work), sample_fps=p.FPS, qa_model=model, qa_processor=processor)
    blocks, started, warmed = [], time.perf_counter(), False
    with torch.inference_mode():
        p.preflight(model, max(videos, key=lambda v: float(v["conversations"][-1]["end_time"])), args.work)
        for i, video in enumerate(videos):
            if i and args.max_hours and time.perf_counter() - started > args.max_hours * 3600:
                break
            sid = video["sample_id"]
            directory, meta = base.prepare_confirmation_features(
                model, video, args.work, d.fingerprint(dict(config=fingerprint, source=assets[sid]["sha256"])))
            video["timestamps"] = p.read_csv(directory / "sampled_timestamps.csv")
            p.atomic_json(args.work / "provenance" / f"features_{sid}.json", dict(metadata=meta, timestamps=video["timestamps"]))
            bank = s.c.FeatureBank(directory, meta)
            for qi in range(len(video["conversations"])):
                dest = args.work / "checkpoints" / f"sample_{sid}_q{qi}.json"
                key = d.fingerprint(dict(config=fingerprint, features=d.fingerprint(meta), sample=sid, question=qi))
                if dest.exists():
                    saved = json.loads(dest.read_text(encoding="utf-8"))
                    if saved["key"] != key or d.fingerprint(saved["block"]) != saved["block_sha256"]:
                        raise d.RunConflict(f"Checkpoint differs: {dest}")
                    block = saved["block"]
                else:
                    p.report_progress("target_binding_question", sample_id=sid, qa_index=qi, completed=len(blocks))
                    if not warmed and d.prefix_end(video, qi) >= 8:
                        evaluate_question(model, analyzer, bank, meta, video, qi)
                        warmed = True
                    block = evaluate_question(model, analyzer, bank, meta, video, qi)
                    p.atomic_json(dest, dict(key=key, block=block, block_sha256=d.fingerprint(block)))
                blocks.append(block)
            del bank
            export(blocks, args.work, complete=False)
            print(f"VIDEO_BLOCK_DONE {i+1}/{len(videos)} questions={len(blocks)}/{manifest['questions']}", flush=True)
    complete = len(blocks) == manifest["questions"]
    if complete and sum(b["rows"][0]["eligible"] for b in blocks) != manifest["primary_questions"]:
        raise ValueError("Primary cohort count differs from the frozen manifest")
    status = "complete" if complete else "partial_time_budget"
    export(blocks, args.work, complete)
    p.atomic_json(args.work / "exports/completion.json", dict(status=status, main_answers=len(blocks) * len(ARMS),
                  main_questions=len(blocks), elapsed_post_model_s=time.perf_counter() - started, config_hash=fingerprint))
    p.report_progress(status, main_questions=len(blocks))
    package(args.work, status)
    print(f"TARGET_BINDING_{status.upper()}", flush=True)


def package(work, status):
    previous = s.ARTIFACT
    try:
        s.ARTIFACT = ARTIFACT
        s.package(work, status)
    finally:
        s.ARTIFACT = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "work", "model", "reuse"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--max-hours", type=float, default=0.0)
    args = parser.parse_args()
    d.validate_workspaces(args.work, args.reuse)
    if not math.isfinite(args.max_hours) or args.max_hours < 0:
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
