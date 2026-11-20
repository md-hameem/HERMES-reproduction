"""Matched fresh-state input controls. Heavy libraries are imported only at runtime.

The companion notebook embeds this runner, the tested pilot utilities and the
audited historical reference. Historical artifacts are data, never executed.
"""
import argparse
import copy
import hashlib
import html
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

try:
    import diagnostic_pilot_utils as p
except ModuleNotFoundError as exc:
    if exc.name != "diagnostic_pilot_utils":
        raise
    import autodl_failure_pilot as p

METHODS = ("question_only", "recent_4", "recent_20", "native_prefix", "support_oracle")
ARTIFACT = "HERMES_input_diagnostic_evidence.zip"
PRIORITY = {(204, 0), (202, 0), (208, 0), (212, 3), (218, 3), (204, 3), (219, 3)}


class RunConflict(ValueError):
    """Existing scientific outputs must not be replaced by a different run."""


def fingerprint(value):
    return hashlib.sha256(p.stable(value).encode()).hexdigest()


def validate_workspaces(work, reuse):
    work, reuse = Path(work).resolve(), Path(reuse).resolve()
    if work == reuse or work.is_relative_to(reuse) or reuse.is_relative_to(work):
        raise ValueError("WORK and reusable pilot directory must be separate non-nested locations")


def check_reference(reference, repo):
    for name, expected in reference["source_sha256"].items():
        actual = hashlib.sha256((repo / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise ValueError(f"Pinned source differs: {name}")
    annotation = json.loads((repo / "data/streamingbench/streamingbench_realtime.json").read_text(encoding="utf-8"))
    by_id = {str(v["video_id"]): v for v in annotation}
    for video in reference["videos"]:
        original = by_id[str(video["video_id"])]
        if original["conversations"] != video["conversations"] or p.sample_id(original) != video["sample_id"]:
            raise ValueError("Historical annotation identity mismatch")


def prefix_end(video, qi):
    return math.ceil(float(video["conversations"][qi]["end_time"]) * p.FPS)


def validate_frames(indices, video, qi, timestamps):
    end = prefix_end(video, qi)
    if any(type(i) is not int for i in indices) or indices != sorted(set(indices)):
        raise ValueError("Frame indices must be sorted unique integers")
    if any(i < 0 or i >= end or i >= len(timestamps) for i in indices):
        raise ValueError("Selected frame outside original query prefix")
    query = float(video["conversations"][qi]["end_time"])
    if any(float(timestamps[i]["start_s"]) > query + 1e-6 for i in indices):
        raise ValueError("Sampled frame starts after query time")
    return indices


def validate_oracles(items, videos, max_frames=20):
    if not isinstance(items, list):
        raise ValueError("Support JSON must be a list")
    by_id = {v["sample_id"]: v for v in videos}
    result = {}
    for item in items:
        sid, qi = item["sample_id"], item["qa_index"]
        if type(sid) is not int or type(qi) is not int or sid not in by_id:
            raise ValueError("Unknown oracle sample or invalid index type")
        video = by_id[sid]
        if not 0 <= qi < len(video["conversations"]) or (sid, qi) in result:
            raise ValueError("Invalid/duplicate oracle question")
        if item.get("verified") is not True or not str(item.get("notes", "")).strip():
            raise ValueError("Oracle requires verified=true and support review notes")
        indices = item["sampled_indices"]
        if not isinstance(indices, list) or not 1 <= len(indices) <= max_frames:
            raise ValueError("Oracle must contain 1..20 sampled frame indices")
        validate_frames(indices, video, qi, video["timestamps"])
        result[(sid, qi)] = item
    return result


def selected_frames(method, video, qi, oracles, native_enabled):
    end = prefix_end(video, qi)
    if method == "question_only":
        return [], None
    if method in ("recent_4", "recent_20"):
        count = 4 if method == "recent_4" else 20
        return list(range(max(0, end - count), end)), None
    if method == "native_prefix":
        return (list(range(end)), None) if native_enabled else ([], "disabled")
    oracle = oracles.get((video["sample_id"], qi))
    return (oracle["sampled_indices"], None) if oracle else ([], "not_requested")


def context_reason(frame_count, prompt_tokens, question_tokens, context_limit, native_cap, method):
    # question_answering uses greedy max_new_tokens=16; 32 extra positions are a safety reserve.
    needed = prompt_tokens + frame_count * 196 + question_tokens + 16 + 32
    if needed > context_limit:
        return "context_limit"
    if method == "native_prefix" and frame_count * 196 > native_cap:
        return "configured_native_cap"
    return None


def feature_slices(indices, chunks):
    """Return chunk-local runs without mixing or duplicating original frame identities."""
    covered, slices = [], []
    for item in chunks:
        chosen = [i for i in indices if item["start"] <= i < item["stop"]]
        for i in chosen:
            covered.append(i)
            if slices and slices[-1]["file"] == item["file"] and slices[-1]["stop"] == i - item["start"]:
                slices[-1]["stop"] += 1
            else:
                slices.append(dict(file=item["file"], start=i-item["start"], stop=i-item["start"]+1))
    if covered != indices:
        raise ValueError("Feature chunks do not cover selected frames exactly once")
    return slices


def read_features(directory, item):
    import torch
    value = torch.load(directory / item["file"], map_location="cpu", weights_only=True)
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] % 196 or value.dtype != torch.float16:
        raise ValueError("Unexpected feature layout/dtype; never guess frame boundaries")
    return value[:, item["start"]*196:item["stop"]*196].contiguous()


def acquire_videos(videos, reuse, work):
    # Only exact trusted historical video hashes can seed resumption/reuse.
    entries, missing = [], []
    for v in videos:
        sid = v["sample_id"]
        candidates = [work / "videos" / f"sample_{sid}_real.mp4", reuse / "videos" / f"sample_{sid}_real.mp4"]
        found = next((x for x in candidates if x.is_file() and p.digest(x) == v["video_sha256"]), None)
        if found is None:
            missing.append(v)
        else:
            entries.append(dict(sample_id=sid, path=str(found.resolve()), bytes=found.stat().st_size,
                                sha256=v["video_sha256"]))
    if not missing:
        assets = dict(mode="verified_existing_videos", videos=entries)
    else:
        downloaded = p.prepare_assets(missing, work)
        assets = dict(mode="verified_existing_plus_downloaded", videos=entries+downloaded["videos"],
                      download_details={k:v for k,v in downloaded.items() if k != "videos"})
    by_id = {x["sample_id"]: x for x in assets["videos"]}
    for v in videos:
        item = by_id[v["sample_id"]]
        if item["sha256"] != v["video_sha256"] or p.digest(item["path"]) != v["video_sha256"]:
            raise ValueError("Video differs from audited source")
        v["video_path"] = item["path"]
    p.atomic_json(work / "asset_manifest.json", assets)


def verify_feature_files(directory, meta):
    for c in meta["chunks"]:
        if Path(c["file"]).name != c["file"] or c["tokens"] != (c["stop"]-c["start"])*196:
            raise ValueError("Invalid feature chunk metadata")
        path = directory / c["file"]
        if not path.is_file() or p.digest(path) != c["sha256"]:
            raise ValueError(f"Missing/corrupt feature: {path}")


def acquire_features(model, video, reuse, work, identity):
    sid = video["sample_id"]
    old_dir = reuse / "features" / f"sample_{sid}"
    expected = video["feature_meta"]
    if (old_dir / "complete.json").is_file():
        try:
            old = json.loads((old_dir / "complete.json").read_text(encoding="utf-8"))
            if old != expected or p.read_csv(old_dir / "sampled_timestamps.csv") != video["timestamps"]:
                raise ValueError("Old feature metadata differs from audited archive")
            verify_feature_files(old_dir, expected)
            meta, directory, origin = old, old_dir, "verified_historical_cache"
            print(f"FEATURE_REUSE_VERIFIED sample={sid}", flush=True)
        except (OSError, ValueError, KeyError) as exc:
            print(f"FEATURE_REUSE_REJECTED sample={sid} reason={type(exc).__name__}; regenerating separately", flush=True)
            directory = None
    else:
        directory = None
    if directory is None:
        directory = work / "features" / f"sample_{sid}"
        meta = p.prepare_features(model, video, work, identity)
        verify_feature_files(directory, meta)
        origin = "diagnostic_cache_verified"
        if p.read_csv(directory / "sampled_timestamps.csv") != video["timestamps"]:
            raise ValueError("Regenerated source sampling differs from original pilot")
        original_layout = [{k:c[k] for k in ("qa_index", "start", "stop", "tokens")} for c in expected["chunks"]]
        if [{k:c[k] for k in ("qa_index", "start", "stop", "tokens")} for c in meta["chunks"]] != original_layout:
            raise ValueError("Regenerated feature chunk boundaries differ")
    saved = dict(origin=origin, metadata=meta, directory=str(directory.resolve()),
                 timestamp_rows=video["timestamps"], feature_fingerprint=fingerprint(meta))
    p.atomic_json(work / "provenance" / f"features_{sid}.json", saved)
    return directory, meta, saved


def question_identity(video, qi, method):
    qa = video["conversations"][qi]
    return dict(sample_id=video["sample_id"], video_id=str(video["video_id"]), qa_index=qi,
                method=method, question=qa["question"], choices=qa["choices"],
                gold_answer=qa["answer"], gold_choice="ABCDEFGH"[qa["choices"].index(qa["answer"])],
                query_time_s=qa["end_time"], task=qa.get("task", ""))


def checkpoint_valid(value, expected, identity, frames):
    try:
        if value["checkpoint_identity"] != expected or value["row_sha256"] != fingerprint(value["row"]):
            return False
        row = value["row"]
        if any(row[k] != v for k,v in identity.items()) or row["sampled_indices"] != frames:
            return False
        if row["status"] == "answered":
            return row["correct"] == int(row["pred_choice"] == row["gold_choice"])
        return row["status"] == "skipped" and bool(row["skip_reason"]) and row["correct"] is None
    except (KeyError, TypeError, ValueError):
        return False


def assert_fresh_cache(model, frames):
    import torch
    expected = model.visual_start_idx + frames * 196
    if model.conv_history or model._get_cache_seq_len_per_layer() != [expected]*model.num_layers:
        raise ValueError("Fresh control retained history or has an unexpected cache length")
    for positions in model._position_ids_cache:
        torch.testing.assert_close(positions, torch.arange(expected, device=model.device), atol=0, rtol=0)


def run_question(model, analyzer, video, qi, method, directory, meta, config_hash, oracles, args):
    import torch
    identity = question_identity(video, qi, method)
    frames, reason = selected_frames(method, video, qi, oracles, args.native)
    validate_frames(frames, video, qi, video["timestamps"])
    check_id = fingerprint(dict(config=config_hash, feature=fingerprint(meta), question=identity, frames=frames))
    dest = args.work / "checkpoints" / method / f"sample_{video['sample_id']}_q{qi}.json"
    if dest.is_file():
        value = json.loads(dest.read_text(encoding="utf-8"))
        if not checkpoint_valid(value, check_id, identity, frames):
            raise RunConflict(f"Existing checkpoint mismatched/corrupt; preserve and use new WORK: {dest}")
        print(f"REUSE {method} sample={video['sample_id']} q={qi}", flush=True)
        return value["row"]
    p.report_progress("question", sample_id=video["sample_id"], qa_index=qi, method=method, selected_frames=len(frames))
    row = dict(**identity, status="skipped", skip_reason=reason or "", sampled_indices=frames,
               source_frame_indices=[int(video["timestamps"][i]["source_frame"]) for i in frames],
               sampled_start_seconds=[float(video["timestamps"][i]["start_s"]) for i in frames],
               pred_answer="", pred_choice="", correct=None, visual_tokens=len(frames)*196,
               ingest_s=0.0, answer_s=0.0, wall_s=0.0, peak_allocated_gib=0.0,
               fresh_state=True, no_compression=True, feature_fingerprint=fingerprint(meta))
    start = time.perf_counter()
    if not reason:
        random.seed(p.SEED)
        torch.manual_seed(p.SEED)
        p.reset(model)
        assert_fresh_cache(model, 0)
        prompt = analyzer.format_mcqa_prompt(identity["question"], identity["choices"])
        qlen = len(model.processor.tokenizer(prompt["prompt"]).input_ids)
        reason = context_reason(len(frames), model.visual_start_idx, qlen,
                                model.language_model.config.max_position_embeddings, args.native_max_tokens, method)
        if not reason and method == "native_prefix":
            # Conservative KV estimate plus 2 GiB workspace; hard cap remains separate.
            base_bytes = sum(k.numel()*k.element_size()+v.numel()*v.element_size() for k,v in model.kv_cache)
            kv_estimate = base_bytes / model.visual_start_idx * (len(frames)*196+qlen+48)
            free, _ = torch.cuda.mem_get_info()
            if free < kv_estimate*1.5 + 2*1024**3:
                reason = "estimated_vram_limit"
        if reason:
            row["skip_reason"] = reason
        else:
            torch.cuda.reset_peak_memory_stats()
            try:
                p.sync()
                before = time.perf_counter()
                for item in feature_slices(frames, meta["chunks"]):
                    value = read_features(directory, item)
                    p.ingest_features(model, value, item["stop"]-item["start"])
                    del value
                p.sync()
                row["ingest_s"] = time.perf_counter()-before
                assert_fresh_cache(model, len(frames))
                row["cache_lengths_before_answer"] = model._get_cache_seq_len_per_layer()
                before = time.perf_counter()
                # Gold is used only AFTER generation. Same prompt/parser/16-token greedy policy as pilot.
                answer = model.question_answering(prompt, max_new_tokens=16)
                p.sync()
                pred = analyzer.extract_characters_regex(answer)
                row.update(status="answered", pred_answer=answer.replace("\n", ""), pred_choice=pred,
                           correct=int(pred == identity["gold_choice"]), answer_s=time.perf_counter()-before,
                           peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
            except torch.cuda.OutOfMemoryError:
                if method != "native_prefix":
                    raise
                row.update(status="skipped", skip_reason="cuda_oom", correct=None)
                print("NATIVE_PREFIX_OOM: case skipped explicitly; other controls continue", flush=True)
            finally:
                p.reset(model)
        # Release even a cap/context-skipped fresh state.
        model.clear_cache()
    row["wall_s"] = time.perf_counter()-start
    value = dict(checkpoint_identity=check_id, row=row, row_sha256=fingerprint(row))
    if not checkpoint_valid(value, check_id, identity, frames):
        raise ValueError("New checkpoint failed identity validation")
    p.atomic_json(dest, value)
    print(f"QUESTION_DONE {method} sample={video['sample_id']} q={qi} status={row['status']} correct={row['correct']} reason={row['skip_reason']}", flush=True)
    return row


def export(rows, reference, work):
    fields = list(question_identity(reference["videos"][0], 0, "question_only")) + [
        "status", "skip_reason", "sampled_indices", "source_frame_indices", "sampled_start_seconds", "pred_answer", "pred_choice", "correct",
        "visual_tokens", "ingest_s", "answer_s", "wall_s", "peak_allocated_gib", "fresh_state", "no_compression", "feature_fingerprint", "cache_lengths_before_answer"]
    p.write_csv(work / "exports/predictions.csv", [{k:r.get(k, "") for k in fields} for r in rows])
    summary = []
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        answered = [r for r in subset if r["status"] == "answered"]
        summary.append(dict(method=method, planned=40, recorded=len(subset), answered=len(answered),
                            skipped=len(subset)-len(answered), correct=sum(r["correct"] for r in answered),
                            accuracy_pct=100*sum(r["correct"] for r in answered)/len(answered) if answered else "",
                            measured_wall_s=sum(r["wall_s"] for r in answered)))
    p.write_csv(work / "exports/summary.csv", summary)
    old = {(int(r["sample_id"]), int(r["qa_index"]), r["method"]): r for r in reference["predictions"]}
    paired = []
    for method in METHODS:
        new = [r for r in rows if r["method"] == method and r["status"] == "answered"]
        for baseline in ("fifo", "hermes", "causal"):
            pairs = [(r, old[(r["sample_id"], r["qa_index"], baseline)]) for r in new]
            paired.append(dict(method=method, historical_baseline=baseline, matched_questions=len(pairs),
                diagnostic_correct=sum(r["correct"] for r,b in pairs),
                historical_correct=sum(int(b["correct"]) for r,b in pairs),
                diagnostic_only_correct=sum(r["correct"] and not int(b["correct"]) for r,b in pairs),
                historical_only_correct=sum(not r["correct"] and int(b["correct"]) for r,b in pairs),
                comparison_scope="Exact shared identities; historical session; changed input/history; descriptive development comparison"))
    p.write_csv(work / "exports/paired_historical.csv", paired)
    text = ["# Fresh-state input diagnostic", "", "Selected development cases; no population or original-method claim.", "",
            "| Method | Answered | Skipped | Correct | Accuracy on answered |", "|---|---:|---:|---:|---:|"]
    for s in summary:
        accuracy = f"{s['accuracy_pct']:.1f}%" if s["answered"] else "N/A"
        text.append(f"| {s['method']} | {s['answered']} | {s['skipped']} | {s['correct']} | {accuracy} |")
    text += ["", "Native-prefix/oracle rows may cover different subsets: use paired_historical.csv for exact-subset comparisons.",
        "All controls reset per question; no eviction or past-answer history. Recent/oracle inputs compact selected frames chronologically.",
        "Historical comparisons reuse old predictions; they do not establish same-session efficiency or isolate memory policy alone.",
        "Oracle is privileged human support, not a deployable method. No oracle runs without verified annotations.",
        "Question wall time includes reset/feature reading/ingestion/answer; excludes setup and shared feature preparation.",
        "Review sheets show original sampled frames, not proof of support annotation. Source videos and feature tensors are excluded from ZIP."]
    (work / "exports/report.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    return summary


def initialize_review(videos, work):
    sheet = work / "review/support_review.csv"
    if not sheet.exists():
        rows = []
        for video in videos:
            for qi, qa in enumerate(video["conversations"]):
                rows.append(dict(sample_id=video["sample_id"], qa_index=qi,
                    priority="first" if (video["sample_id"], qi) in PRIORITY else "later",
                    question=qa["question"], query_time_s=qa["end_time"], review_status="pending",
                    support_intervals_seconds_json="", verified_sampled_indices_json="",
                    sampled_support_sufficient="", ambiguity="", cause="unclassified", notes=""))
        p.write_csv(sheet, rows)
    content = ["<!doctype html><meta charset='utf-8'><title>Priority support review</title>",
               "<h1>Original sampled prefixes: support still needs review</h1>",
               "<p>Indices are sampled-frame indices. All pages stop at the original query prefix. "
               "Gold answers are review labels, never provided as model inputs. Fill support_review.csv separately.</p>"]
    for video in videos:
        for qi, qa in enumerate(video["conversations"]):
            if (video["sample_id"], qi) not in PRIORITY:
                continue
            content += [f"<h2>sample {video['sample_id']} / q{qi}</h2>",
                        f"<p>{html.escape(qa['question'])} | query {qa['end_time']} s</p>",
                        f"<p>Choices: {html.escape(str(qa['choices']))} | gold: {html.escape(str(qa['answer']))}</p>"]
            for page in range(math.ceil(prefix_end(video, qi)/12)):
                name = f"sample_{video['sample_id']}_q{qi}/page_{page+1:02d}.jpg"
                content.append(f"<p><a href='{name}'>Page {page+1}</a></p>")
    (work/"review/index.html").write_text("\n".join(content), encoding="utf-8")


def review_pages(video, work):
    from decord import VideoReader
    from PIL import Image, ImageDraw
    reader = None
    for qi, qa in enumerate(video["conversations"]):
        if (video["sample_id"], qi) not in PRIORITY:
            continue
        directory = work / "review" / f"sample_{video['sample_id']}_q{qi}"
        expected = math.ceil(prefix_end(video, qi)/12)
        if all((directory/f"page_{i+1:02d}.jpg").is_file() for i in range(expected)):
            continue
        directory.mkdir(parents=True, exist_ok=True)
        if reader is None:
            reader = VideoReader(video["video_path"], num_threads=1)
        for start in range(0, prefix_end(video, qi), 12):
            p.report_progress("priority_review_frames", sample_id=video["sample_id"], qa_index=qi, sampled_start=start)
            indices = list(range(start, min(start+12, prefix_end(video, qi))))
            source = [int(video["timestamps"][i]["source_frame"]) for i in indices]
            frames = reader.get_batch(source).asnumpy()
            canvas = Image.new("RGB", (1024, 520), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((4, 4), f"sample={video['sample_id']} q={qi} query={qa['end_time']}s | ORIGINAL SAMPLED PREFIX | support unreviewed", fill="black")
            for n, (frame, i) in enumerate(zip(frames, indices)):
                image = Image.fromarray(frame)
                image.thumbnail((256, 140))
                x, y = n%4*256, 25+n//4*164
                canvas.paste(image, (x,y))
                draw.text((x+3,y+143), f"sampled {i} | {video['timestamps'][i]['start_s']}s", fill="black")
            canvas.save(directory/f"page_{start//12+1:02d}.jpg", quality=85)


def package(work, status):
    captured = work / "captured_logs"
    captured.mkdir(parents=True, exist_ok=True)
    for path in (work / "logs").glob("*"):
        if path.is_file():
            shutil.copy2(path, captured/path.name)
    if (work/"progress.json").is_file():
        shutil.copy2(work/"progress.json", captured/"progress_snapshot.json")
    paths = []
    for area in ("exports", "checkpoints", "provenance", "review", "captured_logs"):
        paths += [x for x in (work/area).rglob("*") if x.is_file() and ".ipynb_checkpoints" not in x.parts]
    for name in ("diagnostic_runner.py", "diagnostic_pilot_utils.py", "reference.json", "launch.json",
                 "asset_manifest.json", "preflight.json", "support_oracle.json"):
        if (work/name).is_file():
            paths.append(work/name)
    manifest = [dict(path=x.relative_to(work).as_posix(), bytes=x.stat().st_size, sha256=p.digest(x)) for x in sorted(set(paths))]
    p.atomic_json(work/"evidence_manifest.json", dict(status=status, files=manifest))
    output = work / ARTIFACT
    with ZipFile(output.with_suffix(".tmp"), "w", compression=ZIP_DEFLATED) as z:
        for item in manifest:
            z.write(work/item["path"], item["path"])
        z.write(work/"evidence_manifest.json", "evidence_manifest.json")
    os.replace(output.with_suffix(".tmp"), output)
    print(f"EVIDENCE_ZIP={output}\nSHA256={p.digest(output)}", flush=True)


def execute(args):
    import numpy as np
    import torch
    import transformers
    args.work, args.repo, args.reuse = args.work.resolve(), args.repo.resolve(), args.reuse.resolve()
    validate_workspaces(args.work, args.reuse)
    if args.model.name != p.MODEL_SHA:
        raise ValueError("Model snapshot revision differs")
    reference = json.loads((args.work / "reference.json").read_text(encoding="utf-8"))
    check_reference(reference, args.repo)
    videos = copy.deepcopy(reference["videos"])
    initialize_review(videos, args.work)
    oracles = validate_oracles(json.loads((args.work/"support_oracle.json").read_text(encoding="utf-8")), videos)
    if torch.__version__ != "2.10.0+cu128" or transformers.__version__ != "4.45.0.dev0" or not torch.cuda.is_available():
        raise RuntimeError("Pinned GPU environment required")
    config = dict(reference_sha256=p.digest(args.work/"reference.json"), runner_sha256=p.digest(Path(__file__)),
                  utils_sha256=p.digest(Path(p.__file__)), model_revision=p.MODEL_SHA, sample_fps=p.FPS,
                  native=args.native, native_max_tokens=args.native_max_tokens,
                  support=sorted(oracles.values(), key=lambda x:(x["sample_id"],x["qa_index"])),
                  torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0),
                  methods=METHODS, seed=p.SEED, fresh_state=True, max_new_tokens=16)
    config_hash = fingerprint(config)
    previous = args.work/"exports/run_config.json"
    if previous.is_file() and json.loads(previous.read_text(encoding="utf-8"))["config_hash"] != config_hash:
        raise RunConflict("Configuration differs from existing run; preserve it and choose a new WORK directory")
    p.atomic_json(previous, dict(**config, config_hash=config_hash))
    failure = args.work/"exports/failure.json"
    if failure.exists():
        target = args.work/"logs"/f"previous_failure_{time.time_ns()}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(failure, target)
    p.report_progress("verifying_or_downloading_videos")
    acquire_videos(videos, args.reuse, args.work)
    p.atomic_json(args.work/"exports/replay_annotations.json", videos)
    sys.path.insert(0, str(args.repo))
    from inference.llavaov_hermes import load_model
    from video_qa.hermes_vqa import HermesVQA
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(p.SEED)
    np.random.seed(p.SEED)
    torch.manual_seed(p.SEED)
    p.report_progress("loading_model")
    model, processor = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=p.FPS)
    model.eval()
    analyzer = HermesVQA(anno=[], save_dir=str(args.work), sample_fps=p.FPS, qa_model=model, qa_processor=processor)
    rows = []
    started = time.perf_counter()
    complete = True
    with torch.inference_mode():
        p.preflight(model, videos[0], args.work)
        for vi, video in enumerate(videos):
            if vi and args.max_hours and time.perf_counter()-started > args.max_hours*3600:
                complete = False
                break
            feature_id = fingerprint(dict(config=config_hash, video=video["video_sha256"]))
            directory, meta, provenance = acquire_features(model, video, args.reuse, args.work, feature_id)
            for qi in range(len(video["conversations"])):
                order = list(METHODS)
                random.Random(p.SEED+video["sample_id"]*10+qi).shuffle(order)
                for method in order:
                    rows.append(run_question(model, analyzer, video, qi, method, directory, meta, config_hash, oracles, args))
            if args.review:
                review_pages(video, args.work)
            summary = export(rows, reference, args.work)
            elapsed = time.perf_counter()-started
            print(f"VIDEO_DONE {vi+1}/8 elapsed_min={elapsed/60:.2f} rough_remaining_min={(7-vi)*elapsed/(vi+1)/60:.2f}", flush=True)
    status = "complete" if complete else "partial_time_budget"
    mandatory = [r for r in rows if r["method"] in METHODS[:3]]
    if complete and (len(mandatory) != 120 or any(r["status"] != "answered" for r in mandatory)):
        raise ValueError("Mandatory 120-answer control matrix incomplete")
    p.atomic_json(args.work/"exports/completion.json", dict(status=status, rows=len(rows),
                  mandatory_answered=sum(r["status"] == "answered" for r in mandatory),
                  config_hash=config_hash, elapsed_post_model_s=time.perf_counter()-started,
                  timing_scope="Includes preflight, shared feature work, per-question controls and review export; excludes setup/download/model loading"))
    p.report_progress(status, recorded=len(rows))
    package(args.work, status)
    print("DIAGNOSTIC_COMPLETE" if complete else "DIAGNOSTIC_PARTIAL", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "work", "model", "reuse"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--native", type=int, default=1, choices=(0,1))
    parser.add_argument("--native-max-tokens", type=int, default=8192)
    parser.add_argument("--review", type=int, default=1, choices=(0,1))
    parser.add_argument("--max-hours", type=float, default=1.0)
    args = parser.parse_args()
    if args.max_hours < 0 or args.native_max_tokens < 196:
        parser.error("Invalid budget/context cap")
    # Reject dangerous output locations before the monitor or failure packager can write there.
    validate_workspaces(args.work, args.reuse)
    with p.ProgressMonitor(args.work):
        try:
            execute(args)
        except RunConflict:
            # Do not replace an older completion/config with this rejected attempt.
            raise
        except Exception as exc:
            p.atomic_json(args.work/"exports/failure.json", dict(status="failed", type=type(exc).__name__, detail=str(exc)[-1500:]))
            p.atomic_json(args.work/"exports/completion.json", dict(status="failed"))
            package(args.work, "failed_partial")
            raise


if __name__ == "__main__":
    main()
