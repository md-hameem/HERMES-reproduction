"""Compare position-preserving physical history removal with masked readout."""
import argparse
import copy
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

try:
    from provenance import matched_base as m
except ModuleNotFoundError as exc:
    if exc.name != "provenance":
        raise
    import autodl_matched_boundary as m

s, d, p = m.s, m.d, m.p
ARMS = ("full_contextual", "masked_contextual", "short_contextual", "compact_recent4")
ARTIFACT = "HERMES_short_query_evidence.zip"
SOURCE_PATHS = (*m.SOURCE_PATHS, "provenance/matched_base.py")


def cache_bytes(state):
    return sum(t.numel() * t.element_size() for pair in state["kv"] for t in pair)


def make_short(state, start, stop):
    """Remove H without reindexing RoPE or changing any retained keys/values."""
    import torch

    short = s.drop_history(state, start, stop)
    for full_pos, short_pos in zip(state["positions"], short["positions"], strict=True):
        torch.testing.assert_close(short_pos, torch.cat((full_pos[:start], full_pos[stop:])), atol=0, rtol=0)
        if short_pos[-1] != full_pos[-1]:
            raise ValueError("Physical removal changed the next logical position")
    if s.cache_length(short["kv"]) != s.cache_length(state["kv"]) - (stop - start):
        raise ValueError("Unexpected physical short cache length")
    return short


def position_trace(model, state, prompt, mask_span):
    """Use the unchanged QA wrapper and capture actual prefill/decode positions."""
    import torch

    s.restore(model, state)
    base = s.cache_length(state["kv"])
    logical = int(state["positions"][0][-1]) + 1
    calls = []

    def record(_module, _args, kwargs):
        calls.append((s.cache_length(kwargs["past_key_values"]), kwargs["position_ids"].detach()))

    handle = model.language_model.register_forward_pre_hook(record, with_kwargs=True)
    try:
        with s.HistoryMask(model, *mask_span) as mask:
            p.sync()
            before = time.perf_counter()
            answer = model.question_answering(prompt, max_new_tokens=16)
            p.sync()
            elapsed = time.perf_counter() - before
    finally:
        handle.remove()
    if not calls or len(calls) > 17:
        raise ValueError("Missing or oversized query trace")
    trace = []
    for step, (past, positions) in enumerate(calls):
        n = positions.shape[-1]
        expected = torch.arange(logical, logical + n, device=positions.device).view(1, -1)
        if past != base or (step > 0 and n != 1) or not torch.equal(positions, expected):
            raise ValueError("Answer wrapper used incorrect cache lengths or logical positions")
        trace.append(dict(past_tokens=past, logical_start=logical, query_tokens=n))
        base += n
        logical += n
    if mask_span[0] != mask_span[1]:
        if len(mask.events) != len(trace) or any(
            (e["blocked_start"], e["blocked_stop"]) != mask_span
            or (e["past_tokens"], e["query_tokens"]) != (t["past_tokens"], t["query_tokens"])
            for e, t in zip(mask.events, trace, strict=True)
        ):
            raise ValueError("Incomplete answer mask")
    s.assert_state(model, state)
    return answer, elapsed, trace


def preflight(model, bank, meta, ids, work):
    import torch

    m.numerical_preflight(model, bank, meta, ids, work / "exports/matched_preflight.json")
    report = {}

    def compare(target, strict):
        states, detail = m.prepare_states(target, bank, meta, [0], [1])
        full = states["A"]
        start, stop = detail["prompt_tokens"], detail["history_stop"]
        short = make_short(full, start, stop)
        masked, _, _ = s.probe(target, full, ids, start, stop)
        removed, _, _ = s.probe(target, short, ids, 0, 0)
        rows = []
        for step, (a, b) in enumerate(zip(masked, removed, strict=True)):
            rows.append(dict(step=step, argmax_equal=torch.equal(a[:, -1].argmax(-1), b[:, -1].argmax(-1)),
                             **s.error_statistics(a, b, .001 if strict else .02, .001 if strict else .005)))
        report[str(target.dtype)] = rows
        p.atomic_json(work / "exports/short_preflight.json", report)
        if strict:
            for a, b in zip(masked, removed, strict=True):
                torch.testing.assert_close(a, b, atol=.001, rtol=.001)
            if not all(r["argmax_equal"] for r in rows):
                raise ValueError("FP32 short query changed preflight argmax")

    with s.fp32_structural_reference(model) as reference:
        compare(reference, True)
    compare(model, False)
    print("SHORT_QUERY_PREFLIGHT_PASS", flush=True)


def compare_rows(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["sample_id"], row["qa_index"]), {})[row["method"]] = row
    pairs = []
    for left, right in (("short_contextual", "masked_contextual"), ("short_contextual", "compact_recent4"),
                        ("short_contextual", "full_contextual")):
        matched = [(g[left], g[right]) for g in groups.values()]
        byte_deltas = [b["kv_bytes"] - a["kv_bytes"] for a, b in matched]
        time_deltas = [b["answer_median_s"] - a["answer_median_s"] for a, b in matched]
        equal_work = [delta for (a, b), delta in zip(matched, time_deltas, strict=True)
                      if [len(t) for t in a["traces"]] == [len(t) for t in b["traces"]]]
        pairs.append(dict(left=left, right=right, questions=len(matched),
                          left_only_correct=sum(a["correct"] > b["correct"] for a, b in matched),
                          right_only_correct=sum(a["correct"] < b["correct"] for a, b in matched),
                          choice_changes=sum(a["pred_choice"] != b["pred_choice"] for a, b in matched),
                          median_kv_bytes_saved=statistics.median(byte_deltas),
                          median_paired_answer_s_saved=statistics.median(time_deltas),
                          equal_decode_count_questions=len(equal_work),
                          median_equal_decode_answer_s_saved=statistics.median(equal_work) if equal_work else None))
    return pairs


def export(rows, work):
    p.write_csv(work / "exports/answers.csv", rows)
    pairs = compare_rows(rows)
    p.write_csv(work / "exports/paired.csv", pairs)
    summary = []
    for arm in ARMS:
        group = [row for row in rows if row["method"] == arm]
        summary.append(dict(arm=arm, answered=len(group), correct=sum(r["correct"] for r in group),
                            median_kv_bytes=statistics.median(r["kv_bytes"] for r in group),
                            median_answer_s=statistics.median(r["answer_median_s"] for r in group)))
    p.write_csv(work / "exports/summary.csv", summary)
    lines = ["# Short query cache development run", "",
             "| Arm | Correct | Answered | Median query KV bytes | Median answer seconds |",
             "|---|---:|---:|---:|---:|"]
    lines.extend(f"| {r['arm']} | {r['correct']} | {r['answered']} | {r['median_kv_bytes']} | {r['median_answer_s']:.4f} |" for r in summary)
    lines.extend(["", "Query KV bytes count retained K/V tensors. Answer timing excludes restore and cache construction.",
                  "The same 40 development questions are used. Timing does not measure total live memory or whole-system latency."])
    (work / "exports/report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_block(model, analyzer, video, qi, states, detail, short, feature_hash, old, repeats, warmup=False):
    start, stop = detail["prompt_tokens"], detail["history_stop"]
    sid = video["sample_id"]
    inputs = {
        ARMS[0]: (states["A"], (0, 0)),
        ARMS[1]: (states["A"], (start, stop)),
        ARMS[2]: (short, (0, 0)),
        ARMS[3]: (states["compact"], (0, 0)),
    }
    identity = d.question_identity(video, qi, ARMS[0])
    prompt = analyzer.format_mcqa_prompt(identity["question"], identity["choices"])
    choices, texts, times, traces = ({a: [] for a in ARMS} for _ in range(4))
    orders = []
    rng = random.Random(p.SEED + sid * 31 + qi)
    for rep in range(-int(warmup), repeats):
        order = list(ARMS)
        rng.shuffle(order)
        if rep >= 0:
            orders.append(order)
        for arm in order:
            state, span = inputs[arm]
            answer, elapsed, trace = position_trace(model, state, prompt, span)
            choice = analyzer.extract_characters_regex(answer)
            if rep >= 0:
                choices[arm].append(choice)
                texts[arm].append(answer.replace("\n", ""))
                times[arm].append(elapsed)
                traces[arm].append(trace)
    rows = []
    for arm in ARMS:
        state, _ = inputs[arm]
        expected = old.get((sid, qi, arm))
        row = d.question_identity(video, qi, arm)
        row.update(pred_choice=choices[arm][0], pred_answer=texts[arm][0],
                   correct=int(choices[arm][0] == identity["gold_choice"]),
                   choices_by_repeat=choices[arm], answer_seconds=times[arm], traces=traces[arm],
                   answer_median_s=statistics.median(times[arm]),
                   physical_tokens=s.cache_length(state["kv"]), kv_bytes=cache_bytes(state),
                   query_logical_start=traces[arm][0][0]["logical_start"], feature_fingerprint=feature_hash,
                   historical_choice=expected, historical_matches=choices[arm][0] == expected if expected is not None else None)
        rows.append(row)
    if rows[1]["query_logical_start"] != rows[2]["query_logical_start"]:
        raise ValueError("Short and masked query starts disagree")
    return rows, orders


def package(work, status):
    name = s.ARTIFACT
    try:
        s.ARTIFACT = ARTIFACT
        s.package(work, status)
    finally:
        s.ARTIFACT = name


def execute(args):
    import numpy as np
    import torch
    import transformers

    if (args.model.name != p.MODEL_SHA or torch.__version__ != "2.10.0+cu128"
            or transformers.__version__ != "4.45.0.dev0" or not torch.cuda.is_available()):
        raise ValueError("Pinned model/CUDA environment required")
    path = args.work / "reference.json"
    reference = json.loads(path.read_text(encoding="utf-8"))
    d.check_reference(reference, args.repo)
    old = {(r["sample_id"], r["qa_index"], r["arm"]): r["choice"] for r in reference["baseline_choices"]}
    if len(old) != 120:
        raise ValueError("Incomplete reference choices")
    config = dict(reference_sha256=p.digest(path), previous_archive_sha256=reference["matched_archive_sha256"],
                  sources={n: p.digest(args.work / n) for n in SOURCE_PATHS}, arms=ARMS, repeats=args.repeats,
                  gpu=torch.cuda.get_device_name(0), model_revision=p.MODEL_SHA, torch=torch.__version__,
                  transformers=transformers.__version__, seed=p.SEED)
    config_hash = d.fingerprint(config)
    config_path = args.work / "exports/run_config.json"
    if config_path.exists() and json.loads(config_path.read_text())["config_hash"] != config_hash:
        raise d.RunConflict("Short query configuration changed; use a new WORK")
    p.atomic_json(config_path, dict(**config, config_hash=config_hash))
    failure = args.work / "exports/failure.json"
    if failure.exists():
        previous = args.work / "logs" / f"previous_failure_{time.time_ns()}.json"
        previous.parent.mkdir(exist_ok=True)
        failure.replace(previous)
    videos = copy.deepcopy(reference["videos"])
    d.acquire_videos(videos, args.reuse, args.work)
    p.atomic_json(args.work / "exports/replay_annotations.json", videos)
    sys.path.insert(0, str(args.repo))
    from inference.llavaov_hermes import load_model
    from video_qa.hermes_vqa import HermesVQA

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(p.SEED)
    np.random.seed(p.SEED)
    torch.manual_seed(p.SEED)
    p.report_progress("loading_short_query_model")
    model, processor = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=p.FPS)
    model.eval()
    analyzer = HermesVQA(anno=[], save_dir=str(args.work), sample_fps=p.FPS, qa_model=model, qa_processor=processor)
    rows, done, started, warmed = [], 0, time.perf_counter(), False
    videos.sort(key=lambda v: (v["sample_id"] != 204, v["sample_id"]))
    with torch.inference_mode():
        for video in videos:
            directory, meta, _ = d.acquire_features(model, video, args.reuse, args.work,
                d.fingerprint(dict(config=config_hash, video=video["video_sha256"])))
            bank = s.c.FeatureBank(directory, meta)
            feature_hash = d.fingerprint(meta)
            if video["sample_id"] == 204:
                p.preflight(model, video, args.work)
                ids = processor.tokenizer("What is visible?", return_tensors="pt").input_ids.to(model.device)
                preflight(model, bank, meta, ids, args.work)
            for qi in range(len(video["conversations"])):
                if done and args.max_hours and time.perf_counter() - started > args.max_hours * 3600:
                    break
                sid = video["sample_id"]
                h, r = s.split_frames(video, qi)
                key = d.fingerprint(dict(config=config_hash, feature=feature_hash, sample=sid, qa=qi, h=h, r=r))
                checkpoint = args.work / "checkpoints" / f"sample_{sid}_q{qi}.json"
                if checkpoint.exists():
                    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                    if saved["key"] != key or d.fingerprint(saved["rows"]) != saved["rows_sha256"]:
                        raise d.RunConflict("Short query checkpoint differs; use a new WORK")
                    block = saved["rows"]
                else:
                    p.report_progress("short_query_question", sample_id=sid, qa_index=qi)
                    p.sync()
                    before = time.perf_counter()
                    states, detail = m.prepare_states(model, bank, meta, h, r)
                    p.sync()
                    preparation_s = time.perf_counter() - before
                    start, stop = detail["prompt_tokens"], detail["history_stop"]
                    before = time.perf_counter()
                    short = make_short(states["A"], start, stop)
                    p.sync()
                    materialize_s = time.perf_counter() - before
                    block, orders = run_block(model, analyzer, video, qi, states, detail, short, feature_hash,
                                              old, args.repeats, warmup=not warmed)
                    warmed = True
                    p.atomic_json(args.work / "preparation" / f"sample_{sid}_q{qi}.json", dict(
                        **detail, history_indices=h, recent_indices=r, short_materialize_s=materialize_s,
                        shared_preparation_s=preparation_s, timing_orders=orders))
                    p.atomic_json(checkpoint, dict(key=key, rows=block, rows_sha256=d.fingerprint(block)))
                    model.clear_cache()
                    del states, short
                if any(row["historical_matches"] is False or len(set(row["choices_by_repeat"])) != 1 for row in block):
                    raise ValueError("Baseline replay or within-arm repeat changed; inspect saved checkpoint")
                if not h and len({row["pred_choice"] for row in block}) != 1:
                    raise ValueError("Empty-history null control disagreed")
                rows.extend(block)
                done += 1
                export(rows, args.work)
                print(f"QUESTION_BLOCK_DONE {done}/40 elapsed_min={(time.perf_counter()-started)/60:.2f}", flush=True)
            del bank
            if args.max_hours and time.perf_counter() - started > args.max_hours * 3600:
                break
    status = "complete" if done == 40 else "partial_time_budget"
    if len(rows) != 4 * done:
        raise ValueError("Short query matrix incomplete")
    p.atomic_json(args.work / "exports/completion.json", dict(status=status, main_questions=done,
        main_answers=len(rows), timed_calls=len(rows)*args.repeats, config_hash=config_hash,
        elapsed_post_model_s=time.perf_counter()-started))
    p.report_progress(status, main_questions=done)
    package(args.work, status)
    print(f"SHORT_QUERY_{status.upper()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "work", "model", "reuse"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--max-hours", type=float, default=1.0)
    args = parser.parse_args()
    for name in ("repo", "work", "model", "reuse"):
        setattr(args, name, getattr(args, name).resolve())
    d.validate_workspaces(args.work, args.reuse)
    if args.max_hours < 0:
        parser.error("Invalid soft budget")
    with p.ProgressMonitor(args.work):
        try:
            execute(args)
        except d.RunConflict:
            raise
        except Exception as exc:
            p.atomic_json(args.work / "exports/failure.json", dict(type=type(exc).__name__, detail=str(exc)[-1500:]))
            p.atomic_json(args.work / "exports/completion.json", dict(status="failed"))
            package(args.work, "failed_partial")
            raise


if __name__ == "__main__":
    main()
