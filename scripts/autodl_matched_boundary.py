"""Isolate history visibility during recent encoding with matched forward shapes."""
import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

try:
    from provenance import stage_base as s
except ModuleNotFoundError as exc:
    if exc.name != "provenance":
        raise
    import autodl_stage_attribution as s

d, p = s.d, s.p
ARMS = s.ARMS
ARTIFACT = "HERMES_matched_boundary_evidence.zip"
SOURCE_PATHS = ("stage_runner.py", "provenance/stage_base.py", "stage_context_utils.py",
                "context_input_utils.py", "diagnostic_pilot_utils.py")


def prepare_states(model, bank, meta, history, recent):
    """Share the original H prefix with previous B/D, then match all R groups."""
    import torch

    p.reset(model)
    original_groups = s.ingest(model, bank, history + recent, meta)
    original = s.snapshot(model)
    prompt = original["prompt"]
    stop = prompt + len(history) * 196
    if history:
        prefix = s.prefix_state(original, stop)
        del original
        s.restore(model, prefix)
        with s.HistoryMask(model, prompt, prompt, force=True) as visible_mask:
            visible_groups = s.ingest(model, bank, recent, meta)
        visible = s.snapshot(model)
        s.restore(model, prefix)
        with s.HistoryMask(model, prompt, stop) as mask:
            hidden_groups = s.ingest(model, bank, recent, meta)
        hidden = s.snapshot(model)
        trace = mask.events
        visible_trace = visible_mask.events
        if len(trace) != len(hidden_groups) or len(visible_trace) != len(visible_groups):
            raise ValueError("Incomplete construction mask trace")
        if not trace or visible_groups != hidden_groups:
            raise ValueError("Recent encoding groups are not matched")
        for a, b in zip(visible_trace, trace, strict=True):
            if a != dict(b, blocked_stop=prompt):
                raise ValueError("Construction masks differ beyond history visibility")
        for i, ((vk, vv), (hk, hv)) in enumerate(zip(visible["kv"], hidden["kv"])):
            torch.testing.assert_close(visible["positions"][i], hidden["positions"][i], atol=0, rtol=0)
            for left, right in ((vk, hk), (vv, hv)):
                torch.testing.assert_close(left[:, :, :stop], right[:, :, :stop], atol=0, rtol=0)
    else:
        visible = hidden = original
        visible_groups = hidden_groups = original_groups
        visible_trace = trace = []
    d.assert_fresh_cache(model, len(history) + len(recent))
    p.reset(model)
    compact_groups = s.ingest(model, bank, recent, meta)
    compact = s.snapshot(model)
    d.assert_fresh_cache(model, len(recent))
    return dict(A=visible, B=hidden, compact=compact), dict(
        prompt_tokens=prompt, history_stop=stop, original_groups=original_groups,
        visible_recent_groups=visible_groups, hidden_recent_groups=hidden_groups,
        compact_groups=compact_groups, visible_construction_mask_trace=visible_trace, hidden_construction_mask_trace=trace,
        original_group_crosses_boundary=any(set(g).intersection(history) and set(g).intersection(recent)
                                            for g in original_groups),
    )


def flip_history(state, start, stop, length):
    changed = s.prefix_state(state, length)
    for keys, values in changed["kv"]:
        keys[:, :, start:stop].neg_()
        values[:, :, start:stop].neg_()
    return changed


def numerical_preflight(model, bank, meta, ids, path):
    """Keep the audited FP32 physical reference and FP16 same-shape checks."""
    import torch

    report = dict(policy="fp32_physical_reference_and_fp16_same_shape_invariance",
                  fp32={}, fp16=dict(construction=[], logits=[]))

    def save():
        p.atomic_json(path, report)

    if model.dtype != torch.float16:
        raise ValueError("Expected the pinned FP16 model")
    print("MATCHED_PREFLIGHT FP32 physical reference", flush=True)
    with s.fp32_structural_reference(model) as reference:
        report["fp32"]["result"] = s._preflight_pass(
            reference, bank, meta, ids, atol=.001, rtol=.001, kv_atol=.0001, kv_rtol=.0001,
            diagnostics=report["fp32"], save=save,
        )
    save()

    print("MATCHED_PREFLIGHT FP16 same-shape invariance", flush=True)
    states, detail = prepare_states(model, bank, meta, [0], [1])
    start, stop = detail["prompt_tokens"], detail["history_stop"]
    hidden = states["B"]
    s.restore(model, flip_history(hidden, start, stop, stop))
    with s.HistoryMask(model, start, stop):
        s.ingest(model, bank, [1], meta)
    changed = s.snapshot(model)
    for layer, ((k, v), (ck, cv)) in enumerate(zip(hidden["kv"], changed["kv"])):
        for kind, a, b in (("K", k, ck), ("V", v, cv)):
            report["fp16"]["construction"].append(dict(
                layer=layer, kind=kind, **s.error_statistics(a[:, :, stop:], b[:, :, stop:], 0, 0)))
    save()
    if any(record["max_abs"] != 0 for record in report["fp16"]["construction"]):
        raise ValueError("Hidden history changed FP16 recent K/V")

    def check_logits(label, left, right):
        for step, (a, b) in enumerate(zip(left, right, strict=True)):
            equal = torch.equal(a[:, -1].argmax(-1), b[:, -1].argmax(-1))
            report["fp16"]["logits"].append(dict(
                comparison=label, step=step, argmax_equal=equal, **s.error_statistics(a, b, .02, .005)))
            save()
            if not equal:
                raise ValueError(f"FP16 preflight argmax changed: {label}, step {step}")

    # Use the same original-partition state as the audited native/explicit control.
    p.reset(model)
    s.ingest(model, bank, [0, 1], meta)
    original = s.snapshot(model)
    native, _, _ = s.probe(model, original, ids, 0, 0)
    explicit, _, _ = s.probe(model, original, ids, 0, 0, force=True)
    check_logits("native_vs_all_visible_mask", native, explicit)
    for label, reference_state in (("original_contextual", original), ("matched_independent", hidden)):
        masked, _, _ = s.probe(model, reference_state, ids, start, stop)
        compact, _, _ = s.probe(model, s.drop_history(reference_state, start, stop), ids, 0, 0)
        check_logits(label + "_physical_removal", masked, compact)
    for state in (states["A"], hidden):
        masked, _, events = s.probe(model, state, ids, start, stop)
        perturbed, _, _ = s.probe(model, flip_history(state, start, stop, s.cache_length(state["kv"])),
                                  ids, start, stop)
        for a, b in zip(masked, perturbed, strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        if len(events) != 3 or [e["query_tokens"] for e in events][1:] != [1, 1]:
            raise ValueError("Missing FP16 prefill/decode mask checks")
    _, weights, _ = s.probe(model, hidden, ids, start, stop, attentions=True)
    if len(weights) != 3 or any(len(layers) != model.num_layers for layers in weights):
        raise ValueError("Incomplete FP16 attention trace")
    for layers in weights:
        for attention in layers:
            if torch.any(attention[..., start:stop]) or torch.any(torch.triu(attention[..., -attention.shape[-2]:], 1)):
                raise ValueError("FP16 attention reached hidden history or future tokens")
    s.assert_state(model, hidden)
    model.clear_cache()
    report.update(status="pass", fp16_masked_readout_exact=True, fp16_hidden_and_future_attention_zero=True)
    save()


def export(rows, old, work):
    s.export(rows, work)
    current = {(r["sample_id"], r["qa_index"], r["method"]): r for r in rows}
    comparisons = [
        ("primary_C_matched_vs_D", "current", 2, "current", 3),
        ("A_matched_vs_B", "current", 0, "current", 1),
        ("C_matched_vs_original_C", "current", 2, "previous", 2),
        ("A_matched_vs_original_A", "current", 0, "previous", 0),
        ("C_matched_vs_compact_recent4", "current", 2, "current", 4),
        ("C_matched_vs_A_matched_readout", "current", 2, "current", 0),
    ]
    details, summary = [], []
    for label, left_source, left_arm, right_source, right_arm in comparisons:
        left_rows = current if left_source == "current" else old
        right_rows = current if right_source == "current" else old
        matched = []
        for sid, qi, method in sorted(current):
            if method != ARMS[left_arm] or (sid, qi, ARMS[right_arm]) not in right_rows:
                continue
            a, b = left_rows[sid, qi, method], right_rows[sid, qi, ARMS[right_arm]]
            matched.append(dict(contrast=label, sample_id=sid, qa_index=qi, history_empty=a["history_empty"],
                                left_choice=a["pred_choice"], right_choice=b["pred_choice"],
                                left_correct=a["correct"], right_correct=b["correct"]))
        details.extend(matched)
        summary.append(dict(contrast=label, questions=len(matched),
                            left_only_correct=sum(r["left_correct"] > r["right_correct"] for r in matched),
                            right_only_correct=sum(r["left_correct"] < r["right_correct"] for r in matched),
                            choice_changes=sum(r["left_choice"] != r["right_choice"] for r in matched)))
    p.write_csv(work / "exports/matched_pairs.csv", summary)
    p.write_csv(work / "exports/matched_question_pairs.csv", details)
    report_path = work / "exports/report.md"
    base_report = report_path.read_text(encoding="utf-8").replace(
        "Long 204/q3 results are separate from the main 40-question matrix. A and compact must replay earlier choices or execution stops.",
        "B, D and compact recent-4 must replay earlier choices. This run uses the 40 main questions.",
    )
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(base_report)
        handle.write("\n## Matched construction contrasts\n\n")
        handle.write("A and C here split recent encoding at the H/R boundary and use an explicit causal mask. "
                     "The primary contrast is C versus D with identical recent groups and hidden history at readout.\n\n")
        handle.write("| Contrast | Questions | Left wins | Right wins | Choice changes |\n|---|---:|---:|---:|---:|\n")
        for row in summary:
            handle.write("| " + " | ".join(map(str, row.values())) + " |\n")


def package(work, status):
    original_name = s.ARTIFACT
    try:
        s.ARTIFACT = ARTIFACT
        s.package(work, status)
    finally:
        s.ARTIFACT = original_name


def execute(args):
    import numpy as np
    import torch
    import transformers

    reference = json.loads((args.work / "reference.json").read_text(encoding="utf-8"))
    d.check_reference(reference, args.repo)
    old = {(r["sample_id"], r["qa_index"], r["method"]): r for r in reference["stage_rows"]}
    if len(old) != 200 or args.long_case:
        raise ValueError("Expected the 40-question matched-boundary experiment only")
    if (args.model.name != p.MODEL_SHA or torch.__version__ != "2.10.0+cu128"
            or transformers.__version__ != "4.45.0.dev0" or not torch.cuda.is_available()):
        raise ValueError("Pinned model/CUDA environment required")
    config = dict(reference_sha256=p.digest(args.work / "reference.json"),
                  sources={name: p.digest(args.work / name) for name in SOURCE_PATHS},
                  previous_stage_archive_sha256=reference["stage_archive_sha256"],
                  model_revision=p.MODEL_SHA, torch=torch.__version__, transformers=transformers.__version__,
                  gpu=torch.cuda.get_device_name(0), seed=p.SEED, arms=ARMS, main_window=20, recent_window=4,
                  construction="Shared native H prefix; identical R groups; H visibility is the only construction intervention",
                  primary_contrast="C_matched_vs_D", exact_replay_arms=[ARMS[1], ARMS[3], ARMS[4]])
    config_hash = d.fingerprint(config)
    config_path = args.work / "exports/run_config.json"
    if config_path.exists() and json.loads(config_path.read_text())["config_hash"] != config_hash:
        raise d.RunConflict("Matched-boundary config changed; use a new WORK")
    p.atomic_json(config_path, dict(**config, config_hash=config_hash))
    failure = args.work / "exports/failure.json"
    if failure.exists():
        previous_failure = args.work / "logs" / f"previous_failure_{time.time_ns()}.json"
        previous_failure.parent.mkdir(exist_ok=True)
        failure.replace(previous_failure)
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
    p.report_progress("loading_matched_model")
    model, processor = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=p.FPS)
    model.eval()
    analyzer = HermesVQA(anno=[], save_dir=str(args.work), sample_fps=p.FPS, qa_model=model, qa_processor=processor)
    rows, done, started = [], 0, time.perf_counter()
    videos.sort(key=lambda v: (v["sample_id"] != 204, v["sample_id"]))
    with torch.inference_mode():
        for video in videos:
            directory, meta, _ = d.acquire_features(model, video, args.reuse, args.work,
                d.fingerprint(dict(config=config_hash, video=video["video_sha256"])))
            bank, feature_hash = s.c.FeatureBank(directory, meta), d.fingerprint(meta)
            if video["sample_id"] == 204:
                p.preflight(model, video, args.work)
                ids = processor.tokenizer("What is visible?", return_tensors="pt").input_ids.to(model.device)
                p.report_progress("matched_mask_position_preflight")
                numerical_preflight(model, bank, meta, ids, args.work / "exports/matched_preflight.json")
                print("MATCHED_PREFLIGHT_PASS", flush=True)
            for qi in range(len(video["conversations"])):
                if done and args.max_hours and time.perf_counter() - started > args.max_hours * 3600:
                    break
                h, r = s.split_frames(video, qi)
                torch.cuda.reset_peak_memory_stats()
                before = time.perf_counter()
                states, detail = prepare_states(model, bank, meta, h, r)
                p.sync()
                detail.update(history_indices=h, recent_indices=r, shared_preparation_s=time.perf_counter() - before,
                              preparation_peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3)
                sid = video["sample_id"]
                p.atomic_json(args.work / "preparation/main" / f"sample_{sid}_q{qi}.json", detail)
                block = []
                for i in (1, 3, 4, 0, 2):
                    arm = ARMS[i]
                    previous = old[sid, qi, arm]["pred_choice"] if i in (1, 3, 4) else None
                    block.append(s.run_answer(model, analyzer, video, qi, arm, "main", states, detail,
                                              h, r, feature_hash, config_hash, args, previous))
                if not h and len({row["pred_choice"] for row in block}) != 1:
                    raise ValueError("Empty-history null control disagreed")
                rows.extend(block)
                del states
                done += 1
                export(rows, old, args.work)
                print(f"QUESTION_BLOCK_DONE {done}/40 elapsed_min={(time.perf_counter() - started) / 60:.2f}", flush=True)
            del bank
            if args.max_hours and time.perf_counter() - started > args.max_hours * 3600:
                break
    status = "complete" if done == 40 else "partial_time_budget"
    if len(rows) != 5 * done or any(row["status"] != "answered" for row in rows):
        raise ValueError("Matched-boundary matrix incomplete")
    p.atomic_json(args.work / "exports/completion.json", dict(status=status, main_questions=done,
        main_answers=len(rows), exact_replays=3 * done, config_hash=config_hash,
        elapsed_post_model_s=time.perf_counter() - started))
    p.report_progress(status, main_questions=done)
    package(args.work, status)
    print(f"MATCHED_BOUNDARY_{status.upper()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "work", "model", "reuse"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--long-case", type=int, choices=(0,), default=0)
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
