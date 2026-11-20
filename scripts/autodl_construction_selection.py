"""Test bounded, visual-only history selection for recent-cache construction."""
import argparse
import copy
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

try:
    from provenance import short_base as q
except ModuleNotFoundError as exc:
    if exc.name != "provenance":
        raise
    import autodl_short_query as q

s, d, p = q.s, q.d, q.p
ARMS = ("selected_two", "latest_two", "attention_mass_two", "no_history", "all_history", "compact_recent4")
ARTIFACT = "HERMES_construction_selection_evidence.zip"
SOURCE_PATHS = (*q.SOURCE_PATHS, "provenance/short_base.py")


def attention_stats(logits, values):
    """Sufficient statistics for one attention key group."""
    import torch

    return torch.logsumexp(logits, dim=-1), logits.softmax(dim=-1) @ values


def merge_stats(left, right):
    import torch

    left_z, left_mean = left
    right_z, right_mean = right
    total = torch.logaddexp(left_z, right_z)
    return total, ((left_z - total).exp().unsqueeze(-1) * left_mean
                   + (right_z - total).exp().unsqueeze(-1) * right_mean)


def select_groups(base, older, count=2):
    """Greedily preserve the full first-layer output; break ties by recency."""
    import torch

    target = base
    for group in older:
        target = merge_stats(target, group)
    masses = [float((z - target[0]).exp().mean()) for z, _ in older]
    selected, gains = [], []
    running = base
    for _ in range(min(len(older), count)):
        candidates = [i for i in range(len(older)) if i not in selected]
        errors = torch.stack([
            (merge_stats(running, older[i])[1] - target[1]).square().mean() for i in candidates
        ]).tolist()
        index, error = min(zip(candidates, errors, strict=True), key=lambda x: (x[1], -x[0]))
        selected.append(index)
        gains.append(error)
        running = merge_stats(running, older[index])
    mass_order = sorted(range(len(older)), key=lambda i: (masses[i], i), reverse=True)[:count]
    def residual(indices):
        result = base
        for index in indices:
            result = merge_stats(result, older[index])
        return float((result[1] - target[1]).square().mean())

    return selected, mass_order, dict(
        attention_mass=masses, selected_residual_mse=gains,
        no_history_residual_mse=residual([]), mass_residual_mse=residual(mass_order),
        latest_residual_mse=residual(range(max(0, len(older) - count), len(older))),
    )


def score_history(model, bank, prefix, recent, count):
    """Project the recent visual tokens once and score only real, pre-query history."""
    import torch
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    layer = model.language_model.model.layers[0]
    attention = layer.self_attn
    x = layer.input_layernorm(bank.get(recent).to(device=model.device, dtype=model.dtype))
    batch, n, _ = x.shape

    def project(projection, heads):
        return projection(x).view(batch, n, heads, attention.head_dim).transpose(1, 2)

    query = project(attention.q_proj, attention.num_heads)
    key = project(attention.k_proj, attention.num_key_value_heads)
    value = project(attention.v_proj, attention.num_key_value_heads)
    start = int(prefix["positions"][0][-1]) + 1
    model._set_rotary_required_len(start + n)
    cos, sin = attention.rotary_emb(value, seq_len=start + n)
    query, key = apply_rotary_pos_emb(query, key, cos, sin, model._build_position_ids(start, n, batch))
    pk, pv = prefix["kv"][0]
    groups = attention.num_key_value_groups
    query = query.float() / math.sqrt(attention.head_dim)
    first = prefix["prompt"]
    past_key = repeat_kv(pk, groups).float()
    past_value = repeat_kv(pv, groups).float()
    recent_key = repeat_kv(key, groups).float()
    recent_value = repeat_kv(value, groups).float()
    current = (query @ recent_key.transpose(-1, -2)).masked_fill(
        torch.ones((n, n), device=query.device, dtype=torch.bool).triu(1), float("-inf"))
    sink = query @ past_key[:, :, :first].transpose(-1, -2)
    base = attention_stats(torch.cat((sink, current), dim=-1),
                           torch.cat((past_value[:, :, :first], recent_value), dim=2))
    older = []
    for offset in range(first, pk.shape[2], 196):
        k = past_key[:, :, offset:offset + 196]
        v = past_value[:, :, offset:offset + 196]
        older.append(attention_stats(query @ k.transpose(-1, -2), v))
    return select_groups(base, older, count)


def slice_prefix(prefix, frames):
    import torch

    start = prefix["prompt"]
    device = prefix["positions"][0].device
    indices = [torch.arange(start, device=device)]
    indices.extend(torch.arange(start + i * 196, start + (i + 1) * 196, device=device) for i in sorted(frames))
    keep = torch.cat(indices)
    return dict(prompt=start, kv=[(k.index_select(2, keep), v.index_select(2, keep)) for k, v in prefix["kv"]],
                positions=[x.index_select(0, keep) for x in prefix["positions"]])


def construction(model, bank, meta, prefix, recent, selected, logical_start):
    """Encode recent frames once with physically selected H at unchanged positions."""
    support = slice_prefix(prefix, selected)
    s.restore(model, support)
    start = prefix["prompt"]
    # All arms use the same explicit causal mask path. The older span is physically selected.
    with s.HistoryMask(model, start, start, force=True):
        groups = s.ingest(model, bank, recent, meta, logical_start=logical_start)
    encoded = s.snapshot(model)
    stop = start + 196 * len(selected)
    expected = logical_start + 196 * len(recent)
    if any(int(pos[-1]) + 1 != expected or int(pos[stop]) != logical_start for pos in encoded["positions"]):
        raise ValueError("Context construction changed logical positions")
    return encoded, groups, stop


def question(model, analyzer, bank, meta, video, qi, history, recent, old):
    sid = video["sample_id"]
    p.sync()
    started = time.perf_counter()
    p.reset(model)
    s.ingest(model, bank, history + recent, meta)
    original = s.snapshot(model)
    start = original["prompt"]
    logical_start = start + 196 * len(history)
    prefix = s.prefix_state(original, logical_start)
    p.sync()
    shared_prefix_s = time.perf_counter() - started

    before = time.perf_counter()
    chosen, mass, trace = score_history(model, bank, prefix, recent, 2) if history else ([], [], {})
    p.sync()
    scoring_s = time.perf_counter() - before if history else 0.0
    selections = dict(selected_two=chosen, latest_two=list(range(max(0, len(history) - 2), len(history))),
                      attention_mass_two=mass, no_history=[], all_history=list(range(len(history))), compact_recent4=[])
    base_identity = d.question_identity(video, qi, ARMS[0])
    prompt = analyzer.format_mcqa_prompt(base_identity["question"], base_identity["choices"])
    order = list(ARMS)
    random.Random(p.SEED + sid * 31 + qi).shuffle(order)
    rows = {}
    for arm in order:
        selected = selections[arm]
        p.sync()
        before = time.perf_counter()
        if arm == "compact_recent4":
            p.reset(model)
            groups = s.ingest(model, bank, recent, meta)
            encoded, stop = s.snapshot(model), start
        elif not history:
            encoded, groups, stop = original, s.c.input_groups(recent, "original", meta), start
        else:
            encoded, groups, stop = construction(model, bank, meta, prefix, recent, selected, logical_start)
        p.sync()
        encoding_s = time.perf_counter() - before
        before = time.perf_counter()
        state = q.make_short(encoded, start, stop) if stop > start else encoded
        p.sync()
        extraction_s = time.perf_counter() - before if stop > start else 0.0
        answer, answer_s, positions = q.position_trace(model, state, prompt, (0, 0))
        choice = analyzer.extract_characters_regex(answer)
        expected = old.get((sid, qi, arm))
        row = d.question_identity(video, qi, arm)
        row.update(pred_choice=choice, pred_answer=answer.replace("\n", ""), correct=int(choice == row["gold_choice"]),
                   feature_fingerprint=d.fingerprint(meta),
                   history_indices=history, recent_indices=recent, selected_history=[history[i] for i in sorted(selected)],
                   recent_groups=groups, position_trace=positions, query_kv_bytes=q.cache_bytes(state),
                   construction_history_slots=196 * len(selected), construction_s=encoding_s,
                   score_s=scoring_s if arm in ("selected_two", "attention_mass_two") else 0.0,
                   extraction_s=extraction_s, answer_s=answer_s,
                   reference_choice=expected, reference_matches=choice == expected if expected is not None else None)
        rows[arm] = row
        model.clear_cache()
    if len({row["query_kv_bytes"] for row in rows.values()}) != 1:
        raise ValueError("Answer cache budget differs between arms")
    if len({rows[arm]["position_trace"][0]["logical_start"] for arm in ARMS[:-1]}) != 1:
        raise ValueError("Experimental arms have different query positions")
    if len({tuple(map(tuple, row["recent_groups"])) for row in rows.values()}) != 1:
        raise ValueError("Recent construction groups differ between arms")
    choices_by_subset = {}
    for arm in ARMS[:-1]:
        row = rows[arm]
        subset = tuple(row["selected_history"])
        if choices_by_subset.setdefault(subset, row["pred_choice"]) != row["pred_choice"]:
            raise ValueError("Same construction subset produced different choices")
    return [rows[arm] for arm in ARMS], dict(history_indices=history, recent_indices=recent,
        shared_prefix_s=shared_prefix_s, score_s=scoring_s, score_trace=trace, answer_order=order)


def export(rows, work):
    p.write_csv(work / "exports/answers.csv", rows)
    grouped = {}
    for row in rows:
        grouped.setdefault((row["sample_id"], row["qa_index"]), {})[row["method"]] = row
    summary, paired = [], []
    for arm in ARMS:
        subset = [group[arm] for group in grouped.values()]
        summary.append(dict(arm=arm, answered=len(subset), correct=sum(r["correct"] for r in subset),
                            median_query_kv_bytes=statistics.median(r["query_kv_bytes"] for r in subset),
                            median_score_and_construction_s=statistics.median(r["score_s"]+r["construction_s"] for r in subset)))
        if arm == "selected_two":
            continue
        pairs = [(g["selected_two"], g[arm]) for g in grouped.values()]
        paired.append(dict(candidate="selected_two", baseline=arm, questions=len(pairs),
                           candidate_only_correct=sum(a["correct"] > b["correct"] for a, b in pairs),
                           baseline_only_correct=sum(a["correct"] < b["correct"] for a, b in pairs),
                           choice_changes=sum(a["pred_choice"] != b["pred_choice"] for a, b in pairs),
                           median_score_and_construction_s_saved=statistics.median(
                               b["score_s"] + b["construction_s"] - a["score_s"] - a["construction_s"] for a, b in pairs)))
    p.write_csv(work / "exports/summary.csv", summary)
    p.write_csv(work / "exports/paired.csv", paired)
    lines = ["# Construction selection development run", "", "| Arm | Correct | Answered |", "|---|---:|---:|"]
    lines.extend(f"| {r['arm']} | {r['correct']} | {r['answered']} |" for r in summary)
    lines.extend(["", "All arms have the same answer K/V budget. The shared history build is excluded from paired construction timing.",
                  "Attention-mass timing conservatively charges the shared scorer. Its standalone cost may be lower."])
    (work / "exports/report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(args):
    import torch
    import transformers

    ref_path = args.work / "reference.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    d.check_reference(ref, args.repo)
    old = {(r["sample_id"], r["qa_index"], r["arm"]): r["choice"] for r in ref["baseline_choices"]}
    if (len(old) != 80 or args.model.name != p.MODEL_SHA or not torch.cuda.is_available()
            or torch.__version__ != "2.10.0+cu128" or transformers.__version__ != "4.45.0.dev0"):
        raise ValueError("Expected pinned model, runtime and 80 reference choices")
    config = dict(reference_sha256=p.digest(ref_path), sources={n: p.digest(args.work/n) for n in SOURCE_PATHS},
                  model=p.MODEL_SHA, gpu=torch.cuda.get_device_name(0), arms=ARMS, history_budget_frames=2, seed=p.SEED,
                  torch=torch.__version__, transformers=transformers.__version__,
                  short_query_archive_sha256=ref["short_query_archive_sha256"])
    config_hash = d.fingerprint(config)
    path = args.work / "exports/run_config.json"
    if path.exists() and json.loads(path.read_text())["config_hash"] != config_hash:
        raise d.RunConflict("Construction selection config changed; use new WORK")
    p.atomic_json(path, dict(**config, config_hash=config_hash))
    videos = copy.deepcopy(ref["videos"])
    d.acquire_videos(videos, args.reuse, args.work)
    p.atomic_json(args.work / "exports/replay_annotations.json", videos)
    sys.path.insert(0, str(args.repo))
    from inference.llavaov_hermes import load_model
    from video_qa.hermes_vqa import HermesVQA

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(p.SEED)
    p.report_progress("loading_selection_model")
    model, processor = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=p.FPS)
    model.eval()
    analyzer = HermesVQA(anno=[], save_dir=str(args.work), sample_fps=p.FPS, qa_model=model, qa_processor=processor)
    rows, done, started = [], 0, time.perf_counter()
    videos.sort(key=lambda v: (v["sample_id"] != 204, v["sample_id"]))
    with torch.inference_mode():
        for video in videos:
            directory, meta, _ = d.acquire_features(model, video, args.reuse, args.work,
                                                   d.fingerprint(dict(config=config_hash, video=video["video_sha256"])))
            bank = s.c.FeatureBank(directory, meta)
            sid = video["sample_id"]
            if sid == 204:
                p.preflight(model, video, args.work)
                ids = processor.tokenizer("What is visible?", return_tensors="pt").input_ids.to(model.device)
                q.preflight(model, bank, meta, ids, args.work)
            for qi in range(len(video["conversations"])):
                if done and args.max_hours and time.perf_counter() - started > args.max_hours * 3600:
                    break
                h, r = s.split_frames(video, qi)
                key = d.fingerprint(dict(config=config_hash, feature=d.fingerprint(meta), sample=sid, qa=qi, h=h, r=r))
                dest = args.work / "checkpoints" / f"sample_{sid}_q{qi}.json"
                if dest.exists():
                    saved = json.loads(dest.read_text())
                    if saved["key"] != key or d.fingerprint(saved["rows"]) != saved["rows_sha256"]:
                        raise d.RunConflict("Construction selection checkpoint differs; use new WORK")
                    block = saved["rows"]
                else:
                    p.report_progress("selection_question", sample_id=sid, qa_index=qi)
                    block, plan = question(model, analyzer, bank, meta, video, qi, h, r, old)
                    p.atomic_json(args.work / "preparation" / dest.name, plan)
                    p.atomic_json(dest, dict(key=key, rows=block, rows_sha256=d.fingerprint(block)))
                if any(row["reference_matches"] is False for row in block):
                    raise ValueError("Historical all-history or recent-four choice changed; inspect saved checkpoint")
                if not h and len({row["pred_choice"] for row in block}) != 1:
                    raise ValueError("Empty-history null control disagreed")
                rows.extend(block)
                done += 1
                export(rows, args.work)
                print(f"QUESTION_BLOCK_DONE {done}/40", flush=True)
            del bank
            if args.max_hours and time.perf_counter() - started > args.max_hours * 3600:
                break
    status = "complete" if done == 40 else "partial_time_budget"
    if len(rows) != len(ARMS) * done:
        raise ValueError("Construction selection matrix incomplete")
    p.atomic_json(args.work / "exports/completion.json", dict(status=status, main_questions=done, main_answers=len(rows),
                   config_hash=config_hash, elapsed_post_model_s=time.perf_counter() - started))
    p.report_progress(status, main_questions=done)
    package(args.work, status)
    print(f"CONSTRUCTION_SELECTION_{status.upper()}", flush=True)


def package(work, status):
    original = s.ARTIFACT
    try:
        s.ARTIFACT = ARTIFACT
        s.package(work, status)
    finally:
        s.ARTIFACT = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "work", "model", "reuse"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--max-hours", type=float, default=1.0)
    args = parser.parse_args()
    if args.max_hours < 0:
        parser.error("Invalid soft budget")
    d.validate_workspaces(args.work, args.reuse)
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
