"""Run a frozen, annotation-blind retrieval and answer-correction screen."""
import argparse
import json
import math
from pathlib import Path
import time

import backbone_runner as backbone

base = backbone.base
ARTIFACT = "HERMES_twinpast_automatic_evidence.zip"
MAX_CANDIDATES = 13
MAX_HISTORY = 6
MIN_LIFT = 0.5
METHODS = ("recent", "latest", "uniform", "relevance", "margin_gate", "history_control_gate")


def spread(count, budget):
    if count <= budget:
        return list(range(count))
    return [i * (count - 1) // (budget - 1) for i in range(budget)]


def sample_prefix(path, query):
    import numpy as np
    from decord import VideoReader

    reader = VideoReader(str(path), num_threads=1)
    times = np.asarray(reader.get_frame_timestamp(list(range(len(reader)))), dtype=np.float64)
    times -= times[0, 0]
    grid = list(range(0, math.ceil(query), 2))
    assert len(grid) >= 8 and grid[-1] < query and grid[-1] <= times[-1, 1] + 1e-6
    older_grid, recent_grid = grid[:-4], grid[-4:]
    picks = [older_grid[i] for i in spread(len(older_grid), MAX_CANDIDATES)] + recent_grid
    indices = np.searchsorted(times[:, 0], picks, side="right") - 1
    sampled_s = times[indices, 0]
    assert indices[0] >= 0 and np.all(np.diff(indices) > 0)
    assert np.all(sampled_s <= picks) and sampled_s[-1] < query
    frames = reader.get_batch(indices.tolist()).asnumpy()
    return frames, dict(grid_s=picks, indices=indices.tolist(), timestamps_s=sampled_s.tolist(),
                        hashes=[base.sha(f.tobytes()) for f in frames])


def encode_frames(model, frames):
    """Vision encoding is frame-local. Limit the encoder batch to four frames."""
    bank = []
    for start in range(0, len(frames), 4):
        batch = frames[start:start + 4]
        encoded = base.features(model, batch)
        bank.extend(encoded.split(encoded.shape[1] // len(batch), dim=1))
    assert len(bank) == len(frames)
    return bank


def best(scores):
    return max(range(len(scores)), key=scores.__getitem__)


def answer_policies(recent, direct, uniform, same_history):
    initial, alternative = best(recent), best(direct)
    margin = direct[alternative] - direct[initial]
    lift = margin - (recent[alternative] - recent[initial])
    specificity = margin - (uniform[alternative] - uniform[initial])
    accept = alternative != initial and lift >= MIN_LIFT
    controlled = accept and not same_history and specificity >= MIN_LIFT
    return dict(relevance=alternative, margin_gate=alternative if accept else initial,
                history_control_gate=alternative if controlled else initial,
                scores=dict(history_margin=margin, lift_from_recent=lift, lift_from_uniform=specificity))


def branch_policy(model, path, query, question, options):
    """Infer for one branch with no evidence labels, gold, or counterpart data."""
    import torch

    started = time.perf_counter()
    frames, sample = sample_prefix(path, query)
    bank = encode_frames(model, frames)
    history_count = len(bank) - 4
    size = min(MAX_HISTORY, history_count)
    recent = list(range(history_count, len(bank)))
    uniform = spread(history_count, size)
    latest = list(range(history_count - size, history_count))
    embedding = model.language_model.get_input_embeddings()
    system = embedding(torch.as_tensor(model.init_prompt_ids, device=model.device))
    calls = []
    logits_cache = {}

    def score(prompt, window, phase, order):
        ids, option_ids, perm, digest = prompt
        key = digest, tuple(window)
        if key not in logits_cache:
            visual = torch.cat([bank[i] for i in window], dim=1)
            tick = time.perf_counter()
            logits, length = base.score(model, system, ids, option_ids, visual)
            values = [0.0] * len(perm)
            for i, value in zip(perm, logits.tolist(), strict=True):
                values[i] = value
            logits_cache[key] = values
            calls.append(dict(phase=phase, order=order, slots=window, forward_s=time.perf_counter()-tick,
                              visual_tokens=visual.shape[1], lm_tokens=length, prompt_sha256=digest))
        return logits_cache[key]

    probe = ("Does this earlier excerpt contain visual evidence needed to answer the following "
             "question about the specified event, state, or entity? " + question)
    prompts = {o: base.make_prompt(model, probe, ["Yes", "No"], o, True) for o in base.ORDERS}
    candidates = []
    relevance = [-math.inf] * history_count
    for slot in range(history_count - 1):
        window = [slot, slot + 1]
        probe_scores = {o: score(prompts[o], window, "retrieval", o) for o in base.ORDERS}
        average = sum(v[0] - v[1] for v in probe_scores.values()) / len(probe_scores)
        candidates.append(dict(slots=window, yes_no=probe_scores, relevance=average))
        for i in window:
            relevance[i] = max(relevance[i], average)
    ranked = sorted(sorted(range(history_count), key=lambda i: (-relevance[i], i))[:size])
    assert len(set(ranked)) == len(uniform) == len(latest) == size
    selection = {"recent": recent, "latest": latest + recent,
                 "uniform": uniform + recent, "relevance": ranked + recent}
    answers = {}
    for order in base.ORDERS:
        prompt = base.make_prompt(model, question, options, order, True)
        vectors = {arm: score(prompt, slots, arm, order) for arm, slots in selection.items()}
        decision = answer_policies(vectors["recent"], vectors["relevance"], vectors["uniform"], ranked == uniform)
        choices = {arm: best(values) for arm, values in vectors.items()}
        choices.update({k: decision[k] for k in ("margin_gate", "history_control_gate")})
        answers[order] = dict(logits=vectors, predictions={arm: base.LETTERS[i] for arm, i in choices.items()},
                              margins=decision["scores"], prompt_sha256=prompt[3])
    return dict(sample=sample, history_budget=size, candidates=candidates, selection=selection, answers=answers,
                calls=calls, wall_s=time.perf_counter()-started)


def summarize(blocks, targets):
    report = {}
    opened = {arm: 0 for arm in ("margin_gate", "history_control_gate")}
    gate_wins, gate_losses = 0, 0
    for arm in METHODS:
        robust, wins, losses = [], 0, 0
        for block in blocks:
            pair_id = block["pair_id"]
            success = True
            for order in base.ORDERS:
                for branch in "ab":
                    values = block["branches"][branch]["answers"][order]["predictions"]
                    gold = targets[pair_id][f"answer_{branch}"]
                    correct = values[arm] == gold
                    reference = values["recent"] == gold
                    wins += correct and not reference
                    losses += reference and not correct
                    success = success and correct
                    if arm in opened:
                        opened[arm] += values[arm] != values["recent"]
                    if arm == "history_control_gate":
                        direct_correct = values["relevance"] == gold
                        gate_wins += correct and not direct_correct
                        gate_losses += direct_correct and not correct
            if success:
                robust.append(pair_id)
        report[arm] = dict(robust_pairs=robust, recoveries_vs_recent=wins, regressions_vs_recent=losses)
    return dict(pairs=len(blocks), branch_order_answers=len(blocks)*4, methods=report, overrides=opened,
                controlled_gate_vs_same_frames_direct=dict(recoveries=gate_wins, regressions=gate_losses),
                total_forward_calls=sum(len(v["calls"]) for b in blocks for v in b["branches"].values()))


def execute(args):
    import torch
    from inference.llavaov_hermes import load_model

    work = args.work
    provenance = work / "provenance"
    spec = json.loads((provenance / "automatic_cases.json").read_bytes())
    assert args.model.name == backbone.MODEL_REVISION == spec["model_revision"]
    cases = {c["pair_id"]: c for c in spec["cases"]}
    assert len(cases) == 7
    paths = backbone.history.acquire(args, spec["media_source"], spec["media_plan"])
    model, processor = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=0.5)
    assert model.dtype == torch.float16 and {str(p.device) for p in model.parameters()} == {"cuda:0"}
    assert json.loads(json.dumps(processor.video_processor.to_dict())) == spec["video_processor"]
    config = dict(model_revision=args.model.name, torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0), min_lift=MIN_LIFT, max_candidates=MAX_CANDIDATES,
                  max_history=MAX_HISTORY, source_archive_sha256=spec["fixed_recent_archive_sha256"],
                  hashes={p.name: base.digest(p) for p in sorted(provenance.iterdir()) if p.is_file()})
    identity = base.sha(json.dumps(config, sort_keys=True).encode())
    exports = work / "exports"
    base.atomic(exports / "run_config.json", config)
    checkpoints = work / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    blocks = []
    with torch.inference_mode():
        for i, case in enumerate(spec["cases"]):
            pair_id = case["pair_id"]
            checkpoint = checkpoints / f"{pair_id}.json"
            if checkpoint.is_file():
                saved = json.loads(checkpoint.read_bytes())
                assert saved["identity"] == identity and saved["block"]["pair_id"] == pair_id
                blocks.append(saved["block"])
                print(f"PAIR_REUSED {i+1}/7 {pair_id}", flush=True)
                continue
            torch.cuda.reset_peak_memory_stats()
            branches = {b: branch_policy(model, paths[pair_id][b], case["query_s"], case["question"], case["options"])
                        for b in "ab"}
            left, right = branches["a"], branches["b"]
            assert left["sample"]["grid_s"] == right["sample"]["grid_s"]
            assert left["sample"]["hashes"][-4:] == right["sample"]["hashes"][-4:] == case["recent_hashes"]
            assert left["history_budget"] == right["history_budget"]
            for order in base.ORDERS:
                assert left["answers"][order]["predictions"]["recent"] == right["answers"][order]["predictions"]["recent"]
            block = dict(pair_id=pair_id, branches=branches,
                         peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
            base.atomic(checkpoint, dict(identity=identity, block=block))
            blocks.append(block)
            print(f"PAIR_DONE {i+1}/7 {pair_id}", flush=True)
    # Ground truth is loaded only after all branch predictions have been written.
    targets = json.loads((provenance / "evaluation_targets.json").read_bytes())
    assert targets.keys() == cases.keys()
    report = summarize(blocks, targets)
    coverage = []
    for block in blocks:
        pair_id = block["pair_id"]
        for branch, trace in block["branches"].items():
            intervals = targets[pair_id]["evidence_intervals"][branch.upper()]
            timestamps = trace["sample"]["timestamps_s"]
            inside = [i for i, t in enumerate(timestamps[:-4]) if any(lo <= t <= hi for lo, hi in intervals)]
            coverage.append(dict(pair_id=pair_id, branch=branch, candidate_slots_in_evidence=inside,
                                 selected_slots_in_evidence={name: sorted(set(inside) & set(trace["selection"][name]))
                                                             for name in ("latest", "uniform", "relevance")}))
    base.atomic(exports / "postrun_evidence_overlap.json", coverage)
    base.atomic(exports / "summary.json", report)
    base.atomic(work / "completion.json", dict(status="complete", pairs=len(blocks)))
    print("TWINPAST_AUTOMATIC_COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
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
