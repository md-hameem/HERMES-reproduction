"""Evaluate endpoint-preserving retrieval on a new TwinPast development screen."""
import argparse
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path
import time

import automatic_runner as auto

base = auto.base
backbone = auto.backbone
ARTIFACT = "HERMES_twinpast_temporal_windows_evidence.zip"
METHODS = ("recent", "uniform", "individual", "endpoints")
WINDOW_SPAN = 3
WINDOW_CAP = 12


def decode_grid(path, grid):
    import numpy as np
    from decord import VideoReader

    reader = VideoReader(str(path), num_threads=1)
    times = np.asarray(reader.get_frame_timestamp(list(range(len(reader)))), dtype=np.float64)
    times -= times[0, 0]
    if grid is None:
        # Source-screening frames never enter the policy.
        grid = range(0, math.ceil(times[-1, 1]), 2)
    grid = np.asarray(list(grid))
    if grid[-1] > times[-1, 1] + 1e-6:
        raise ValueError("Clip ends before the frozen grid")
    indices = np.searchsorted(times[:, 0], grid, side="right") - 1
    if indices[0] < 0 or not np.all(np.diff(indices) > 0):
        raise ValueError("Clip does not contain distinct frames on the frozen grid")
    frames = reader.get_batch(indices.tolist()).asnumpy()
    return frames, dict(grid_s=grid.tolist(), indices=indices.tolist(), timestamps_s=times[indices, 0].tolist(),
                        hashes=[base.sha(frame.tobytes()) for frame in frames])


def fingerprints(path):
    import numpy as np
    from PIL import Image

    frames, trace = decode_grid(path, None)
    signatures = []
    for frame, sha in zip(frames, trace["hashes"], strict=True):
        image = Image.fromarray(frame)
        gray = np.asarray(image.convert("L").resize((9, 8)))
        bits = int.from_bytes(np.packbits((gray[:, 1:] > gray[:, :-1]).reshape(-1)).tobytes(), "big")
        rgb = np.asarray(image.resize((4, 4)), dtype=int).reshape(-1)
        if gray.std() >= 8:
            signatures.append((sha, bits, rgb))
    return signatures


def shared_frames(a, b):
    """Flag repeat footage on at least three distinct sampled frame matches."""
    import numpy as np

    matches = set()
    for sha_a, hash_a, rgb_a in a:
        for sha_b, hash_b, rgb_b in b:
            if sha_a == sha_b or ((hash_a ^ hash_b).bit_count() <= 4 and np.abs(rgb_a - rgb_b).mean() <= 6):
                matches.add(sha_b)
                break
    return len(matches)


def screen_pairs(cases, excluded, paths, max_pairs):
    signature_bank = {pair: {b: fingerprints(paths[pair][b]) for b in "ab"} for pair in excluded}
    decisions, accepted = [], []
    for case in cases:
        pair = case["pair_id"]
        if len(accepted) == max_pairs:
            break
        grid = list(range(math.ceil(case["query_s"])))
        try:
            recent = {b: decode_grid(paths[pair][b], grid[-4:])[1]["hashes"] for b in "ab"}
        except ValueError as exc:
            decisions.append(dict(pair_id=pair, disposition="invalid_grid", detail=str(exc)))
            continue
        if recent["a"] != recent["b"]:
            decisions.append(dict(pair_id=pair, disposition="recent_context_mismatch"))
            continue
        candidate = {b: fingerprints(paths[pair][b]) for b in "ab"}
        overlaps = []
        for reference, signature in signature_bank.items():
            matches = max(shared_frames(candidate[b], signature[c]) for b in "ab" for c in "ab")
            if matches >= 3:
                overlaps.append(dict(reference=reference, sampled_matches=matches))
        if overlaps:
            decisions.append(dict(pair_id=pair, disposition="possible_media_overlap", matches=overlaps))
            continue
        signature_bank[pair] = candidate
        accepted.append(case)
        decisions.append(dict(pair_id=pair, disposition="accepted", recent_hashes=recent["a"]))
    return accepted, decisions


def select_windows(windows):
    order = sorted(windows, key=lambda row: row["slots"])
    compatible = (group for group in combinations(order, 3)
                  if all(left["slots"][-1] < right["slots"][0] for left, right in zip(group, group[1:])))
    best = max(compatible, key=lambda group: sum(r["relevance"] for r in group))
    return [i for r in best for i in r["slots"]]


def branch_policy(model, path, case):
    import torch

    tick = time.perf_counter()
    frames, sample = decode_grid(path, range(math.ceil(case["query_s"])))
    assert sample["timestamps_s"][-1] < case["query_s"]
    bank = auto.encode_frames(model, frames)
    num_history = len(frames) - 4
    recent = list(range(num_history, len(frames)))
    system = model.language_model.get_input_embeddings()(torch.as_tensor(model.init_prompt_ids, device=model.device))
    log, cache = [], {}

    def score(prompt, slots, phase, order):
        ids, option_ids, permutation, prompt_hash = prompt
        key = (prompt_hash, tuple(slots))
        if key not in cache:
            visual = torch.cat([bank[i] for i in slots], dim=1)
            started = time.perf_counter()
            values, tokens = base.score(model, system, ids, option_ids, visual)
            canonical = [0.0] * len(permutation)
            for i, value in zip(permutation, values.tolist(), strict=True):
                canonical[i] = value
            cache[key] = canonical
            log.append(dict(phase=phase, order=order, slots=slots, forward_s=time.perf_counter()-started,
                            lm_tokens=tokens, visual_tokens=visual.shape[1]))
        return cache[key]

    question = case["question"]
    probe = ("Does this earlier excerpt contain visual evidence needed to answer the following "
             "question about the specified event, state, or entity? " + question)
    prompts = {order: base.make_prompt(model, probe, ["Yes", "No"], order, True) for order in base.ORDERS}
    windows, individual_scores = [], {}
    for start in auto.spread(num_history - WINDOW_SPAN, WINDOW_CAP):
        endpoints = [start, start + WINDOW_SPAN]
        vectors = {o: score(prompts[o], endpoints, "retrieval", o) for o in base.ORDERS}
        value = sum(v[0] - v[1] for v in vectors.values()) / len(vectors)
        windows.append(dict(slots=endpoints, relevance=value, yes_no=vectors))
        for i in endpoints:
            individual_scores[i] = max(value, individual_scores.get(i, -math.inf))
    selection = dict(recent=recent, uniform=auto.spread(num_history, 6) + recent,
                     individual=sorted(sorted(individual_scores, key=lambda i: (-individual_scores[i], i))[:6]) + recent,
                     endpoints=select_windows(windows) + recent)
    assert len(selection["recent"]) == 4 and all(len(selection[m]) == 10 for m in METHODS[1:])
    assert all(slots == sorted(set(slots)) for slots in selection.values())
    answers = {}
    for o in base.ORDERS:
        prompt = base.make_prompt(model, question, case["options"], o, True)
        vectors = {m: score(prompt, selection[m], m, o) for m in METHODS}
        answers[o] = dict(logits=vectors, predictions={m: base.LETTERS[auto.best(v)] for m, v in vectors.items()})
    return dict(sample=sample, windows=windows, selection=selection, answers=answers,
                calls=log, wall_s=time.perf_counter()-tick)


def evaluate(blocks, targets):
    methods = {}
    for method in METHODS:
        pairs, correct = [], 0
        for block in blocks:
            p = block["pair_id"]
            wins = [block["branches"][b]["answers"][o]["predictions"][method] == targets[p][f"answer_{b}"]
                    for b in "ab" for o in base.ORDERS]
            correct += sum(wins)
            if all(wins):
                pairs.append(p)
        methods[method] = dict(robust_pairs=pairs, correct_branch_orders=correct)
    endpoints = set(methods["endpoints"]["robust_pairs"])
    matched = {m: dict(recoveries=sorted(endpoints - set(methods[m]["robust_pairs"])),
                       regressions=sorted(set(methods[m]["robust_pairs"]) - endpoints))
               for m in ("uniform", "individual")}
    calls = Counter(call["phase"] for p in blocks for b in p["branches"].values() for call in b["calls"])
    return dict(pairs=len(blocks), methods=methods, matched_pair_changes=matched,
                endpoints_more_robust_than_both_dense_baselines=len(methods["endpoints"]["robust_pairs"]) >
                max(len(methods[m]["robust_pairs"]) for m in ("uniform", "individual")),
                forward_calls=sum(calls.values()), forward_calls_by_phase=dict(calls))


def execute(args):
    import torch
    from inference.llavaov_hermes import load_model

    work = args.work
    provenance = work / "provenance"
    spec = json.loads((provenance / "temporal_cases.json").read_bytes())
    assert args.model.name == backbone.MODEL_REVISION == spec["model_revision"]
    paths = backbone.history.acquire(args, spec["media_source"], spec["media_plan"])
    cases, screening = screen_pairs(spec["cases"], spec["excluded"], paths, spec["expected_max_pairs"])
    exports = work / "exports"
    base.atomic(exports / "source_screen.json", dict(decisions=screening, source_status=spec["source_status"]))
    if not cases:
        raise RuntimeError("No pairs passed the shared-input and media-overlap screen")
    print(f"SOURCE_SCREEN_PASS {len(cases)}/{spec['expected_max_pairs']} pairs", flush=True)
    model, processor = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=0.5)
    assert model.dtype == torch.float16 and {str(p.device) for p in model.parameters()} == {"cuda:0"}
    assert json.loads(json.dumps(processor.video_processor.to_dict())) == spec["video_processor"]
    config = dict(model_revision=args.model.name, cuda=torch.version.cuda, torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(0), accepted=[c["pair_id"] for c in cases],
                  window_span=WINDOW_SPAN, window_cap=WINDOW_CAP,
                  hashes={p.name: base.digest(p) for p in sorted(provenance.iterdir()) if p.is_file()})
    base.atomic(exports / "run_config.json", config)
    identity = base.sha(json.dumps(config, sort_keys=True).encode())
    checkpoints = work / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    blocks = []
    with torch.inference_mode():
        for i, case in enumerate(cases):
            pair = case["pair_id"]
            path = checkpoints / f"{pair}.json"
            if path.is_file():
                saved = json.loads(path.read_bytes())
                assert saved["identity"] == identity and saved["block"]["pair_id"] == pair
                block = saved["block"]
                print(f"PAIR_REUSED {i+1}/{len(cases)} {pair}", flush=True)
            else:
                torch.cuda.reset_peak_memory_stats()
                branches = {b: branch_policy(model, paths[pair][b], case) for b in "ab"}
                assert branches["a"]["sample"]["hashes"][-4:] == branches["b"]["sample"]["hashes"][-4:]
                assert all(branches["a"]["answers"][o]["predictions"]["recent"] ==
                           branches["b"]["answers"][o]["predictions"]["recent"] for o in base.ORDERS)
                block = dict(pair_id=pair, branches=branches, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
                base.atomic(path, dict(identity=identity, block=block))
                print(f"PAIR_DONE {i+1}/{len(cases)} {pair}", flush=True)
            blocks.append(block)
    targets = json.loads((provenance / "evaluation_targets.json").read_bytes())
    report = evaluate(blocks, targets)
    base.atomic(exports / "summary.json", report)
    base.atomic(work / "completion.json", dict(status="complete", pairs=len(blocks)))
    print("TWINPAST_TEMPORAL_WINDOWS_COMPLETE", json.dumps(report), flush=True)


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
