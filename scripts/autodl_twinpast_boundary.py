"""Evaluate the frozen TwinPast scene-boundary policy against matched baselines."""
import argparse
import json
from pathlib import Path
from statistics import median
import time

import numpy as np
from decord import VideoReader

from readout_runner import REPLAY_ATOL, acquire, auto, base, choose
from twinpast_boundary_policy import FPS, THRESHOLD, pick, screen_candidates, select

ARTIFACT = "HERMES_twinpast_boundary_evidence.zip"
METHODS = ("coarse_uniform", "local_uniform", "boundary")
REPLAY_METHODS = METHODS[:2]


def timed(operation, device):
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    value = operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return value, time.perf_counter() - start


def retrieval(model, system, coarse, question, candidates):
    import torch

    probe = ("Does this earlier excerpt contain visual evidence needed to answer the following "
             "question about the specified event, state, or entity? " + question)
    prompts = {order: base.make_prompt(model, probe, ["Yes", "No"], order, True) for order in base.ORDERS}
    ranked, checks = [], []
    for candidate in candidates:
        start, end = candidate["slots"]
        visual = torch.cat((coarse[start], coarse[end]), dim=1)
        result = {}
        for order, (ids, labels, perm, _) in prompts.items():
            values, _ = base.score(model, system, ids, labels, visual)
            canonical = [0.0] * len(perm)
            for i, value in zip(perm, values.tolist(), strict=True):
                canonical[i] = value
            result[order] = canonical
            expected = candidate["yes_no"][order]
            checks.append(max(abs(a-b) for a, b in zip(canonical, expected, strict=True)))
        relevance = sum(v[0]-v[1] for v in result.values()) / 2
        ranked.append(dict(slots=candidate["slots"], relevance=relevance, yes_no=result))
    return ranked, max(checks)


def boundary_inputs(path, candidates, cutoff):
    reader = VideoReader(str(path), num_threads=1)
    stamps = np.asarray(reader.get_frame_timestamp(list(range(len(reader)))), dtype=np.float64)
    starts = stamps[:, 0] - stamps[0, 0]
    windows, scanned = screen_candidates(reader, starts, candidates, cutoff)
    original, constrained = select(windows)
    indices = pick(constrained, True)
    frames = reader.get_batch(indices).asnumpy()
    return dict(original_windows=[list(w["window"]) for w in original],
                constrained_windows=[list(w["window"]) for w in constrained], indices=indices,
                hashes=[base.sha(frame.tobytes()) for frame in frames],
                timestamps_s=starts[indices].tolist(), scanned_indices=scanned, frames=frames)


def compare(case, side, model, system, path):
    import torch

    archived = case["branches"][side]
    coarse, local = archived["coarse"], archived["local"]
    seconds = {}
    bank, seconds["coarse_decode_encode"] = timed(
        lambda: auto.encode_frames(model, base.decode(
            path, dict(source_frames=coarse["indices"], decoded_rgb_sha256=coarse["hashes"])
        )), model.device,
    )
    recent = bank[-4:]
    (candidates, retrieval_delta), seconds["question_relevance"] = timed(
        lambda: retrieval(model, system, bank, case["question"], archived["candidates"]), model.device
    )
    if retrieval_delta > REPLAY_ATOL:
        raise ValueError(f"Retrieval replay differs for {case['pair_id']}/{side}: {retrieval_delta}")
    cut, seconds["boundary_scan_and_select"] = timed(
        lambda: boundary_inputs(path, candidates, coarse["indices"][-4]), model.device
    )
    frozen = archived["boundary"]
    for field in ("original_windows", "constrained_windows", "indices"):
        if cut[field] != frozen[field]:
            raise ValueError(
                f"Frozen selection differs for {case['pair_id']}/{side}/{field}: "
                f"expected {frozen[field]}, got {cut[field]}"
            )
    if cut["scanned_indices"] != frozen["scanned_indices"]:
        raise ValueError(f"Boundary probe frame count differs for {case['pair_id']}/{side}")

    def encode_history(frames):
        return torch.cat(auto.encode_frames(model, frames)+recent, dim=1)

    def local_input():
        frames = base.decode(path, dict(source_frames=local["indices"], decoded_rgb_sha256=local["hashes"]))
        return encode_history(frames[local["slots"]])

    visuals = {}
    for method, operation in (
        ("coarse_uniform", lambda: torch.cat([bank[i] for i in coarse["slots"]]+recent, dim=1)),
        ("local_uniform", local_input),
        ("boundary", lambda: encode_history(cut["frames"])),
    ):
        visuals[method], seconds[f"{method}_history"] = timed(operation, model.device)
    assert len({tuple(item.shape) for item in visuals.values()}) == 1
    rows, failures = {}, []
    for order in base.ORDERS:
        rows[order] = {}
        for method in METHODS:
            row, elapsed = timed(lambda: choose(model, system, visuals[method], case["question"],
                                                case["options"], order), model.device)
            row["synchronized_answer_s"] = elapsed
            if method in REPLAY_METHODS:
                expected = archived["direct"][method][order]
                delta = max(abs(a-b) for a, b in zip(row["logits"], expected["logits"], strict=True))
                row["replay_max_abs"] = delta
                if delta > REPLAY_ATOL or any(row[k] != expected[k] for k in
                                               ("prediction", "prompt_sha256", "visual_tokens", "lm_tokens")):
                    failures.append(dict(frame_set=method, order=order, expected=expected, actual=row))
            rows[order][method] = row
    cut.pop("frames")
    return dict(answers=rows, stages_s=seconds, boundary=cut, retrieval_max_abs=retrieval_delta,
                replay_failures=failures)


def summarize(blocks, targets):
    methods = {}
    for method in METHODS:
        robust, correct, totals = [], 0, []
        extra = {"coarse_uniform": (), "local_uniform": ("question_relevance",),
                 "boundary": ("question_relevance", "boundary_scan_and_select")}[method]
        for block in blocks:
            pair = block["pair_id"]
            pair_correct = 0
            for side, trace in block["branches"].items():
                scores = [trace["answers"][order][method] for order in base.ORDERS]
                pair_correct += sum(row["prediction"] == targets[pair][f"answer_{side}"] for row in scores)
                stages = trace["stages_s"]
                total = sum(stages[name] for name in ("coarse_decode_encode", f"{method}_history", *extra))
                totals.append(total + sum(row["synchronized_answer_s"] for row in scores) / len(scores))
            correct += pair_correct
            if pair_correct == 4:
                robust.append(pair)
        methods[method] = dict(robust_pairs=robust, correct_branch_orders=correct,
                               median_stage_sum_per_branch_s=median(totals))
    boundary = set(methods["boundary"]["robust_pairs"])
    paired = {m: dict(recoveries=sorted(boundary-set(methods[m]["robust_pairs"])),
                      regressions=sorted(set(methods[m]["robust_pairs"])-boundary)) for m in REPLAY_METHODS}
    primary = paired["coarse_uniform"]
    return dict(pairs=len(blocks), scores=len(blocks)*2*len(base.ORDERS)*len(METHODS), methods=methods,
                boundary_against=paired, development_criterion_met=len(blocks)==8 and bool(primary["recoveries"])
                and not primary["regressions"])


def execute(args):
    import torch
    from inference.llavaov_hermes import load_model

    work = args.work
    provenance = work / "provenance"
    spec = json.loads((provenance / "boundary_cases.json").read_bytes())
    assert args.model.name == spec["model_revision"]
    assert len(spec["cases"]) == 8 and spec["cut_threshold"] == THRESHOLD and spec["cut_fps"] == FPS
    paths = acquire(args, spec)
    model, processor = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=0.5)
    assert model.dtype == torch.float16 and {str(p.device) for p in model.parameters()} == {"cuda:0"}
    assert json.loads(json.dumps(processor.video_processor.to_dict())) == spec["video_processor"]
    exports = work / "exports"
    config = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  model_revision=args.model.name, replay_atol=REPLAY_ATOL,
                  refinement_archive_sha256=spec["source_archive_sha256"], cpu_audit_sha256=spec["cpu_audit_sha256"])
    base.atomic(exports / "run_config.json", config)
    blocks = []
    with torch.inference_mode():
        system = model.language_model.get_input_embeddings()(torch.as_tensor(model.init_prompt_ids, device=model.device))
        for i, case in enumerate(spec["cases"], 1):
            pair = case["pair_id"]
            block = dict(pair_id=pair, branches={side: compare(case, side, model, system, paths[pair, side])
                                                  for side in "ab"})
            base.atomic(work / "checkpoints" / f"{pair}.json", block)
            blocks.append(block)
            print(f"BOUNDARY_PAIR {i}/8 {pair}", flush=True)
    branches = [branch for block in blocks for branch in block["branches"].values()]
    failures = [dict(pair_id=block["pair_id"], branch=side, **failure) for block in blocks
                for side, branch in block["branches"].items() for failure in branch["replay_failures"]]
    archived_scores = [result[m]["replay_max_abs"] for branch in branches for result in branch["answers"].values()
                       for m in REPLAY_METHODS]
    base.atomic(exports / "replay_gate.json", dict(scores=len(archived_scores), max_abs=max(archived_scores),
                retrieval_max_abs=max(b["retrieval_max_abs"] for b in branches), failures=failures))
    if failures:
        raise RuntimeError("Archived answer replay failed. See exports/replay_gate.json.")
    targets = json.loads((provenance / "evaluation_targets.json").read_bytes())
    assert set(targets) == {block["pair_id"] for block in blocks}
    report = summarize(blocks, targets)
    assert report["scores"] == 96
    base.atomic(exports / "summary.json", report)
    base.atomic(work / "completion.json", dict(status="complete", pairs=len(blocks)))
    print("TWINPAST_BOUNDARY_COMPLETE", json.dumps(report), flush=True)


def main():
    from filelock import FileLock

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "model", "previous"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    work = args.work
    with FileLock(str(work / "run.lock"), timeout=0):
        for name in ("exports", "checkpoints"):
            (work / name).mkdir(exist_ok=True)
        spec = json.loads((work / "provenance/boundary_cases.json").read_bytes())
        for case in spec["cases"]:
            (work / "checkpoints" / f"{case['pair_id']}.json").unlink(missing_ok=True)
        for name in ("summary.json", "replay_gate.json", "run_config.json"):
            (work / "exports" / name).unlink(missing_ok=True)
        base.ARTIFACT = ARTIFACT
        base.atomic(work / "completion.json", dict(status="running"))
        try:
            execute(args)
        except BaseException as exc:
            base.atomic(work / "completion.json", dict(status="failed", error_type=type(exc).__name__, message=str(exc)))
            raise
        finally:
            base.package(work)


if __name__ == "__main__":
    main()
