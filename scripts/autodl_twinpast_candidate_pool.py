"""Compare capped and 1 FPS causal pools using the same annotation-guided diagnostic."""
import argparse
import json
from pathlib import Path

import backbone_runner as backbone

base = backbone.base
ARTIFACT = "HERMES_twinpast_candidate_pool_evidence.zip"
REPLAY_ATOL = 0.05
ARMS = ("curated_replay", "capped_pool", "dense_pool")


def pick_history(timestamps, intervals):
    eligible = [i for i, t in enumerate(timestamps) if any(lo - 1e-6 <= t <= hi + 1e-6 for lo, hi in intervals)]
    if len(eligible) >= 6:
        picks = [eligible[i * (len(eligible) - 1) // 5] for i in range(6)]
    else:
        outside = [i for i in range(len(timestamps)) if i not in eligible]
        picks = eligible + outside[:6 - len(eligible)]
    assert len(set(picks)) == 6
    return sorted(picks)


def materialize(path, aged_grid, intervals, shared_recent_start):
    import numpy as np
    from decord import VideoReader

    reader = VideoReader(str(path), num_threads=1)
    seconds = np.asarray(reader.get_frame_timestamp(list(range(len(reader)))), dtype=np.float64)[:, 0]
    seconds -= seconds[0]
    grid = np.asarray(aged_grid)
    indices = np.searchsorted(seconds, grid, side="right") - 1
    assert np.all(indices >= 0) and np.all(np.diff(indices) > 0)
    assert np.all(seconds[indices] <= grid) and grid[-1] < shared_recent_start
    picks = pick_history(seconds[indices], intervals)
    chosen = indices[picks].tolist()
    frames = reader.get_batch(chosen).asnumpy()
    trace = dict(grid_s=grid.tolist(), candidate_timestamps_s=seconds[indices].tolist(),
                 chosen_indices=chosen, chosen_timestamps_s=seconds[chosen].tolist(),
                 chosen_hashes=[base.sha(f.tobytes()) for f in frames],
                 chosen_in_intervals=sum(any(lo - 1e-6 <= seconds[i] <= hi + 1e-6 for lo, hi in intervals) for i in chosen))
    return frames, trace


def evaluate_pair(model, paths, case, smoke=False):
    import numpy as np
    import torch

    annotation = case["annotation"]
    curated = backbone.reproduce_windows(paths, case["curated_rows"])
    recent = {b: base.decode(paths[b], case["recent"][b]) for b in "ab"}
    assert np.array_equal(recent["a"], recent["b"])
    recent_features = base.features(model, recent["a"])
    system = model.language_model.get_input_embeddings()(torch.as_tensor(model.init_prompt_ids, device=model.device))
    original = {(r["order"], r["branch"]): r for r in case["curated_rows"]}
    inputs, sampling = {}, {}
    for b in "ab":
        assert curated[b]["seconds"][-1] < case["recent"][b]["start_s"]
        intervals = annotation["evidence_intervals"][b.upper()]
        capped, capped_trace = materialize(paths[b], case["capped_grid_s"], intervals, case["recent_grid_s"][0])
        dense, dense_trace = materialize(paths[b], case["dense_grid_s"], intervals, case["recent_grid_s"][0])
        assert max(capped_trace["chosen_timestamps_s"] + dense_trace["chosen_timestamps_s"]) < case["recent"][b]["start_s"]
        inputs[b] = {name: torch.cat((base.features(model, frames), recent_features), dim=1)
                     for name, frames in zip(ARMS, (curated[b]["frames"], capped, dense), strict=True)}
        assert len({tuple(v.shape) for v in inputs[b].values()}) == 1
        sampling[b] = {"capped_pool": capped_trace, "dense_pool": dense_trace}
    assert inputs["a"]["curated_replay"].shape == inputs["b"]["curated_replay"].shape
    rows = {arm: [] for arm in ARMS}
    replay = []
    for order in base.ORDERS:
        ids, option_ids, permutation, prompt_hash = base.make_prompt(
            model, annotation["question"], annotation["options"], order, True)
        for b in "ab":
            old = original[order, b]
            assert old["prompt_sha256"] == prompt_hash and old["option_token_ids"] == option_ids
            for arm in ARMS:
                output, tokens = base.score(model, system, ids, option_ids, inputs[b][arm])
                assert tokens == old["lm_tokens"]
                if smoke and arm == "dense_pool":
                    repeated, count = base.score(model, system, ids, option_ids, inputs[b][arm])
                    torch.testing.assert_close(output, repeated, rtol=0, atol=0)
                    assert count == tokens
                scores = [0.0] * len(permutation)
                for i, value in zip(permutation, output.tolist(), strict=True):
                    scores[i] = value
                predicted = base.LETTERS[max(range(len(scores)), key=scores.__getitem__)]
                rows[arm].append(dict(order=order, branch=b, scores=scores, prediction=predicted,
                                      gold=annotation[f"answer_{b}"], lm_tokens=tokens, prompt_sha256=prompt_hash))
                if arm == "curated_replay":
                    error = max(abs(x-y) for x, y in zip(scores, old["scores"], strict=True))
                    replay.append(dict(order=order, branch=b, max_abs=error, passed=error <= REPLAY_ATOL
                                       and predicted == old["prediction"]))
    return dict(pair_id=annotation["pair_id"], rows=rows, sampling=sampling, curated_replay_checks=replay,
                outcomes={name: backbone.history.outcome(rows[name], annotation) for name in ARMS})


def summarize(blocks):
    correct = {arm: [b["pair_id"] for b in blocks if b["outcomes"][arm]["robust"]] for arm in ARMS}
    gains = sorted(set(correct["dense_pool"]) - set(correct["capped_pool"]))
    losses = sorted(set(correct["capped_pool"]) - set(correct["dense_pool"]))
    failures = [dict(pair_id=b["pair_id"], **r) for b in blocks for r in b["curated_replay_checks"] if not r["passed"]]
    return dict(pairs=len(blocks), scores=sum(len(rows) for b in blocks for rows in b["rows"].values()),
                robust_pairs=correct, dense_only=gains, capped_only=losses, replay_failures=failures,
                pool_expansion_supported=len(blocks) == 4 and bool(gains) and not losses and not failures)


def execute(args):
    import torch
    from inference.llavaov_hermes import load_model

    work = args.work
    provenance = work / "provenance"
    spec = json.loads((provenance / "candidate_pool_cases.json").read_bytes())
    assert args.model.name == backbone.MODEL_REVISION == spec["model_revision"] and len(spec["cases"]) == 4
    paths = backbone.history.acquire(args, spec["media_source"], spec["media_plan"])
    model, processor = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=0.5)
    assert model.dtype == torch.float16 and {str(p.device) for p in model.parameters()} == {"cuda:0"}
    assert json.loads(json.dumps(processor.video_processor.to_dict())) == spec["video_processor"]
    config = dict(model_revision=args.model.name, torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0), replay_atol=REPLAY_ATOL,
                  source_archives=spec["source_archives"],
                  hashes={p.name: base.digest(p) for p in sorted(provenance.iterdir()) if p.is_file()})
    identity = base.sha(json.dumps(config, sort_keys=True).encode())
    base.atomic(work / "exports" / "run_config.json", config)
    checkpoints = work / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    blocks = []
    with torch.inference_mode():
        for i, case in enumerate(spec["cases"]):
            pair = case["annotation"]["pair_id"]
            path = checkpoints / f"{pair}.json"
            if path.is_file():
                saved = json.loads(path.read_bytes())
                assert saved["identity"] == identity and saved["block"]["pair_id"] == pair
                block = saved["block"]
                print(f"PAIR_REUSED {i+1}/4 {pair}", flush=True)
            else:
                torch.cuda.reset_peak_memory_stats()
                block = evaluate_pair(model, paths[pair], case, smoke=i == 0)
                block["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
                base.atomic(path, dict(identity=identity, block=block))
                if i == 0:
                    print("TWINPAST_CANDIDATE_POOL_PREFLIGHT_PASS", flush=True)
                print(f"PAIR_DONE {i+1}/4 {pair}", flush=True)
            blocks.append(block)
    report = summarize(blocks)
    assert report["pairs"] == 4 and report["scores"] == 48
    base.atomic(work / "exports" / "summary.json", report)
    base.atomic(work / "completion.json", dict(status="complete", pairs=4, scores=48))
    print("TWINPAST_CANDIDATE_POOL_COMPLETE", json.dumps(report), flush=True)


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
