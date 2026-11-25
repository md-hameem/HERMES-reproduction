"""Compare native-timestamp annotated evidence with the archived uniform input."""
import argparse
import json
from pathlib import Path

import temporal_runner as temporal

base = temporal.base
backbone = temporal.backbone
ARTIFACT = "HERMES_twinpast_subsecond_evidence.zip"
REPLAY_ATOL = 0.05


def annotated_indices(seconds, intervals, uniform, cutoff):
    eligible = [i for i, t in enumerate(seconds) if t < cutoff and any(
        lo - 1e-6 <= t <= hi + 1e-6 for lo, hi in intervals)]
    if len(eligible) >= 6:
        return [eligible[i * (len(eligible) - 1) // 5] for i in range(6)], eligible
    supplements = [i for i in uniform if i not in eligible]
    return sorted(eligible + supplements[:6-len(eligible)]), eligible


def native_input(path, sample, uniform, intervals):
    import numpy as np
    from decord import VideoReader

    reader = VideoReader(str(path), num_threads=1)
    starts = np.asarray(reader.get_frame_timestamp(list(range(len(reader)))), dtype=np.float64)[:, 0]
    starts -= starts[0]
    cutoff = sample["timestamps_s"][-4]
    baseline_indices = [sample["indices"][i] for i in uniform[:-4]]
    assert len(baseline_indices) == 6 and all(starts[i] < cutoff for i in baseline_indices)
    indices, eligible = annotated_indices(starts, intervals, baseline_indices, cutoff)
    assert len(indices) == len(set(indices)) == 6 and all(starts[i] < cutoff for i in indices)
    frames = reader.get_batch(indices).asnumpy()
    return dict(frames=frames, indices=indices, seconds=starts[indices].tolist(),
                hashes=[base.sha(frame.tobytes()) for frame in frames], eligible_native_frames=len(eligible),
                matching_selected=sum(i in eligible for i in indices))


def evaluate_pair(model, paths, case, work):
    import torch

    pair = case["annotation"]["pair_id"]
    annotation = case["annotation"]
    system = model.language_model.get_input_embeddings()(torch.as_tensor(model.init_prompt_ids, device=model.device))
    rows = {"uniform": [], "annotated_native": []}
    checks, inputs, reviews, sampling = [], {}, {"uniform": {}, "annotated_native": {}}, {}
    for b in "ab":
        old = case["baseline"][b]
        sample = old["sample"]
        frames = base.decode(paths[b], dict(source_frames=sample["indices"], decoded_rgb_sha256=sample["hashes"]))
        bank = temporal.auto.encode_frames(model, frames)
        uniform = old["selection"]
        assert len(uniform) == 10 and uniform[-4:] == list(range(len(bank)-4, len(bank)))
        oracle = native_input(paths[b], sample, uniform, annotation["evidence_intervals"][b.upper()])
        inputs[b] = dict(uniform=torch.cat([bank[i] for i in uniform], dim=1), annotated_native=torch.cat(
            temporal.auto.encode_frames(model, oracle["frames"]) + bank[-4:], dim=1))
        assert inputs[b]["uniform"].shape == inputs[b]["annotated_native"].shape
        recent_hashes = sample["hashes"][-4:]
        assert recent_hashes == case["baseline"]["a"]["sample"]["hashes"][-4:]
        reviews["uniform"][b] = dict(frames=frames[uniform[:-4]], seconds=[sample["timestamps_s"][i] for i in uniform[:-4]])
        reviews["annotated_native"][b] = oracle
        sampling[b] = {k: v for k, v in oracle.items() if k != "frames"}
        sampling[b]["shared_recent_hashes"] = recent_hashes
    for order in base.ORDERS:
        ids, option_ids, perm, prompt_hash = base.make_prompt(model, annotation["question"],
                                                            annotation["options"], order, True)
        for b in "ab":
            for arm, visual in inputs[b].items():
                output, tokens = base.score(model, system, ids, option_ids, visual)
                values = [0.0] * len(perm)
                for i, value in zip(perm, output.tolist(), strict=True):
                    values[i] = value
                prediction = base.LETTERS[temporal.auto.best(values)]
                rows[arm].append(dict(order=order, branch=b, prediction=prediction, gold=annotation[f"answer_{b}"],
                                      scores=values, lm_tokens=tokens, prompt_sha256=prompt_hash))
                if arm == "uniform":
                    old = case["baseline"][b]["scores"][order]
                    error = max(abs(x-y) for x,y in zip(values, old, strict=True))
                    checks.append(dict(order=order, branch=b, max_abs=error,
                                       passed=error <= REPLAY_ATOL and prediction == base.LETTERS[temporal.auto.best(old)]))
    for arm, windows in reviews.items():
        backbone.history.contact_sheet(work / "exports" / "review" / f"{pair}_{arm}.jpg", pair, arm, windows)
    return dict(pair_id=pair, rows=rows, sampling=sampling, replay_checks=checks,
                outcomes={arm: backbone.history.outcome(v, annotation) for arm, v in rows.items()})


def summarize(blocks):
    results = {arm: [b["pair_id"] for b in blocks if b["outcomes"][arm]["robust"]]
               for arm in ("uniform", "annotated_native")}
    failures = [dict(pair_id=b["pair_id"], **check) for b in blocks for check in b["replay_checks"] if not check["passed"]]
    recoveries = sorted(set(results["annotated_native"]) - set(results["uniform"]))
    return dict(pairs=len(blocks), scores=sum(len(rows) for b in blocks for rows in b["rows"].values()),
                robust_pairs=results, recovered_pairs=recoveries, replay_failures=failures,
                development_criterion_met=len(blocks)==7 and len(recoveries)>=2 and not failures)


def execute(args):
    import torch
    from inference.llavaov_hermes import load_model

    work = args.work
    provenance = work / "provenance"
    spec = json.loads((provenance / "subsecond_cases.json").read_bytes())
    assert args.model.name == backbone.MODEL_REVISION == spec["model_revision"] and len(spec["cases"]) == 7
    paths = backbone.history.acquire(args, spec["media_source"], spec["media_plan"])
    model, processor = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=0.5)
    assert model.dtype == torch.float16 and {str(p.device) for p in model.parameters()} == {"cuda:0"}
    assert json.loads(json.dumps(processor.video_processor.to_dict())) == spec["video_processor"]
    config = dict(model_revision=args.model.name, cuda=torch.version.cuda, torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(0), replay_atol=REPLAY_ATOL,
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
            else:
                block = evaluate_pair(model, paths[pair], case, work)
                base.atomic(path, dict(identity=identity, block=block))
            blocks.append(block)
            print(f"PAIR_DONE {i+1}/7 {pair}", flush=True)
    report = summarize(blocks)
    assert report["pairs"] == 7 and report["scores"] == 56
    base.atomic(work / "exports" / "summary.json", report)
    base.atomic(work / "completion.json", dict(status="complete", pairs=7, scores=56))
    print("TWINPAST_SUBSECOND_COMPLETE", json.dumps(report), flush=True)


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
