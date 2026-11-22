"""Replay the frozen history-only windows on the larger same-family backbone."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import history_only_runner as history

base = history.base
ARTIFACT = "HERMES_twinpast_backbone_evidence.zip"
MODEL_REVISION = "0d50680527681998e456c7b78950205bedd8a068"


def reproduce_windows(paths, records):
    windows = {}
    for branch in "ab":
        left, right = (next(r for r in records if r["branch"] == branch and r["order"] == order)
                       for order in base.ORDERS)
        assert all(left[k] == right[k] for k in ("frame_indices", "frame_seconds", "frame_hashes"))
        frames = base.decode(paths[branch], dict(source_frames=left["frame_indices"],
                                                 decoded_rgb_sha256=left["frame_hashes"]))
        windows[branch] = dict(frames=frames, indices=left["frame_indices"],
                               seconds=left["frame_seconds"], hashes=left["frame_hashes"])
    return windows


def run_pair(model, paths, case, smoke=False):
    windows = reproduce_windows(paths, case["reference_rows"])
    original = {(r["order"], r["arm"][-1]): r for r in case["original_prompt_rows"]}
    reference = {}
    for row in case["reference_rows"]:
        key = row["order"], row["branch"]
        assert row["prompt_sha256"] == original[key]["prompt_sha256"]
        reference[key] = {**row, "label_token_ids": original[key]["label_token_ids"]}
    rows, checks = history.evaluate(model, case["annotation"], windows, reference, preflight=smoke)
    prior = history.outcome(case["reference_rows"], case["annotation"])
    current = history.outcome(rows, case["annotation"])
    return dict(pair_id=case["annotation"]["pair_id"], cohort=case["cohort"], prior=prior,
                current=current, preflight=checks, rows=rows)


def summarize(blocks):
    states = [x for x in blocks if x["cohort"] == "historical_state"]
    controls = [x for x in blocks if x["cohort"] == "positive_control"]
    robust_states = sum(x["current"]["robust"] for x in states)
    robust_controls = sum(x["current"]["robust"] for x in controls)
    return dict(pairs=len(blocks), scores=sum(len(x["rows"]) for x in blocks),
                historical_state_robust=robust_states, historical_state_total=len(states),
                positive_control_robust=robust_controls, positive_control_total=len(controls),
                development_criterion_met=len(states) == 4 and len(controls) == 3
                and robust_states >= 2 and robust_controls >= 2,
                per_pair=[{k: v for k, v in x.items() if k not in {"preflight", "rows"}} for x in blocks])


def execute(args):
    import torch
    from inference.llavaov_hermes import load_model

    work = args.work
    provenance = work / "provenance"
    bundle = json.loads((provenance / "bundle.json").read_bytes())
    frozen = json.loads((provenance / "backbone_cases.json").read_bytes())
    cases = {c["annotation"]["pair_id"]: c for c in frozen["cases"]}
    assert len(cases) == 7 and frozen["model_revision"] == MODEL_REVISION == args.model.name
    selection = [pair for pair in base.cases(bundle) if pair[0]["pair_id"] in cases]
    assert {p["pair_id"] for p, _, _ in selection} == cases.keys()
    # Start with the six-frame historical-state case, which exercises the longest input.
    assert cases[selection[0][0]["pair_id"]]["cohort"] == "historical_state"
    paths = history.acquire(SimpleNamespace(work=work, previous=args.previous, media_reuse=args.media_reuse),
                            bundle, selection)
    print("LOADING_PINNED_7B", args.model.name, flush=True)
    model, processor = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=0.5)
    assert model.device.type == "cuda" and model.dtype == torch.float16
    assert {str(p.device) for p in model.parameters()} == {"cuda:0"}, "7B must fit fully on GPU 0"
    config = dict(model_revision=MODEL_REVISION, gpu=torch.cuda.get_device_name(0),
                  torch=torch.__version__, cuda=torch.version.cuda, dtype=str(model.dtype),
                  video_processor=processor.video_processor.to_dict(),
                  runtime_source_hashes={p.name: base.digest(p) for p in sorted(provenance.iterdir()) if p.is_file()},
                  source_history_only_archive=frozen["history_only_archive_sha256"])
    ident = base.sha(json.dumps(config, sort_keys=True).encode())
    base.atomic(work / "exports" / "run_config.json", config)
    checkpoints = work / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    blocks = []
    with torch.inference_mode():
        for i, (plan, _, _) in enumerate(selection):
            pair = plan["pair_id"]
            checkpoint = checkpoints / f"{pair}.json"
            saved = json.loads(checkpoint.read_bytes()) if checkpoint.is_file() else None
            if saved is not None:
                assert saved["identity"] == ident and saved["block"]["pair_id"] == pair
            if i and saved is not None:
                blocks.append(saved["block"])
                print(f"PAIR_REUSED {i + 1}/7 {pair}", flush=True)
                continue
            print(f"PAIR_RUNNING {i + 1}/7 {pair}", flush=True)
            torch.cuda.reset_peak_memory_stats()
            block = run_pair(model, paths[pair], cases[pair], smoke=i == 0)
            block["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
            if i == 0:
                base.atomic(work / "exports" / "preflight.json",
                            dict(status="pass", pair_id=pair, checks=block["preflight"],
                                 peak_gpu_allocated_bytes=block["peak_gpu_allocated_bytes"]))
                print("TWINPAST_BACKBONE_PREFLIGHT_PASS", flush=True)
            base.atomic(checkpoint, dict(identity=ident, block=block))
            blocks.append(block)
            print(f"PAIR_DONE {i + 1}/7 {pair} robust={block['current']['robust']}", flush=True)
    report = summarize(blocks)
    assert report["pairs"] == 7 and report["scores"] == 28
    base.atomic(work / "exports" / "summary.json", report)
    base.atomic(work / "completion.json", dict(status="complete", pairs=7, scores=28))
    print("TWINPAST_BACKBONE_COMPLETE", json.dumps({k: v for k, v in report.items() if k != "per_pair"}), flush=True)


def main():
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


if __name__ == "__main__":
    main()
