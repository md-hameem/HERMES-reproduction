"""Check whether the 7B history-only results survive matched recent context."""
import argparse
import json
from pathlib import Path
import time

import backbone_runner as backbone

history = backbone.history
base = backbone.base
ARTIFACT = "HERMES_twinpast_fixed_recent_evidence.zip"
REPLAY_ATOL = 0.05


def evaluate_pair(model, paths, case, media, smoke=False):
    import numpy as np
    import torch

    annotation = case["annotation"]
    archived = {(r["order"], r["branch"]): r for r in case["backbone_rows"]}
    original = {(r["order"], r["arm"][-1]): r for r in case["original_prompt_rows"]}
    windows = backbone.reproduce_windows(paths, case["backbone_rows"])
    recent_frames = {}
    recent_hashes = media["decoded"]["a"]["decoded_rgb_sha256"][-4:]
    for branch in "ab":
        record = media["decoded"][branch]
        assert windows[branch]["seconds"][-1] < record["frame_start_s"][-4]
        assert record["decoded_rgb_sha256"][-4:] == recent_hashes
        recent_frames[branch] = base.decode(paths[branch], dict(
            source_frames=record["source_frames"][-4:], decoded_rgb_sha256=recent_hashes))
    assert np.array_equal(recent_frames["a"], recent_frames["b"])
    recent = base.features(model, recent_frames["a"])
    earlier = {b: base.features(model, windows[b]["frames"]) for b in "ab"}
    assert earlier["a"].shape == earlier["b"].shape
    combined = {b: torch.cat((earlier[b], recent), dim=1) for b in "ab"}
    system = model.language_model.get_input_embeddings()(
        torch.as_tensor(model.init_prompt_ids, device=model.device))
    groups = {"history_only": [], "history_with_recent": [], "recent_only": []}
    replay_checks, smoke_checks = [], []
    for order in base.ORDERS:
        ids, option_ids, permutation, prompt_hash = base.make_prompt(
            model, annotation["question"], annotation["options"], order, True)
        for condition, branch, visual in (
            ("history_only", "a", earlier["a"]),
            ("history_only", "b", earlier["b"]),
            ("history_with_recent", "a", combined["a"]),
            ("history_with_recent", "b", combined["b"]),
            ("recent_only", None, recent),
        ):
            start = time.perf_counter()
            scores, tokens = base.score(model, system, ids, option_ids, visual)
            elapsed = time.perf_counter() - start
            if smoke and condition == "history_with_recent":
                repeated, repeat_tokens = base.score(model, system, ids, option_ids, visual)
                torch.testing.assert_close(scores, repeated, rtol=0, atol=0)
                assert tokens == repeat_tokens
                smoke_checks.append(dict(order=order, branch=branch, lm_tokens=tokens))
            values = [0.0] * len(permutation)
            for i, value in zip(permutation, scores.tolist(), strict=True):
                values[i] = value
            prediction = base.LETTERS[max(range(len(values)), key=values.__getitem__)]
            row = dict(order=order, branch=branch, prediction=prediction, scores=values,
                       option_token_ids=option_ids, permutation=permutation, prompt_sha256=prompt_hash,
                       visual_tokens=visual.shape[1], lm_tokens=tokens, forward_s=elapsed,
                       history_frame_hashes=windows[branch]["hashes"] if branch else [],
                       recent_frame_hashes=recent_hashes if condition != "history_only" else [])
            if branch:
                old = archived[order, branch]
                assert old["prompt_sha256"] == prompt_hash
                assert original[order, branch]["label_token_ids"] == option_ids
                row["gold"] = annotation[f"answer_{branch}"]
                if condition == "history_only":
                    assert tokens == old["lm_tokens"]
                    delta = max(abs(a - b) for a, b in zip(values, old["scores"], strict=True))
                    replay_checks.append(dict(order=order, branch=branch, archived_prediction=old["prediction"],
                                              max_abs_score_difference=delta,
                                              pass_replay=prediction == old["prediction"] and delta <= REPLAY_ATOL))
            groups[condition].append(row)
    for condition in ("history_only", "history_with_recent"):
        for a, b in zip(groups[condition][::2], groups[condition][1::2], strict=True):
            assert a["order"] == b["order"] and a["lm_tokens"] == b["lm_tokens"]
    return dict(pair_id=annotation["pair_id"], cohort=case["cohort"], rows=groups,
                history_only=history.outcome(groups["history_only"], annotation),
                history_with_recent=history.outcome(groups["history_with_recent"], annotation),
                replay_checks=replay_checks, smoke_checks=smoke_checks)


def summarize(blocks, recovered):
    robust_states = {r["pair_id"] for r in blocks
                     if r["cohort"] == "historical_state" and r["history_with_recent"]["robust"]}
    controls = sum(r["history_with_recent"]["robust"] for r in blocks if r["cohort"] == "positive_control")
    failures = [dict(pair_id=r["pair_id"], **check) for r in blocks for check in r["replay_checks"]
                if not check["pass_replay"]]
    return dict(pairs=len(blocks), scores=sum(len(rows) for r in blocks for rows in r["rows"].values()),
                replay_failures=failures, replay_atol=REPLAY_ATOL,
                previously_recovered_states=sorted(recovered),
                retained_recovered_states=sorted(robust_states & recovered),
                newly_recovered_states=sorted(robust_states - recovered),
                positive_controls_retained=controls,
                development_criterion_met=len(blocks) == 7 and not failures and recovered <= robust_states and controls >= 2,
                per_pair=[{k: v for k, v in r.items() if k not in {"rows", "smoke_checks"}} for r in blocks])


def execute(args):
    import torch
    from inference.llavaov_hermes import load_model

    work = args.work
    provenance = work / "provenance"
    bundle = json.loads((provenance / "bundle.json").read_bytes())
    frozen = json.loads((provenance / "fixed_recent_cases.json").read_bytes())
    cases = {c["annotation"]["pair_id"]: c for c in frozen["cases"]}
    recovered = set(frozen["recovered_states"])
    assert len(cases) == 7 and len(recovered) == 2
    assert args.model.name == backbone.MODEL_REVISION == frozen["prior_config"]["model_revision"]
    selection = [item for item in base.cases(bundle) if item[0]["pair_id"] in cases]
    assert {p["pair_id"] for p, _, _ in selection} == cases.keys()
    paths = history.acquire(args, bundle, selection)
    print("LOADING_PINNED_7B", args.model.name, flush=True)
    model, processor = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=0.5)
    assert model.dtype == torch.float16
    assert {str(p.device) for p in model.parameters()} == {"cuda:0"}, "7B must fit fully on GPU 0"
    processor_config = json.loads(json.dumps(processor.video_processor.to_dict()))
    assert processor_config == frozen["prior_config"]["video_processor"], "Video processor differs from the prior run"
    config = dict(model_revision=args.model.name, torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0), dtype=str(model.dtype), replay_atol=REPLAY_ATOL,
                  source_archive_sha256=frozen["backbone_archive_sha256"],
                  provenance_hashes={p.name: base.digest(p) for p in sorted(provenance.iterdir()) if p.is_file()})
    identity = base.sha(json.dumps(config, sort_keys=True).encode())
    base.atomic(work / "exports" / "run_config.json", config)
    checkpoints = work / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    blocks = []
    with torch.inference_mode():
        for i, (plan, media, _) in enumerate(selection):
            pair = plan["pair_id"]
            checkpoint = checkpoints / f"{pair}.json"
            if checkpoint.is_file():
                saved = json.loads(checkpoint.read_bytes())
                assert saved["identity"] == identity and saved["block"]["pair_id"] == pair
                blocks.append(saved["block"])
                print(f"PAIR_REUSED {i + 1}/7 {pair}", flush=True)
                continue
            torch.cuda.reset_peak_memory_stats()
            block = evaluate_pair(model, paths[pair], cases[pair], media, smoke=i == 0)
            block["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
            if i == 0:
                base.atomic(work / "exports" / "preflight.json",
                            dict(status="pass", pair_id=pair, checks=block["smoke_checks"],
                                 peak_gpu_allocated_bytes=block["peak_gpu_allocated_bytes"]))
                print("TWINPAST_FIXED_RECENT_PREFLIGHT_PASS", flush=True)
            base.atomic(checkpoint, dict(identity=identity, block=block))
            blocks.append(block)
            print(f"PAIR_DONE {i + 1}/7 {pair} robust={block['history_with_recent']['robust']}", flush=True)
    report = summarize(blocks, recovered)
    assert report["pairs"] == 7 and report["scores"] == 70
    base.atomic(work / "exports" / "summary.json", report)
    base.atomic(work / "completion.json", dict(status="complete", pairs=7, scores=70))
    print("TWINPAST_FIXED_RECENT_COMPLETE", json.dumps({k: v for k, v in report.items() if k != "per_pair"}), flush=True)


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
