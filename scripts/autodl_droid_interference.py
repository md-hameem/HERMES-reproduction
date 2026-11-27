"""Test earlier competing-state masking with fixed terminal and recent frames."""
import argparse
import json
from pathlib import Path

import numpy as np

from droid_state_masks import MASK_RGB, apply_mask
from natural_history_runner import auto, base, choose, load_frames

ARTIFACT = "HERMES_droid_state_interference_evidence.zip"
METHODS = ("original", "target_mask", "non_target_mask")
REPLAY_ATOL = 0.05


def compare(model, system, case, pixels, recent_features):
    import torch

    result = {}
    for side, branch in case["branches"].items():
        names = sorted(set(branch["frames"]["uniform"]) | set(branch["frames"]["annotated"]))
        # Match the original natural-history vision batch order for replay.
        bank = dict(zip(names, auto.encode_frames(model, np.stack([pixels[n] for n in names])), strict=True))
        history = branch["frames"]["annotated"]
        slots = case["mask"]["edited_slots"]
        edited = [history[k] for k in slots]
        assert slots == [0, 1] and len(history) == 6 and set(edited).isdisjoint(case["recent_frames"])
        visual = {"original": torch.cat([bank[n] for n in history]+recent_features, dim=1)}
        claims = case["mask"]["edited_rgb_sha256"][side]
        assert set(edited) == set(claims)
        receipts = {}
        for method, kind in (("target_mask", "target"), ("non_target_mask", "control")):
            frames = [apply_mask(pixels[n], case["mask"][kind]) for n in edited]
            hashes = [base.sha(f.tobytes()) for f in frames]
            assert hashes == [claims[n][kind] for n in edited]
            receipts[method] = dict(zip(edited, hashes, strict=True))
            altered = dict(zip(edited, auto.encode_frames(model, np.stack(frames)), strict=True))
            visual[method] = torch.cat([altered.get(n, bank[n]) for n in history]+recent_features, dim=1)
        assert len({tuple(v.shape) for v in visual.values()}) == 1
        rows, failures = {}, []
        for order in base.ORDERS:
            rows[order] = {m: choose(model, system, value, case["question"], case["options"], order)
                           for m, value in visual.items()}
            original, prior = rows[order]["original"], branch["archived_annotated"][order]
            error = max(abs(x-y) for x, y in zip(original["logits"], prior["logits"], strict=True))
            original["replay_max_abs"] = error
            if error > REPLAY_ATOL or any(original[k] != prior[k] for k in
                                         ("prediction", "prompt_sha256", "visual_tokens", "lm_tokens")):
                failures.append(dict(order=order, replay=original, archived=prior))
        result[side] = dict(answers=rows, mask_receipts=receipts, replay_failures=failures)
    return dict(pair_id=case["pair_id"], source_site=case["source_site"], scene_id=case["scene_id"], branches=result)


def summarize(blocks, targets):
    methods = {m: dict(robust_pairs=[], correct_branch_orders=0) for m in METHODS}
    count = 0
    for block in blocks:
        pair = block["pair_id"]
        for method in METHODS:
            answers = [r["answers"][order][method]["prediction"] == targets[pair][side]
                       for side, r in block["branches"].items() for order in base.ORDERS]
            methods[method]["correct_branch_orders"] += sum(answers)
            count += len(answers)
            if all(answers):
                methods[method]["robust_pairs"].append(pair)
    target = set(methods["target_mask"]["robust_pairs"])
    original = set(methods["original"]["robust_pairs"])
    control = set(methods["non_target_mask"]["robust_pairs"])
    recovered = sorted(target-original-control)
    return dict(pairs=len(blocks), scores=count, methods=methods,
                target_only_recoveries=recovered, lost_vs_original=sorted(original-target),
                lost_vs_non_target=sorted(control-target),
                exploratory_criterion_met=len(blocks) == 7 and len(recovered) >= 2 and not (original|control)-target)


def main():
    import torch
    from filelock import FileLock
    from inference.llavaov_hermes import load_model

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "model", "frames"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    work = args.work
    with FileLock(str(work / "run.lock"), timeout=0):
        export = work / "exports"
        export.mkdir(exist_ok=True)
        for name in ("summary.json", "pair_scores.json", "replay_gate.json", "run_config.json"):
            (export / name).unlink(missing_ok=True)
        base.ARTIFACT = ARTIFACT
        base.atomic(work / "completion.json", dict(status="running"))
        try:
            spec = json.loads((work / "provenance/droid_interference_cases.json").read_bytes())
            assert len(spec["cases"]) == 7 and args.model.name == spec["model_revision"]
            assert tuple(spec["mask_fill_rgb"]) == MASK_RGB and base.digest(args.frames) == spec["frame_zip_sha256"]
            pixels = load_frames(args.frames, spec["frame_receipts"])
            print("DROID_INTERFERENCE_INPUTS_VERIFIED", len(pixels), flush=True)
            model, processor = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=0.5)
            assert model.dtype == torch.float16 and {str(p.device) for p in model.parameters()} == {"cuda:0"}
            assert json.loads(json.dumps(processor.video_processor.to_dict())) == spec["video_processor"]
            base.atomic(export / "run_config.json", dict(torch=torch.__version__, cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(), model_revision=args.model.name, replay_atol=REPLAY_ATOL,
                         mask_plan_sha256=spec["mask_plan_sha256"],
                         natural_history_archive_sha256=spec["natural_history_archive_sha256"],
                         frame_zip_sha256=spec["frame_zip_sha256"]))
            blocks = []
            with torch.inference_mode():
                system = model.language_model.get_input_embeddings()(
                    torch.as_tensor(model.init_prompt_ids, device=model.device)
                )
                recent_names = spec["cases"][0]["recent_frames"]
                assert len(recent_names) == len(set(recent_names)) == 4
                recent = auto.encode_frames(model, np.stack([pixels[n] for n in recent_names]))
                for i, case in enumerate(spec["cases"], 1):
                    assert case["recent_frames"] == recent_names
                    blocks.append(compare(model, system, case, pixels, recent))
                    base.atomic(export / "pair_scores.json", blocks)
                    print(f"DROID_INTERFERENCE_PAIR {i}/7 {case['pair_id']}", flush=True)
            rows = [trace for block in blocks for trace in block["branches"].values()]
            deltas = [a["original"]["replay_max_abs"] for row in rows for a in row["answers"].values()]
            failures = [dict(pair_id=block["pair_id"], branch=side, **f) for block in blocks
                        for side, trace in block["branches"].items() for f in trace["replay_failures"]]
            assert len(deltas) == 28
            base.atomic(export / "replay_gate.json", dict(scores=len(deltas), max_abs=max(deltas), failures=failures))
            if failures:
                raise RuntimeError("Archived annotated replay failed. See exports/replay_gate.json.")
            targets = json.loads((work / "provenance/droid_natural_targets.json").read_bytes())
            report = summarize(blocks, targets)
            assert report["scores"] == 84
            base.atomic(export / "summary.json", report)
            base.atomic(work / "completion.json", dict(status="complete", pairs=len(blocks)))
            print("DROID_STATE_INTERFERENCE_COMPLETE", json.dumps(report), flush=True)
        except BaseException as exc:
            base.atomic(work / "completion.json", dict(status="failed", error_type=type(exc).__name__, message=str(exc)))
            raise
        finally:
            base.package(work)


if __name__ == "__main__":
    main()
