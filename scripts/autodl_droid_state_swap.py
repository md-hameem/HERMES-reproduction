"""Test earlier-state content at fixed frame count and terminal positions."""
import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np

from natural_history_runner import auto, base, choose, load_frames

ARTIFACT = "HERMES_droid_state_swap_evidence.zip"
METHODS = ("original", "swapped")
REPLAY_ATOL = 0.05


def compare(model, system, case, pixels, shared):
    import torch

    banks = {}
    for branch in case["branches"].values():
        views = branch["frames"]
        names = sorted(set(views["uniform"]) | set(views["annotated"]))
        encoded = auto.encode_frames(model, np.stack([pixels[n] for n in names]))
        banks.update(zip(names, encoded, strict=True))
    result = {}
    for side, branch in case["branches"].items():
        swap = case["swap"]["branches"][side]
        original, swapped = swap["original_history"], swap["swapped_history"]
        other = case["branches"][swap["donor_branch"]]["frames"]["annotated"]
        assert original == branch["frames"]["annotated"] and len(original) == len(swapped) == 6
        assert swapped == other[:2]+original[2:] and len(set(swapped)) == 6 and len(shared) == 4
        visuals = {method: torch.cat([banks[n] for n in frames]+shared, dim=1)
                   for method, frames in zip(METHODS, (original, swapped), strict=True)}
        assert visuals["original"].shape == visuals["swapped"].shape
        answers, failures = {}, []
        for order in base.ORDERS:
            answers[order] = {m: choose(model, system, v, case["question"], case["options"], order)
                              for m, v in visuals.items()}
            replay = answers[order]["original"]
            prior = branch["archived_full"][order]
            delta = max(abs(a-b) for a, b in zip(replay["logits"], prior["logits"], strict=True))
            replay["replay_max_abs"] = delta
            if delta > REPLAY_ATOL or any(replay[k] != prior[k] for k in
                                        ("prediction", "prompt_sha256", "visual_tokens", "lm_tokens")):
                failures.append(dict(order=order, archived=prior, replay=replay))
        result[side] = dict(answers=answers, replay_failures=failures)
    return dict(pair_id=case["pair_id"], scene_id=case["scene_id"], source_site=case["source_site"], branches=result)


def summarize(rows, targets):
    methods = {m: dict(robust_pairs=[], correct_branch_orders=0) for m in METHODS}
    margin_deltas = []
    scored = 0
    for row in rows:
        pair = row["pair_id"]
        correct = dict.fromkeys(METHODS, 0)
        for side, branch in row["branches"].items():
            gold = targets[pair][side]
            index = base.LETTERS.index(gold)
            assert index in (0, 1)
            for answers in branch["answers"].values():
                margins = []
                for method in METHODS:
                    answer = answers[method]
                    correct[method] += answer["prediction"] == gold
                    scores = answer["logits"]
                    assert len(scores) == 2
                    margins.append(scores[index]-scores[1-index])
                    scored += 1
                margin_deltas.append(margins[1]-margins[0])
        for method, n in correct.items():
            methods[method]["correct_branch_orders"] += n
            if n == 4:
                methods[method]["robust_pairs"].append(pair)
    before, after = (set(methods[m]["robust_pairs"]) for m in METHODS)
    return dict(pairs=len(rows), scores=scored, methods=methods, recoveries=sorted(after-before),
                regressions=sorted(before-after), median_gold_margin_delta=median(margin_deltas),
                positive_gold_margin_deltas=sum(delta > 0 for delta in margin_deltas))


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
            provenance = work / "provenance"
            spec = json.loads((provenance / "droid_swap_cases.json").read_bytes())
            assert len(spec["cases"]) == 7 and args.model.name == spec["model_revision"]
            assert base.digest(args.frames) == spec["frame_zip_sha256"]
            pixels = load_frames(args.frames, spec["frame_receipts"])
            print("DROID_SWAP_INPUTS_VERIFIED", len(pixels), flush=True)
            model, processor = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=0.5)
            assert model.dtype == torch.float16 and {str(p.device) for p in model.parameters()} == {"cuda:0"}
            assert json.loads(json.dumps(processor.video_processor.to_dict())) == spec["video_processor"]
            base.atomic(export / "run_config.json", dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                        cuda=torch.version.cuda, model_revision=args.model.name, replay_atol=REPLAY_ATOL,
                        prior_archive_sha256=spec["prior_archive_sha256"], prior_gpu=spec["prior_gpu"],
                        swap_plan_sha256=spec["swap_plan_sha256"], frame_zip_sha256=spec["frame_zip_sha256"]))
            rows = []
            with torch.inference_mode():
                system = model.language_model.get_input_embeddings()(
                    torch.as_tensor(model.init_prompt_ids, device=model.device)
                )
                recent = spec["cases"][0]["recent_frames"]
                assert len(recent) == len(set(recent)) == 4
                shared = auto.encode_frames(model, np.stack([pixels[n] for n in recent]))
                for i, case in enumerate(spec["cases"], 1):
                    assert case["recent_frames"] == recent
                    rows.append(compare(model, system, case, pixels, shared))
                    base.atomic(export / "pair_scores.json", rows)
                    print(f"DROID_SWAP_PAIR {i}/7 {case['pair_id']}", flush=True)
            failures = [dict(pair_id=row["pair_id"], branch=side, **f) for row in rows
                        for side, branch in row["branches"].items() for f in branch["replay_failures"]]
            deltas = [a["original"]["replay_max_abs"] for row in rows for b in row["branches"].values()
                      for a in b["answers"].values()]
            assert len(deltas) == 28
            base.atomic(export / "replay_gate.json", dict(scores=len(deltas), max_abs=max(deltas), failures=failures))
            if failures:
                raise RuntimeError("Original history replay failed. See exports/replay_gate.json.")
            targets = json.loads((provenance / "droid_natural_targets.json").read_bytes())
            report = summarize(rows, targets)
            assert report["scores"] == 56
            base.atomic(export / "summary.json", report)
            base.atomic(work / "completion.json", dict(status="complete", pairs=len(rows)))
            print("DROID_STATE_SWAP_COMPLETE", json.dumps(report), flush=True)
        except BaseException as exc:
            base.atomic(work / "completion.json", dict(status="failed", error_type=type(exc).__name__, message=str(exc)))
            raise
        finally:
            base.package(work)


if __name__ == "__main__":
    main()
