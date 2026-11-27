"""Compare exact archived terminal frames under three context lengths."""
import argparse
import json
from pathlib import Path

import numpy as np

from natural_history_runner import auto, base, choose, load_frames

ARTIFACT = "HERMES_droid_terminal_context_evidence.zip"
FRAME_COUNTS = {"terminal_only": 3, "terminal_recent": 7, "full_annotated": 10}
REPLAY_ATOL = 0.05


def compare(model, system, case, pixels, recent_features):
    import torch

    branches = {}
    for side, branch in case["branches"].items():
        histories = branch["frames"]
        names = sorted(set(histories["uniform"]) | set(histories["annotated"]))
        # Match the archived encoding order and reuse identical embeddings across arms.
        bank = dict(zip(names, auto.encode_frames(model, np.stack([pixels[n] for n in names])), strict=True))
        history = [bank[n] for n in histories["annotated"]]
        terminal = history[-3:]
        assert len(history) == 6 and len(recent_features) == 4
        inputs = dict(terminal_only=terminal, terminal_recent=terminal+recent_features,
                      full_annotated=history+recent_features)
        token_count = terminal[0].shape[1]
        views = {method: torch.cat(features, dim=1) for method, features in inputs.items()}
        assert all(v.shape[1] == FRAME_COUNTS[m]*token_count for m, v in views.items())
        rows, failures = {}, []
        for order in base.ORDERS:
            rows[order] = {m: choose(model, system, v, case["question"], case["options"], order)
                           for m, v in views.items()}
            replay, prior = rows[order]["full_annotated"], branch["archived_full"][order]
            delta = max(abs(x-y) for x, y in zip(replay["logits"], prior["logits"], strict=True))
            replay["replay_max_abs"] = delta
            if delta > REPLAY_ATOL or any(replay[k] != prior[k] for k in
                                        ("prediction", "lm_tokens", "visual_tokens", "prompt_sha256")):
                failures.append(dict(order=order, replay=replay, archived=prior))
        branches[side] = dict(answers=rows, terminal_frames=histories["annotated"][-3:], replay_failures=failures)
    return dict(pair_id=case["pair_id"], scene_id=case["scene_id"], source_site=case["source_site"], branches=branches)


def summarize(blocks, targets):
    methods = {m: dict(frames=frames, robust_pairs=[], correct_branch_orders=0, option_order_flips=0)
               for m, frames in FRAME_COUNTS.items()}
    scores = 0
    for block in blocks:
        pair = block["pair_id"]
        for method, result in methods.items():
            correct = 0
            for side, branch in block["branches"].items():
                labels = [branch["answers"][order][method]["prediction"] for order in base.ORDERS]
                result["option_order_flips"] += len(set(labels)) != 1
                correct += sum(label == targets[pair][side] for label in labels)
                scores += len(labels)
            result["correct_branch_orders"] += correct
            if correct == 4:
                result["robust_pairs"].append(pair)
    added = {}
    for before, after, stage in (("terminal_only", "terminal_recent", "added_recent"),
                                 ("terminal_recent", "full_annotated", "added_earlier")):
        left, right = (set(methods[m]["robust_pairs"]) for m in (before, after))
        added[stage] = dict(gains=sorted(right-left), losses=sorted(left-right))
    return dict(pairs=len(blocks), scores=scores, methods=methods, context_transitions=added)


def main():
    import torch
    from filelock import FileLock
    from inference.llavaov_hermes import load_model

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "model", "frames"):
        parser.add_argument(f"--{name}", required=True, type=Path)
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
            spec = json.loads((work / "provenance/droid_terminal_cases.json").read_bytes())
            assert len(spec["cases"]) == 7 and args.model.name == spec["model_revision"]
            assert base.digest(args.frames) == spec["frame_zip_sha256"]
            pixels = load_frames(args.frames, spec["frame_receipts"])
            print("DROID_TERMINAL_INPUTS_VERIFIED", len(pixels), flush=True)
            model, processor = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=0.5)
            assert model.dtype == torch.float16 and {str(p.device) for p in model.parameters()} == {"cuda:0"}
            assert json.loads(json.dumps(processor.video_processor.to_dict())) == spec["video_processor"]
            base.atomic(export / "run_config.json", dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                        cuda=torch.version.cuda, model_revision=args.model.name, replay_atol=REPLAY_ATOL,
                        archived_full_gpu=spec["archived_full_gpu"], archived_full_zip_sha256=spec["archived_full_zip_sha256"],
                        frame_zip_sha256=spec["frame_zip_sha256"], frame_counts=FRAME_COUNTS))
            blocks = []
            with torch.inference_mode():
                system = model.language_model.get_input_embeddings()(
                    torch.as_tensor(model.init_prompt_ids, device=model.device)
                )
                names = spec["cases"][0]["recent_frames"]
                assert len(names) == len(set(names)) == 4
                shared = auto.encode_frames(model, np.stack([pixels[n] for n in names]))
                for i, case in enumerate(spec["cases"], 1):
                    assert case["recent_frames"] == names
                    blocks.append(compare(model, system, case, pixels, shared))
                    base.atomic(export / "pair_scores.json", blocks)
                    print(f"DROID_TERMINAL_PAIR {i}/7 {case['pair_id']}", flush=True)
            failures = [dict(pair_id=block["pair_id"], branch=side, **failure) for block in blocks
                        for side, branch in block["branches"].items() for failure in branch["replay_failures"]]
            deltas = [row["full_annotated"]["replay_max_abs"] for block in blocks
                      for branch in block["branches"].values() for row in branch["answers"].values()]
            assert len(deltas) == 28
            base.atomic(export / "replay_gate.json", dict(scores=len(deltas), max_abs=max(deltas), failures=failures))
            if failures:
                raise RuntimeError("Ten-frame baseline replay failed. See exports/replay_gate.json.")
            targets = json.loads((work / "provenance/droid_natural_targets.json").read_bytes())
            report = summarize(blocks, targets)
            assert report["scores"] == 84
            base.atomic(export / "summary.json", report)
            base.atomic(work / "completion.json", dict(status="complete", pairs=len(blocks)))
            print("DROID_TERMINAL_CONTEXT_COMPLETE", json.dumps(report), flush=True)
        except BaseException as exc:
            base.atomic(work / "completion.json", dict(status="failed", error_type=type(exc).__name__, message=str(exc)))
            raise
        finally:
            base.package(work)


if __name__ == "__main__":
    main()
