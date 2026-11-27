"""Test annotated interior evidence against fixed windows on seven frozen video clips."""
import argparse
import json
from pathlib import Path

import numpy as np

from droid_attention_runner import discrepancy
from droid_span_runner import score_masks
from natural_history_runner import auto, base, choose, load_frames

ARTIFACT = "HERMES_droid_interior_evidence.zip"
METHODS = ("native", "fixed", "recency", "annotated")


def evaluate(model, system, case, pixels, fixed, verified):
    import torch

    embeddings = auto.encode_frames(model, np.stack([pixels[name] for name in case["frames"]]))
    assert len(embeddings) == 10 and len({tuple(frame.shape) for frame in embeddings}) == 1
    visual = torch.cat(embeddings, dim=1)
    question = case["question"]
    windows = dict(fixed=fixed, recency=(7, 8, 9), annotated=tuple(case["annotated_window"]),
                   causal=tuple(range(10)))
    results = {}
    for order in base.ORDERS:
        native = choose(model, system, visual, question, ["Open", "Closed"], order)
        rows = score_masks(model, system, visual, question, order, windows, verified)
        error, passed = discrepancy(rows["causal"], native)
        results[order] = dict(native=native, **rows, parity_error=error, parity_passed=passed)
    gold = "A" if case["state"] == "open" else "B"
    return dict(case_id=case["id"], gold=gold, orders=results)


def summarize(blocks):
    by_method = {m: dict(correct_orders=0, correct_both_orders=[]) for m in METHODS}
    for block in blocks:
        for method in METHODS:
            n = sum(row[method]["prediction"] == block["gold"] for row in block["orders"].values())
            by_method[method]["correct_orders"] += n
            if n == 2:
                by_method[method]["correct_both_orders"].append(block["case_id"])
    contrasts = {}
    selected = set(by_method["annotated"]["correct_both_orders"])
    for method in ("fixed", "recency", "native"):
        reference = set(by_method[method]["correct_both_orders"])
        contrasts[method] = dict(recoveries=sorted(selected-reference), losses=sorted(reference-selected),
                                 net=len(selected)-len(reference))
    return dict(clips=len(blocks), choice_scores=len(blocks)*2*len(METHODS),
                by_method=by_method, annotated_contrasts=contrasts)


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
        base.ARTIFACT = ARTIFACT
        export = work / "exports"
        export.mkdir(exist_ok=True)
        for name in ("scores.json", "summary.json", "run_config.json"):
            (export / name).unlink(missing_ok=True)
        base.atomic(work / "completion.json", dict(status="running"))
        try:
            spec = json.loads((work / "provenance/droid_interior_cases.json").read_bytes())
            assert args.model.name == spec["model_revision"] and base.digest(args.frames) == spec["frame_zip_sha256"]
            pixels = load_frames(args.frames, spec["frame_receipts"])
            assert len(spec["cases"]) == 7 and len(pixels) == 70
            print("DROID_INTERIOR_FRAMES_VERIFIED", len(pixels), flush=True)
            model, processor = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=0.5)
            assert model.dtype == torch.float16 and {str(p.device) for p in model.parameters()} == {"cuda:0"}
            assert json.loads(json.dumps(processor.video_processor.to_dict())) == spec["video_processor"]
            base.atomic(export / "run_config.json", dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                        model_revision=args.model.name, attention_backend=model.language_model.config._attn_implementation))
            blocks, verified = [], {}
            with torch.inference_mode():
                system = model.language_model.get_input_embeddings()(
                    torch.as_tensor(model.init_prompt_ids, device=model.device)
                )
                for i, case in enumerate(spec["cases"], 1):
                    row = evaluate(model, system, case, pixels, tuple(spec["static_window"]), verified)
                    blocks.append(row)
                    base.atomic(export / "scores.json", blocks)
                    if not all(order["parity_passed"] for order in row["orders"].values()):
                        raise RuntimeError(f"Causal-mask parity failed on {case['id']}")
                    print(f"DROID_INTERIOR_CLIP {i}/7 {case['id']}", flush=True)
            report = summarize(blocks)
            report["verified_attention_layers"] = verified
            base.atomic(export / "summary.json", report)
            base.atomic(work / "completion.json", dict(status="complete", clips=len(blocks)))
            print("DROID_INTERIOR_COMPLETE", json.dumps(report), flush=True)
        except BaseException as exc:
            base.atomic(work / "completion.json", dict(status="failed", error_type=type(exc).__name__, message=str(exc)))
            raise
        finally:
            base.package(work)


if __name__ == "__main__":
    main()
