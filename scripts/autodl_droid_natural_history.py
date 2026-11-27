"""Compare uniform and annotated real histories at a matched ten-frame budget."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from readout_runner import auto, base, choose

ARTIFACT = "HERMES_droid_natural_history_evidence.zip"
METHODS = ("uniform", "annotated")


def load_frames(path, receipts):
    with ZipFile(path) as zipped:
        names = zipped.namelist()
        assert len(names) == len(set(names)) and set(names) == set(receipts)
        frames = {}
        for name in names:
            raw = zipped.read(name)
            receipt = receipts[name]
            assert base.sha(raw) == receipt["png_sha256"]
            with Image.open(BytesIO(raw)) as image:
                assert image.size == (640, 360) and image.mode == "RGB"
                rgb = np.array(image)
            assert base.sha(rgb.tobytes()) == receipt["rgb_sha256"]
            frames[name] = rgb
    return frames


def compare(model, system, case, pixels, recent_features):
    import torch

    branches = {}
    for side, branch in case["branches"].items():
        names = sorted(set(branch["frames"]["uniform"]) | set(branch["frames"]["annotated"]))
        bank = dict(zip(names, auto.encode_frames(model, np.stack([pixels[name] for name in names])), strict=True))
        visuals = {}
        for method, path in branch["frames"].items():
            assert len(path) == len(set(path)) == 6
            assert set(path).isdisjoint(case["recent_frames"])
            visuals[method] = torch.cat([bank[name] for name in path] + recent_features, dim=1)
        assert len({tuple(v.shape) for v in visuals.values()}) == 1
        results, repeats = {}, {}
        for order in base.ORDERS:
            results[order] = {
                m: choose(model, system, visual, case["question"], case["options"], order)
                for m, visual in visuals.items()
            }
            repeat = choose(model, system, visuals["uniform"], case["question"], case["options"], order)
            left = results[order]["uniform"]
            repeats[order] = max(abs(a-b) for a, b in zip(left["logits"], repeat["logits"], strict=True))
            if repeats[order] != 0.0 or left["prediction"] != repeat["prediction"]:
                raise RuntimeError(f"Uniform replay differs for {case['pair_id']}/{side}/{order}")
        branches[side] = dict(answers=results, repeat_max_abs=repeats)
    return dict(pair_id=case["pair_id"], source_site=case["source_site"], scene_id=case["scene_id"], branches=branches)


def summarize(rows, targets):
    methods = {m: dict(robust_pairs=[], correct_branch_orders=0) for m in METHODS}
    count, replay_count = 0, 0
    for pair in rows:
        label = pair["pair_id"]
        for method in METHODS:
            correct = [r["answers"][o][method]["prediction"] == targets[label][side]
                       for side, r in pair["branches"].items() for o in base.ORDERS]
            methods[method]["correct_branch_orders"] += sum(correct)
            count += len(correct)
            if all(correct):
                methods[method]["robust_pairs"].append(label)
        replay_count += sum(len(r["repeat_max_abs"]) for r in pair["branches"].values())
    uniform, annotated = (set(methods[m]["robust_pairs"]) for m in METHODS)
    recovered, lost = sorted(annotated-uniform), sorted(uniform-annotated)
    return dict(pairs=len(rows), scores=count, uniform_replays=replay_count, methods=methods,
                recoveries=recovered, regressions=lost,
                exploratory_criterion_met=len(rows)==7 and len(recovered) >= 2 and not lost)


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
        for name in ("summary.json", "pair_scores.json", "run_config.json"):
            (export / name).unlink(missing_ok=True)
        base.ARTIFACT = ARTIFACT
        base.atomic(work / "completion.json", dict(status="running"))
        try:
            provenance = work / "provenance"
            spec = json.loads((provenance / "droid_natural_cases.json").read_bytes())
            assert len(spec["cases"]) == 7 and args.model.name == spec["model_revision"]
            assert base.digest(args.frames) == spec["frame_zip_sha256"]
            pixels = load_frames(args.frames, spec["frame_receipts"])
            print("DROID_NATURAL_INPUTS_VERIFIED", len(pixels), flush=True)
            model, processor = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=0.5)
            assert model.dtype == torch.float16 and {str(p.device) for p in model.parameters()} == {"cuda:0"}
            assert json.loads(json.dumps(processor.video_processor.to_dict())) == spec["video_processor"]
            base.atomic(export / "run_config.json", dict(model_revision=args.model.name, gpu=torch.cuda.get_device_name(),
                         torch=torch.__version__, cuda=torch.version.cuda, manifest_sha256=spec["manifest_sha256"],
                         frame_zip_sha256=spec["frame_zip_sha256"]))
            rows = []
            with torch.inference_mode():
                system = model.language_model.get_input_embeddings()(
                    torch.as_tensor(model.init_prompt_ids, device=model.device)
                )
                recent_names = spec["cases"][0]["recent_frames"]
                assert len(recent_names) == len(set(recent_names)) == 4
                recent = auto.encode_frames(model, np.stack([pixels[name] for name in recent_names]))
                for i, case in enumerate(spec["cases"], start=1):
                    assert case["recent_frames"] == recent_names
                    rows.append(compare(model, system, case, pixels, recent))
                    base.atomic(export / "pair_scores.json", rows)
                    print(f"DROID_NATURAL_PAIR {i}/7 {case['pair_id']}", flush=True)
            targets = json.loads((provenance / "droid_natural_targets.json").read_bytes())
            report = summarize(rows, targets)
            assert report["scores"] == 56 and report["uniform_replays"] == 28
            base.atomic(export / "summary.json", report)
            base.atomic(work / "completion.json", dict(status="complete", pairs=len(rows)))
            print("DROID_NATURAL_HISTORY_COMPLETE", json.dumps(report), flush=True)
        except BaseException as exc:
            base.atomic(work / "completion.json", dict(status="failed", error_type=type(exc).__name__, message=str(exc)))
            raise
        finally:
            base.package(work)


if __name__ == "__main__":
    main()
