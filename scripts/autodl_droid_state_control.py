"""Run the aligned and off-grid controlled DROID state readouts."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from readout_runner import auto, base, choose

ARTIFACT = "HERMES_droid_state_control_evidence.zip"
METHODS = ("aligned_uniform", "off_grid_uniform", "off_grid_annotated", "cue_only")


def frame_schedule(grids):
    history, recent = grids["history_grid_s"], grids["recent_grid_s"]
    assert len(history) == 6 and len(recent) == 4 and max(history) < min(recent) < grids["query_s"]
    aligned, off_grid = (grids[name] for name in ("aligned", "off_grid"))

    def visible(schedule, condition):
        return [condition["cue_start_s"] <= timestamp < condition["cue_end_s"] for timestamp in schedule]

    return dict(aligned_uniform=visible(history+recent, aligned), off_grid_uniform=visible(history+recent, off_grid),
                off_grid_annotated=visible(off_grid["oracle_grid_s"]+recent, off_grid), cue_only=[True])


def compare(model, system, case, grids, work, image_receipts):
    import torch

    schedules = frame_schedule(grids)
    assert schedules["aligned_uniform"] == schedules["off_grid_annotated"]
    assert sum(schedules["aligned_uniform"]) == 1 and not any(schedules["off_grid_uniform"])
    inputs, filler_feature = {}, None
    for side in "ab":
        path = work / "input_frames" / f"{case['pair_id']}_{side}.png"
        receipt = image_receipts[path.name]
        assert base.digest(path) == receipt["png_sha256"]
        with Image.open(path) as image:
            cue = np.array(image.convert("RGB"))
        assert list(cue.shape) == receipt["shape"] and base.sha(cue.tobytes()) == receipt["rgb_sha256"]
        if filler_feature is None:
            neutral = np.empty_like(cue)
            neutral[:] = grids["neutral_rgb"]
            filler_feature, = auto.encode_frames(model, neutral[None])
        assert cue.shape == neutral.shape and not np.array_equal(cue, neutral)
        cue_feature, = auto.encode_frames(model, cue[None])
        inputs[side] = {method: torch.cat([cue_feature if visible else filler_feature for visible in schedule], dim=1)
                        for method, schedule in schedules.items()}
    results = {side: {order: {method: choose(model, system, video, case["question"], case["options"], order)
                              for method, video in inputs[side].items()}
                      for order in base.ORDERS} for side in "ab"}
    assert not any(schedules["off_grid_annotated"][-4:])
    for order in base.ORDERS:
        hidden_a, hidden_b = (results[side][order]["off_grid_uniform"] for side in "ab")
        assert hidden_a["logits"] == hidden_b["logits"]
        for side in "ab":
            by_method = results[side][order]
            assert by_method["aligned_uniform"]["logits"] == by_method["off_grid_annotated"]["logits"]
            assert len({by_method[m]["visual_tokens"] for m in METHODS[:3]}) == 1
    return dict(pair_id=case["pair_id"], source_group=case["source_group"], scores=results)


def summarize(blocks, targets):
    result, scored = {}, 0
    for method in METHODS:
        robust, correct = [], 0
        for block in blocks:
            pair = block["pair_id"]
            matches = [block["scores"][side][order][method]["prediction"] == targets[pair][side]
                       for side in "ab" for order in base.ORDERS]
            scored += len(matches)
            correct += sum(matches)
            if all(matches):
                robust.append(pair)
        result[method] = dict(robust_pairs=robust, correct_branch_orders=correct)
    assert not result["off_grid_uniform"]["robust_pairs"]
    assert result["aligned_uniform"] == result["off_grid_annotated"]
    return dict(pairs=len(blocks), scores=scored, methods=result,
                controlled_capability_criterion_met=len(result["off_grid_annotated"]["robust_pairs"]) >= 4)


def main():
    import torch
    from filelock import FileLock
    from inference.llavaov_hermes import load_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()
    work = args.work
    with FileLock(str(work / "run.lock"), timeout=0):
        export = work / "exports"
        export.mkdir(exist_ok=True)
        summary = export / "summary.json"
        summary.unlink(missing_ok=True)
        (export / "pair_scores.json").unlink(missing_ok=True)
        base.ARTIFACT = ARTIFACT
        base.atomic(work / "completion.json", dict(status="running"))
        try:
            provenance = work / "provenance"
            spec = json.loads((provenance / "droid_cases.json").read_bytes())
            image_receipts = json.loads((provenance / "droid_frame_receipts.json").read_bytes())
            assert args.model.name == spec["model_revision"]
            model, processor = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=0.5)
            assert model.dtype == torch.float16 and {str(p.device) for p in model.parameters()} == {"cuda:0"}
            assert json.loads(json.dumps(processor.video_processor.to_dict())) == spec["video_processor"]
            base.atomic(export / "run_config.json", dict(model_revision=args.model.name, gpu=torch.cuda.get_device_name(),
                        torch=torch.__version__, cuda=torch.version.cuda, manifest_sha256=spec["manifest_sha256"]))
            blocks = []
            with torch.inference_mode():
                system = model.language_model.get_input_embeddings()(
                    torch.as_tensor(model.init_prompt_ids, device=model.device)
                )
                for i, case in enumerate(spec["cases"], start=1):
                    blocks.append(compare(model, system, case, spec["timeline"], work, image_receipts))
                    print(f"DROID_CONTROL_PAIR {i}/6 {case['pair_id']}", flush=True)
            base.atomic(export / "pair_scores.json", blocks)
            targets = json.loads((provenance / "droid_targets.json").read_bytes())
            result = summarize(blocks, targets)
            assert result["pairs"] == 6 and result["scores"] == 96
            base.atomic(summary, result)
            base.atomic(work / "completion.json", dict(status="complete", pairs=len(blocks)))
            print("DROID_STATE_CONTROL_COMPLETE", json.dumps(result), flush=True)
        except BaseException as exc:
            base.atomic(work / "completion.json", dict(status="failed", error_type=type(exc).__name__, message=str(exc)))
            raise
        finally:
            base.package(work)


if __name__ == "__main__":
    main()
