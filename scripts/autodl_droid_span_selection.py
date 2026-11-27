"""Compare question-conditioned visual spans with matched controls on fresh DROID scenes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from droid_attention_runner import build_mask, discrepancy, verify_hooks
from droid_span_selector import RECENCY, endpoint_window, select_window
from natural_history_runner import auto, base, choose, load_frames

ARTIFACT = "HERMES_droid_span_selection_evidence.zip"
METHODS = ("native", "recency", "endpoint", "selector", "annotated")
TASKS = ("first", "last")


def score_masks(model, system, visual, question, order, windows, verified):
    import torch

    lm = model.language_model
    prompt, ids, permutation, prompt_sha = base.make_prompt(model, question, ["Open", "Closed"], order, True)
    inputs = torch.cat((system, visual, lm.get_input_embeddings()(prompt)), dim=1)
    assert visual.shape[1] % 10 == 0
    scores = {}
    for window in dict.fromkeys(windows.values()):
        if inputs.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        mask = build_mask(inputs, system.shape[1], visual.shape[1] // 10, window)
        output = lm(inputs_embeds=inputs, attention_mask=mask, use_cache=False, return_dict=True)
        logits = output.logits[0, -1, ids].float().cpu().tolist()
        del output
        values = [0.0, 0.0]
        for index, score in zip(permutation, logits, strict=True):
            values[index] = score
        if not np.isfinite(values).all():
            raise ValueError("Choice logits are not finite")
        scores[window] = dict(logits=values, prediction=base.LETTERS[auto.best(values)],
                              prompt_sha256=prompt_sha, lm_tokens=inputs.shape[1], visual_tokens=visual.shape[1],
                              slots=window, forward_s=time.perf_counter()-start)
        kind = "causal" if len(window) == 10 else "three_frame"
        if kind not in verified:
            verified[kind] = verify_hooks(lm, inputs, mask)
    return {method: scores[window] for method, window in windows.items()}


def evaluate(model, system, case, pixels, verified):
    import torch

    results = {}
    for side, branch in case["branches"].items():
        start = time.perf_counter()
        encoded = auto.encode_frames(model, np.stack([pixels[name] for name in branch["frames"]]))
        assert len(encoded) == 10 and len({tuple(frame.shape) for frame in encoded}) == 1
        if system.is_cuda:
            torch.cuda.synchronize()
        encoding_s = time.perf_counter()-start
        visual = torch.cat(encoded, dim=1)
        start = time.perf_counter()
        vectors = torch.cat([frame.float().mean(dim=1) for frame in encoded]).cpu().numpy()
        pooling_s = time.perf_counter()-start
        tasks = {}
        for task, question in case["questions"].items():
            start = time.perf_counter()
            selected, contrast = select_window(question, vectors)
            selector_s = time.perf_counter()-start
            label = branch["labels"][task]
            windows = dict(recency=RECENCY, endpoint=endpoint_window(question), selector=selected,
                           annotated=tuple(label["annotation_window"]), causal=tuple(range(10)))
            items = {}
            for order in base.ORDERS:
                native = choose(model, system, visual, question, ["Open", "Closed"], order)
                rows = score_masks(model, system, visual, question, order, windows, verified)
                causal = rows.pop("causal")
                error, passed = discrepancy(causal, native)
                items[order] = dict(native=native, **rows, causal=causal, causal_max_abs=error, passed=passed)
            tasks[task] = dict(selection=windows, contrast=contrast, selector_s=selector_s, orders=items)
        results[side] = dict(encoding_s=encoding_s, selector_pooling_s=pooling_s, tasks=tasks)
    return dict(pair_id=case["pair_id"], branches=results)


def summarize(blocks, cases):
    metrics = {task: {m: Counter(robust=0, correct=0, support=0) for m in METHODS} for task in TASKS}
    scene_deltas = {m: [] for m in ("recency", "endpoint")}
    for item, case in zip(blocks, cases, strict=True):
        assert item["pair_id"] == case["pair_id"]
        delta = Counter()
        for task in TASKS:
            good = Counter()
            for side in "ab":
                label = case["branches"][side]["labels"][task]
                row = item["branches"][side]["tasks"][task]
                for method in METHODS:
                    good[method] += sum(s[method]["prediction"] == label["gold"] for s in row["orders"].values())
                    keys = row["selection"].get(method, range(10))
                    metrics[task][method]["support"] += len(set(keys) & set(label["support_slots"]))
            for method in METHODS:
                metrics[task][method]["robust"] += good[method] == 4
                metrics[task][method]["correct"] += good[method]
            for method in scene_deltas:
                delta[method] += int(good["selector"] == 4) - int(good[method] == 4)
        for method in scene_deltas:
            scene_deltas[method].append(delta[method] / len(TASKS))
    rng = np.random.default_rng(20261127)
    picks = rng.integers(len(blocks), size=(10_000, len(blocks)))
    differences = {}
    for method, values in scene_deltas.items():
        bootstrap = np.asarray(values)[picks].mean(axis=1)
        differences[method] = dict(absolute=float(np.mean(values)),
                                   scene_bootstrap_95=np.quantile(bootstrap, [.025, .975]).tolist())
    return dict(scenes=len(blocks), tasks=metrics, contrasts=differences,
                total_reported_choice_scores=len(blocks) * len(TASKS) * 4 * len(METHODS))


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
        for name in ("pair_scores.json", "run_config.json", "summary.json"):
            (export / name).unlink(missing_ok=True)
        base.atomic(work / "completion.json", dict(status="running"))
        try:
            spec = json.loads((work / "provenance/droid_span_cohort.json").read_bytes())
            assert args.model.name == spec["model_revision"] and base.digest(args.frames) == spec["frame_zip_sha256"]
            data = load_frames(args.frames, spec["frame_receipts"])
            cases = spec["cases"]
            assert len(cases) >= 6 and len({row["source_site"] for row in cases}) >= 3
            print("DROID_SPAN_FRAMES_VERIFIED", len(data), flush=True)
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
                for number, case in enumerate(cases, 1):
                    row = evaluate(model, system, case, data, verified)
                    blocks.append(row)
                    base.atomic(export / "pair_scores.json", blocks)
                    if not all(trace["passed"] for b in row["branches"].values() for t in b["tasks"].values()
                               for trace in t["orders"].values()):
                        raise RuntimeError(f"Causal-mask parity failed on {case['pair_id']}")
                    print(f"DROID_SPAN_SCENE {number}/{len(cases)} {case['pair_id']}", flush=True)
            report = summarize(blocks, cases)
            report["verified_attention_layers"] = verified
            base.atomic(export / "summary.json", report)
            base.atomic(work / "completion.json", dict(status="complete", scenes=len(blocks)))
            print("DROID_SPAN_SELECTION_COMPLETE", json.dumps(report), flush=True)
        except BaseException as exc:
            base.atomic(work / "completion.json", dict(status="failed", error_type=type(exc).__name__, message=str(exc)))
            raise
        finally:
            base.package(work)


if __name__ == "__main__":
    main()
