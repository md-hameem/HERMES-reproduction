"""Test direct visual attention to the terminal span with unchanged input positions."""
import argparse
import json
from pathlib import Path
from statistics import median
import time

import numpy as np

from natural_history_runner import auto, base, choose, load_frames

ARTIFACT = "HERMES_droid_terminal_attention_evidence.zip"
METHODS = ("native", "causal_control", "terminal_access", "earlier_access")
VISIBLE = {"causal_control": tuple(range(10)), "terminal_access": (3, 4, 5), "earlier_access": (0, 1, 2)}
REPLAY_ATOL = 0.05


def build_mask(embeddings, system_tokens, frame_tokens, kept_frames):
    import torch

    assert len(set(kept_frames)) == len(kept_frames) and all(0 <= i < 10 for i in kept_frames)
    length = embeddings.shape[1]
    text_start = system_tokens + 10*frame_tokens
    assert system_tokens > 0 and frame_tokens > 0 and text_start < length
    floor = torch.finfo(embeddings.dtype).min
    positions = torch.arange(length, device=embeddings.device)
    mask = torch.zeros((1, 1, length, length), dtype=embeddings.dtype, device=embeddings.device)
    mask.masked_fill_(positions[None, :] > positions[:, None], floor)
    for frame in sorted(set(range(10)) - set(kept_frames)):
        a = system_tokens+frame*frame_tokens
        mask[:, :, text_start:, a:a+frame_tokens] = floor
    return mask


def verify_hooks(lm, inputs, mask):
    """Check that Qwen passes the explicit mask to its first and last attention layers."""
    import torch

    assert lm.config._attn_implementation in {"sdpa", "eager"}
    dispatched = []

    def checked(_module, _args, kwargs):
        actual = kwargs["attention_mask"]
        assert actual.shape == mask.shape and torch.equal(actual, mask)
        dispatched.append(actual.shape[-1])

    layers = lm.model.layers
    selected = [layers[0].self_attn, layers[-1].self_attn] if len(layers) > 1 else [layers[0].self_attn]
    handles = [layer.register_forward_pre_hook(checked, with_kwargs=True) for layer in selected]
    try:
        lm(inputs_embeds=inputs, attention_mask=mask, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    assert len(dispatched) == len(selected)
    return len(dispatched)


def explicit_scores(model, system, visual, question, options, order, audited):
    import torch

    lm = model.language_model
    prompt, ids, permutation, prompt_sha = base.make_prompt(model, question, options, order, True)
    embeddings = torch.cat((system, visual, lm.get_input_embeddings()(prompt)), dim=1)
    assert visual.shape[1] % 10 == 0
    frame_tokens = visual.shape[1] // 10
    rows = {}
    for method, kept in VISIBLE.items():
        mask = build_mask(embeddings, system.shape[1], frame_tokens, kept)
        if method not in audited:
            audited[method] = verify_hooks(lm, embeddings, mask)
        begin = time.perf_counter()
        output = lm(inputs_embeds=embeddings, attention_mask=mask, use_cache=False, return_dict=True)
        scores = output.logits[0, -1, ids].float().cpu().tolist()
        del output
        values = [0.0] * len(permutation)
        for i, score in zip(permutation, scores, strict=True):
            values[i] = score
        assert all(np.isfinite(values))
        rows[method] = dict(logits=values, prediction=base.LETTERS[auto.best(values)],
                            lm_tokens=embeddings.shape[1], visual_tokens=visual.shape[1], prompt_sha256=prompt_sha,
                            visible_frames=list(kept), forward_s=time.perf_counter()-begin)
    return rows


def discrepancy(actual, expected):
    delta = max(abs(a-b) for a, b in zip(actual["logits"], expected["logits"], strict=True))
    equal = all(actual[k] == expected[k] for k in ("prediction", "prompt_sha256", "lm_tokens", "visual_tokens"))
    return delta, delta <= REPLAY_ATOL and equal


def compare(model, system, case, pixels, shared, audited):
    import torch

    result = {}
    for side, branch in case["branches"].items():
        frames = branch["frames"]
        names = sorted(set(frames["uniform"]) | set(frames["annotated"]))
        bank = dict(zip(names, auto.encode_frames(model, np.stack([pixels[n] for n in names])), strict=True))
        assert len(frames["annotated"]) == 6 and len(shared) == 4
        visual = torch.cat([bank[n] for n in frames["annotated"]]+shared, dim=1)
        answers, checks = {}, []
        for order in base.ORDERS:
            native = choose(model, system, visual, case["question"], case["options"], order)
            explicit = explicit_scores(model, system, visual, case["question"], case["options"], order, audited)
            answers[order] = dict(native=native, **explicit)
            assert len({r["lm_tokens"] for r in answers[order].values()}) == 1
            for gate, actual, expected in (
                ("archived_replay", native, branch["archived_native"][order]),
                ("explicit_control", explicit["causal_control"], native),
            ):
                delta, passed = discrepancy(actual, expected)
                checks.append(dict(gate=gate, order=order, max_abs=delta, passed=passed))
        result[side] = dict(answers=answers, checks=checks)
    return dict(pair_id=case["pair_id"], source_site=case["source_site"], scene_id=case["scene_id"], branches=result)


def summarize(blocks, targets):
    methods = {m: dict(robust_pairs=[], correct_branch_orders=0) for m in METHODS}
    deltas = []
    count = 0
    for row in blocks:
        pair = row["pair_id"]
        totals = dict.fromkeys(methods, 0)
        for side, branch in row["branches"].items():
            gold = targets[pair][side]
            g = base.LETTERS.index(gold)
            assert g in (0, 1)
            for by_method in branch["answers"].values():
                margins = {}
                for method, score in by_method.items():
                    totals[method] += score["prediction"] == gold
                    y = score["logits"]
                    margins[method] = y[g]-y[1-g]
                    count += 1
                deltas.append(margins["terminal_access"]-margins["causal_control"])
        for method, correct in totals.items():
            methods[method]["correct_branch_orders"] += correct
            if correct == 4:
                methods[method]["robust_pairs"].append(pair)
    terminal = set(methods["terminal_access"]["robust_pairs"])
    control = set(methods["causal_control"]["robust_pairs"])
    earlier = set(methods["earlier_access"]["robust_pairs"])
    gains = sorted(terminal-control-earlier)
    losses = sorted(control-terminal)
    return dict(pairs=len(blocks), scores=count, methods=methods, terminal_only_recoveries=gains,
                lost_vs_causal_control=losses, median_terminal_gold_margin_delta=median(deltas),
                feasibility_criterion_met=len(blocks) == 7 and len(gains) >= 2 and not losses)


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
            provenance = work / "provenance"
            spec = json.loads((provenance / "droid_attention_cases.json").read_bytes())
            assert len(spec["cases"]) == 7 and args.model.name == spec["model_revision"]
            assert base.digest(args.frames) == spec["frame_zip_sha256"]
            pixels = load_frames(args.frames, spec["frame_receipts"])
            print("DROID_ATTENTION_INPUTS_VERIFIED", len(pixels), flush=True)
            model, processor = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=0.5)
            assert model.dtype == torch.float16 and {str(p.device) for p in model.parameters()} == {"cuda:0"}
            assert json.loads(json.dumps(processor.video_processor.to_dict())) == spec["video_processor"]
            backend = model.language_model.config._attn_implementation
            if backend not in {"sdpa", "eager"}:
                raise RuntimeError(f"The pinned backend does not support the 4D mask: {backend}")
            base.atomic(export / "run_config.json", dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                        cuda=torch.version.cuda, model_revision=args.model.name, attention_backend=backend,
                        replay_atol=REPLAY_ATOL, frame_zip_sha256=spec["frame_zip_sha256"],
                        prior_archive_sha256=spec["prior_archive_sha256"], visible_visual_slots=VISIBLE))
            blocks, audited = [], {}
            with torch.inference_mode():
                system = model.language_model.get_input_embeddings()(
                    torch.as_tensor(model.init_prompt_ids, device=model.device)
                )
                names = spec["cases"][0]["recent_frames"]
                assert len(names) == len(set(names)) == 4
                shared = auto.encode_frames(model, np.stack([pixels[n] for n in names]))
                for i, case in enumerate(spec["cases"], 1):
                    assert case["recent_frames"] == names
                    blocks.append(compare(model, system, case, pixels, shared, audited))
                    base.atomic(export / "pair_scores.json", blocks)
                    print(f"DROID_ATTENTION_PAIR {i}/7 {case['pair_id']}", flush=True)
            checks = [dict(pair_id=b["pair_id"], branch=side, **c) for b in blocks
                      for side, branch in b["branches"].items() for c in branch["checks"]]
            failures = [row for row in checks if not row["passed"]]
            assert len(checks) == 56 and set(audited) == set(VISIBLE)
            maxima = {gate: max(r["max_abs"] for r in checks if r["gate"] == gate)
                      for gate in ("archived_replay", "explicit_control")}
            base.atomic(export / "replay_gate.json", dict(comparisons=len(checks), max_abs=maxima,
                        verified_attention_layers=audited, failures=failures))
            if failures:
                raise RuntimeError("Archive replay or explicit causal-mask control failed. See exports/replay_gate.json.")
            targets = json.loads((provenance / "droid_natural_targets.json").read_bytes())
            report = summarize(blocks, targets)
            assert report["scores"] == 112
            base.atomic(export / "summary.json", report)
            base.atomic(work / "completion.json", dict(status="complete", pairs=len(blocks)))
            print("DROID_TERMINAL_ATTENTION_COMPLETE", json.dumps(report), flush=True)
        except BaseException as exc:
            base.atomic(work / "completion.json", dict(status="failed", error_type=type(exc).__name__, message=str(exc)))
            raise
        finally:
            base.package(work)


if __name__ == "__main__":
    main()
