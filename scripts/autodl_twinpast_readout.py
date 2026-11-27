"""Compare option-blind observation and direct readouts on archived TwinPast frames."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import automatic_runner as auto

base = auto.base
ARTIFACT = "HERMES_twinpast_readout_evidence.zip"
FRAME_SETS = ("coarse_uniform", "state_change")
REPLAY_ATOL = 0.05
NOTE_TOKENS = 24


def note_prompt(question):
    return ("Describe only the visible state of the object or event named in the question. "
            "Use the requested moment in the video. If the state is unclear, write 'unclear'. "
            "Write one short observation.\nQuestion: " + question)


def observe(model, system, visual, question):
    import torch

    tokenizer = model.processor.tokenizer
    text = model.get_prompt(note_prompt(question))
    tokens = tokenizer.encode(text, add_special_tokens=False)
    ids = torch.tensor([tokens], device=model.device)
    lm = model.language_model
    prompt = torch.cat((system, visual, lm.get_input_embeddings()(ids)), dim=1)
    stops = lm.generation_config.eos_token_id
    stops = set(stops if isinstance(stops, (list, tuple)) else [stops]) | {tokenizer.eos_token_id}
    generated, consumed = [], prompt.shape[1]
    start = time.perf_counter()
    output = lm(inputs_embeds=prompt, use_cache=True, return_dict=True)
    forwards = 1
    for step in range(NOTE_TOKENS):
        last = output.logits[:, -1, :]
        if not torch.isfinite(last).all():
            raise ValueError("Non-finite observation logits")
        token = last.argmax(dim=-1, keepdim=True)
        if token.item() in stops:
            break
        generated.append(token.item())
        if step + 1 < NOTE_TOKENS:
            output = lm(input_ids=token, past_key_values=output.past_key_values, use_cache=True, return_dict=True)
            consumed += 1
            forwards += 1
    note = " ".join(tokenizer.decode(generated, skip_special_tokens=True).split()) or "unclear"
    return dict(text=note, generated_token_ids=generated, forward_calls=forwards, input_tokens=consumed,
                prompt_sha256=base.sha(text.encode("utf-8")), forward_s=time.perf_counter()-start)


def choose(model, system, visual, question, options, order):
    prompt, option_ids, permutation, prompt_hash = base.make_prompt(model, question, options, order, True)
    start = time.perf_counter()
    scores, tokens = base.score(model, system, prompt, option_ids, visual)
    values = [0.0] * len(permutation)
    for i, score in zip(permutation, scores.tolist(), strict=True):
        values[i] = score
    return dict(logits=values, prediction=base.LETTERS[auto.best(values)], lm_tokens=tokens,
                visual_tokens=visual.shape[1], prompt_sha256=prompt_hash, forward_s=time.perf_counter()-start)


def fixed_inputs(model, path, branch):
    import torch

    coarse, fine = branch["coarse"], branch["rgb"]
    frames = base.decode(path, dict(source_frames=coarse["indices"], decoded_rgb_sha256=coarse["hashes"]))
    rgb = base.decode(path, dict(source_frames=fine["indices"], decoded_rgb_sha256=fine["hashes"]))
    # Preserve the original vision encoder batch boundaries for numerical replay.
    bank = auto.encode_frames(model, frames)
    recent = bank[-4:]
    history = [bank[i] for i in coarse["slots"]]
    return dict(coarse_uniform=torch.cat(history+recent, dim=1),
                state_change=torch.cat(auto.encode_frames(model, rgb)+recent, dim=1))


def acquire(args, spec):
    def clip(case, side):
        receipt = case["media"][side]
        relative = receipt["file"]
        cached = args.previous / "media" / relative
        if cached.is_file() and cached.stat().st_size == receipt["bytes"] and base.digest(cached) == receipt["sha256"]:
            print("MEDIA_REUSED", relative, flush=True)
            return cached
        target = args.work / "media" / relative
        base.fetch(spec["media_source"], relative, target, receipt["sha256"], receipt["bytes"])
        return target

    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {(case["pair_id"], side): pool.submit(clip, case, side)
                for case in spec["cases"] for side in "ab"}
        return {key: job.result() for key, job in jobs.items()}


def replay(case, system, model, paths):
    pair = case["pair_id"]
    records, visuals, failures = {}, {}, []
    for side in "ab":
        archived = case["branches"][side]
        for frame_set, visual in fixed_inputs(model, paths[pair, side], archived).items():
            visuals[side, frame_set] = visual.cpu()
            for order in base.ORDERS:
                result = choose(model, system, visual, case["question"], case["options"], order)
                expected = archived["direct"][frame_set][order]
                delta = max(abs(a-b) for a, b in zip(result["logits"], expected["logits"], strict=True))
                result["replay_max_abs"] = delta
                if delta > REPLAY_ATOL or any(result[k] != expected[k] for k in
                                               ("prediction", "prompt_sha256", "lm_tokens", "visual_tokens")):
                    failures.append(dict(pair_id=pair, branch=side, frame_set=frame_set, order=order,
                                         archived=expected, replay=result))
                records[side, frame_set, order] = result
    return records, visuals, failures


def compare(case, model, system, packed):
    direct, visuals, _ = packed
    rows = []
    for (side, frame_set), cpu_visual in visuals.items():
        visual = cpu_visual.to(model.device)
        note = observe(model, system, visual, case["question"])
        with_note = case["question"] + "\nVisual observation from the first pass (may be incomplete): " + note["text"]
        results = {order: dict(direct=direct[side, frame_set, order],
                               observation=choose(model, system, visual, with_note, case["options"], order))
                   for order in base.ORDERS}
        rows.append(dict(branch=side, frame_set=frame_set, observation=note, orders=results))
    return dict(pair_id=case["pair_id"], rows=rows)


def summarize(blocks, targets):
    summary = {}
    for frame_set in FRAME_SETS:
        methods = {m: dict(robust_pairs=[], correct_branch_orders=0, forward_calls=0, input_tokens=0, forward_s=0.0)
                   for m in ("direct", "observation")}
        max_replay = 0.0
        for block in blocks:
            count = dict.fromkeys(methods, 0)
            pair = block["pair_id"]
            for branch in (r for r in block["rows"] if r["frame_set"] == frame_set):
                note = branch["observation"]
                for field in ("forward_calls", "input_tokens", "forward_s"):
                    methods["observation"][field] += note[field]
                for result in branch["orders"].values():
                    max_replay = max(max_replay, result["direct"]["replay_max_abs"])
                    for method, row in result.items():
                        count[method] += row["prediction"] == targets[pair][f"answer_{branch['branch']}"]
                        methods[method]["forward_calls"] += 1
                        methods[method]["input_tokens"] += row["lm_tokens"]
                        methods[method]["forward_s"] += row["forward_s"]
            for method, correct in count.items():
                methods[method]["correct_branch_orders"] += correct
                if correct == 4:
                    methods[method]["robust_pairs"].append(pair)
        direct, observed = (set(methods[m]["robust_pairs"]) for m in methods)
        summary[frame_set] = dict(methods=methods, recoveries=sorted(observed-direct),
                                  regressions=sorted(direct-observed), max_replay_abs=max_replay)
    primary = summary["coarse_uniform"]
    score_count = sum(len(result) for block in blocks for row in block["rows"] for result in row["orders"].values())
    return dict(pairs=len(blocks), choice_scores=score_count, frame_sets=summary,
                development_criterion_met=len(blocks) == 8 and bool(primary["recoveries"]) and not primary["regressions"])


def execute(args):
    import torch
    from inference.llavaov_hermes import load_model

    work = args.work
    provenance = work / "provenance"
    spec = json.loads((provenance / "readout_cases.json").read_bytes())
    if args.model.name != spec["model_revision"]:
        raise ValueError("Model revision differs from the archived run")
    paths = acquire(args, spec)
    model, processor = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=0.5)
    assert model.dtype == torch.float16 and {str(p.device) for p in model.parameters()} == {"cuda:0"}
    assert json.loads(json.dumps(processor.video_processor.to_dict())) == spec["video_processor"]
    exports = work / "exports"
    base.atomic(exports / "run_config.json", dict(model_revision=args.model.name, torch=torch.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), replay_atol=REPLAY_ATOL,
                observation_max_new_tokens=NOTE_TOKENS, source_archive_sha256=spec["source_archive_sha256"]))
    with torch.inference_mode():
        system = model.language_model.get_input_embeddings()(torch.as_tensor(model.init_prompt_ids, device=model.device))
        replays = []
        print("REPLAY_STARTED", flush=True)
        for i, case in enumerate(spec["cases"], 1):
            replays.append(replay(case, system, model, paths))
            print(f"REPLAY_PAIR {i}/8 {case['pair_id']}", flush=True)
        failures = [failure for _, _, checks in replays for failure in checks]
        base.atomic(exports / "replay_gate.json", dict(scores=sum(len(r) for r, _, _ in replays),
                    max_abs=max(x["replay_max_abs"] for r, _, _ in replays for x in r.values()), failures=failures))
        if failures:
            raise RuntimeError("Archived direct replay failed. See exports/replay_gate.json.")
        print("REPLAY_PASS", flush=True)
        blocks = []
        for i, case in enumerate(spec["cases"], 1):
            block = compare(case, model, system, replays[i-1])
            base.atomic(work / "checkpoints" / f"{case['pair_id']}.json", block)
            blocks.append(block)
            print(f"READOUT_PAIR {i}/8 {case['pair_id']}", flush=True)
    targets = json.loads((provenance / "evaluation_targets.json").read_bytes())
    assert set(targets) == {b["pair_id"] for b in blocks}
    report = summarize(blocks, targets)
    base.atomic(exports / "summary.json", report)
    base.atomic(work / "completion.json", dict(status="complete", pairs=len(blocks)))
    print("TWINPAST_READOUT_COMPLETE", json.dumps(report), flush=True)


def main():
    from filelock import FileLock

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "model", "previous"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    work = args.work
    with FileLock(str(work / "run.lock"), timeout=0):
        for name in ("exports", "checkpoints"):
            (work / name).mkdir(exist_ok=True)
        # A rerun must not package scores from an earlier incomplete attempt.
        for case in json.loads((work / "provenance/readout_cases.json").read_bytes())["cases"]:
            (work / "checkpoints" / f"{case['pair_id']}.json").unlink(missing_ok=True)
        for name in ("summary.json", "replay_gate.json", "run_config.json"):
            (work / "exports" / name).unlink(missing_ok=True)
        base.ARTIFACT = ARTIFACT
        base.atomic(work / "completion.json", dict(status="running"))
        try:
            execute(args)
        except BaseException as exc:
            base.atomic(work / "completion.json", dict(status="failed", error_type=type(exc).__name__, message=str(exc)))
            raise
        finally:
            base.package(work)


if __name__ == "__main__":
    main()
