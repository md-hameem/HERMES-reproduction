"""Compare change-directed and uniform refinement in the same causal windows."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time

import temporal_runner as temporal

base = temporal.base
backbone = temporal.backbone
ARTIFACT = "HERMES_twinpast_refinement_evidence.zip"
METHODS = ("coarse_uniform", "local_uniform", "state_change")


def acquire(args, spec):
    paths, pending = {}, []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for plan, media, _ in spec["media_plan"]:
            pair = plan["pair_id"]
            paths[pair] = {}
            for branch, receipt in media["decoded"].items():
                name = Path(plan[f"video_{branch}"])
                options = [args.previous / "media" / name,
                           args.work.with_name("hermes_twinpast_diagnostic") / "media" / name,
                           args.media_reuse / name.relative_to("videos")]
                match = next((p for p in options if p.is_file() and base.digest(p) == receipt["sha256"]), None)
                if match is not None:
                    paths[pair][branch] = match
                    print(f"MEDIA_REUSED {name.as_posix()}", flush=True)
                else:
                    dest = paths[pair][branch] = args.work / "media" / name
                    pending.append(pool.submit(base.fetch, spec["media_source"], name.as_posix(), dest,
                                               receipt["sha256"], receipt["bytes"]))
        for download in pending:
            download.result()
    return paths


def refine(path, windows):
    import numpy as np
    from PIL import Image

    grid = [start + i/4 for start, end in windows for i in range(4*(end-start)+1)]
    frames, trace = temporal.decode_grid(path, grid)
    rows, picks = [], {"local_uniform": [], "state_change": []}
    for k in range(3):
        origin = 13*k
        chunk = frames[origin:origin+13]
        rgb = np.stack([np.asarray(Image.fromarray(f).resize((32, 18)), dtype=np.float32) for f in chunk])
        changes = np.abs(np.diff(rgb, axis=0)).mean(axis=(1,2,3)).tolist()
        peak = max(range(len(changes)), key=changes.__getitem__)
        uniform = [origin+3, origin+9]
        changed = [origin+peak, origin+peak+1]
        picks["local_uniform"].extend(uniform)
        picks["state_change"].extend(changed)
        rows.append(dict(window_s=windows[k], adjacent_rgb_mae=changes, uniform_slots=uniform,
                         change_slots=changed))
    assert all(len(v) == len(set(v)) == 6 for v in picks.values())
    return frames, trace, rows, picks


def branch_policy(model, path, case):
    import torch

    tick = time.perf_counter()
    frames, coarse = temporal.decode_grid(path, range(math.ceil(case["query_s"])))
    n = len(frames)-4
    bank = temporal.auto.encode_frames(model, frames)
    recent = bank[-4:]
    system = model.language_model.get_input_embeddings()(torch.as_tensor(model.init_prompt_ids, device=model.device))
    calls = []

    def score(prompt, visual, arm, order):
        ids, option_ids, perm, sha = prompt
        start = time.perf_counter()
        values, tokens = base.score(model, system, ids, option_ids, visual)
        canonical = [0.0]*len(perm)
        for i,v in zip(perm, values.tolist(), strict=True):
            canonical[i] = v
        calls.append(dict(phase=arm, order=order, forward_s=time.perf_counter()-start, lm_tokens=tokens,
                          visual_tokens=visual.shape[1], prompt_sha256=sha))
        return canonical

    probe = ("Does this earlier excerpt contain visual evidence needed to answer the following "
             "question about the specified event, state, or entity? " + case["question"])
    probes = {o: base.make_prompt(model, probe, ["Yes", "No"], o, True) for o in base.ORDERS}
    candidates = []
    for start in temporal.auto.spread(n-temporal.WINDOW_SPAN, temporal.WINDOW_CAP):
        stop = start+temporal.WINDOW_SPAN
        visual = torch.cat([bank[start], bank[stop]], dim=1)
        raw = {o: score(probes[o], visual, "shared_retrieval", o) for o in base.ORDERS}
        relevance = sum(v[0]-v[1] for v in raw.values()) / 2
        candidates.append(dict(slots=[start, stop], relevance=relevance, yes_no=raw))
    selected = temporal.select_windows(candidates)
    windows = [selected[i:i+2] for i in range(0, 6, 2)]
    samples, fine, changes, picks = refine(path, windows)
    assert len(samples)==39 and fine["timestamps_s"][-1]<coarse["timestamps_s"][-4]<case["query_s"]
    coarse_slots = temporal.auto.spread(n,6)
    features = {"coarse_uniform": [bank[i] for i in coarse_slots]}
    for arm, use in picks.items():
        features[arm] = temporal.auto.encode_frames(model, samples[use])
    inputs = {arm: torch.cat(v+recent, dim=1) for arm,v in features.items()}
    assert len({tuple(x.shape) for x in inputs.values()})==1
    answers = {}
    for o in base.ORDERS:
        prompt = base.make_prompt(model, case["question"], case["options"], o, True)
        logits = {m: score(prompt, inputs[m], m, o) for m in METHODS}
        answers[o] = {m: dict(logits=v, prediction=base.LETTERS[temporal.auto.best(v)]) for m,v in logits.items()}
    return dict(coarse=coarse, candidates=candidates, selected_windows=windows, refined=fine, changes=changes,
                coarse_uniform_slots=coarse_slots, refined_picks=picks, answers=answers, calls=calls,
                inspected_refinement_frames=len(samples), selected_frames_per_arm=6, wall_s=time.perf_counter()-tick)


def summarize(blocks, targets):
    methods = {}
    for arm in METHODS:
        robust, correct = [], 0
        for b in blocks:
            pair = b["pair_id"]
            answers = [b["branches"][side]["answers"][o][arm]["prediction"]==targets[pair][f"answer_{side}"]
                       for side in "ab" for o in base.ORDERS]
            correct += sum(answers)
            if all(answers):
                robust.append(pair)
        methods[arm] = dict(robust_pairs=robust, correct_branch_orders=correct)
    change = set(methods["state_change"]["robust_pairs"])
    uniform = set(methods["local_uniform"]["robust_pairs"])
    calls = Counter(c["phase"] for r in blocks for b in r["branches"].values() for c in b["calls"])
    return dict(pairs=len(blocks), methods=methods, state_change_only=sorted(change-uniform),
                local_uniform_only=sorted(uniform-change), state_change_net_robust_gain=len(change)-len(uniform),
                forward_calls_by_phase=dict(calls))


def execute(args):
    import torch
    from inference.llavaov_hermes import load_model

    work = args.work
    provenance = work/"provenance"
    spec = json.loads((provenance/"refinement_cases.json").read_bytes())
    assert args.model.name == backbone.MODEL_REVISION == spec["model_revision"]
    paths = acquire(args, spec)
    cases, decisions = temporal.screen_pairs(spec["cases"], spec["reference_ids"], paths, spec["expected_max_pairs"])
    exports = work/"exports"
    base.atomic(exports/"source_screen.json", dict(decisions=decisions, source_status=spec["source_status"]))
    if not cases:
        raise RuntimeError("No pairs passed the media and recent-frame screen")
    print(f"SOURCE_SCREEN_PASS {len(cases)}/{spec['expected_max_pairs']} pairs", flush=True)
    model, processor = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=0.5)
    assert model.dtype == torch.float16 and {str(p.device) for p in model.parameters()} == {"cuda:0"}
    assert json.loads(json.dumps(processor.video_processor.to_dict())) == spec["video_processor"]
    config = dict(model_revision=args.model.name, cuda=torch.version.cuda, torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(0), accepted=[c["pair_id"] for c in cases],
                  hashes={p.name: base.digest(p) for p in sorted(provenance.iterdir()) if p.is_file()})
    identity = base.sha(json.dumps(config, sort_keys=True).encode())
    base.atomic(exports/"run_config.json", config)
    checkpoints = work/"checkpoints"
    checkpoints.mkdir(exist_ok=True)
    blocks = []
    with torch.inference_mode():
        for i, case in enumerate(cases):
            pair = case["pair_id"]
            cp = checkpoints/f"{pair}.json"
            if cp.is_file():
                saved = json.loads(cp.read_bytes())
                assert saved["identity"] == identity and saved["block"]["pair_id"] == pair
                block = saved["block"]
            else:
                torch.cuda.reset_peak_memory_stats()
                branches = {b: branch_policy(model, paths[pair][b], case) for b in "ab"}
                assert branches["a"]["coarse"]["hashes"][-4:] == branches["b"]["coarse"]["hashes"][-4:]
                block = dict(pair_id=pair, branches=branches, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
                base.atomic(cp, dict(identity=identity, block=block))
            blocks.append(block)
            print(f"PAIR_DONE {i+1}/{len(cases)} {pair}", flush=True)
    targets = json.loads((provenance/"evaluation_targets.json").read_bytes())
    report = summarize(blocks, targets)
    base.atomic(exports/"summary.json", report)
    base.atomic(work/"completion.json", dict(status="complete", pairs=len(blocks)))
    print("TWINPAST_REFINEMENT_COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "model", "media-reuse", "previous"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    from filelock import FileLock

    with FileLock(str(args.work/"run.lock"), timeout=0):
        (args.work/"exports").mkdir(exist_ok=True)
        base.ARTIFACT = ARTIFACT
        base.atomic(args.work/"completion.json", dict(status="running"))
        try:
            execute(args)
        except BaseException as exc:
            base.atomic(args.work/"completion.json", dict(status="failed", error_type=type(exc).__name__, message=str(exc)))
            raise
        finally:
            base.package(args.work)
