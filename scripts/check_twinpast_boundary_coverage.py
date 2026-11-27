"""Test a scene-cut constraint on the frozen TwinPast windows without model inference."""
import argparse
import json
from pathlib import Path

from decord import VideoReader
import numpy as np

from autodl_twinpast_diagnostic import digest as media_digest
from validate_twinpast_refinement_evidence import ARCHIVE, MANIFEST, ROOT, digest, verify_inventory

from twinpast_boundary_policy import FPS, THRESHOLD, pick, screen_candidates, select

OUTCOMES = ("any_interval", "final_interval")


def hit(times, intervals):
    return any(lo-1e-6 <= t <= hi+1e-6 for t in times for lo, hi in intervals)


def evaluate(path, receipt, trace, intervals):
    assert path.stat().st_size == receipt["bytes"] and media_digest(path) == receipt["sha256"], path
    reader = VideoReader(str(path), num_threads=1)
    stamps = np.asarray(reader.get_frame_timestamp(list(range(len(reader)))), dtype=np.float64)
    starts = stamps[:, 0] - stamps[0, 0]
    first_recent = trace["coarse"]["indices"][-4]
    options, decoded = screen_candidates(reader, starts, trace["candidates"], first_recent)
    original, constrained = select(options)
    assert [list(w["window"]) for w in original] == trace["selected_windows"]
    baseline = pick(original, False)
    assert baseline == [trace["refined"]["indices"][i] for i in trace["refined_picks"]["local_uniform"]]
    histories = dict(local_uniform=baseline, rank_only=pick(constrained, False),
                     anchor_only=pick(original, True), combined=pick(constrained, True))
    assert all(max(indices) < first_recent for indices in histories.values())
    overlap = None
    if intervals:
        pools = {method: starts[indices] for method, indices in histories.items()}
        pools.update(all_candidates=starts[[i for w in options for i in w["indices"]]],
                     original_windows=starts[[i for w in original for i in w["indices"]]],
                     constrained_windows=starts[[i for w in constrained for i in w["indices"]]])
        last = [max(intervals, key=lambda interval: interval[1])]
        overlap = {name: dict(any_interval=hit(times, intervals), final_interval=hit(times, last))
                   for name, times in pools.items()}
    return dict(decoded_probe_frames=decoded, candidates=options,
                original_windows=[w["window"] for w in original],
                constrained_windows=[w["window"] for w in constrained], history_indices=histories,
                overlap=overlap)


def summarize(rows):
    annotated = [row for row in rows if row["overlap"] is not None]
    coverage = {name: {key: sum(r["overlap"][name][key] for r in annotated) for key in OUTCOMES}
                for name in annotated[0]["overlap"]}
    paired_annotations = {}
    for row in annotated:
        paired_annotations.setdefault(row["pair_id"], []).append(row["overlap"])
    paired_annotations = [branches for branches in paired_annotations.values() if len(branches) == 2]
    both_final = {name: sum(all(b[name]["final_interval"] for b in pair) for pair in paired_annotations)
                  for name in coverage}
    paired = {}
    for arm in ("rank_only", "anchor_only", "combined"):
        paired[arm] = {}
        for key in OUTCOMES:
            gains, losses = [], []
            for row in annotated:
                before, after = (row["overlap"][method][key] for method in ("local_uniform", arm))
                label = f"{row['pair_id']}/{row['branch']}"
                if after and not before:
                    gains.append(label)
                if before and not after:
                    losses.append(label)
            paired[arm][key] = dict(gained=gains, lost=losses)
    target = paired["combined"]["final_interval"]
    return dict(branches=len(rows), annotated_branches=len(annotated), coverage=coverage, paired=paired,
                pairs_annotated_on_both_branches=len(paired_annotations), pairs_with_both_final_intervals=both_final,
                changed_rankings=sum(r["original_windows"] != r["constrained_windows"] for r in rows),
                changed_combined_history=sum(r["history_indices"]["local_uniform"] != r["history_indices"]["combined"]
                                             for r in rows),
                decoded_probe_frames=sum(r["decoded_probe_frames"] for r in rows),
                development_criterion_met=len(target["gained"]) >= 2 and not target["lost"]
                and coverage["combined"]["any_interval"] >= coverage["local_uniform"]["any_interval"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", type=Path, default=ROOT / "evidence/local/twinpast-refinement-media")
    parser.add_argument("--out", type=Path, default=ROOT / "evidence/audits/twinpast_boundary_cpu_2026-11-27.json")
    args = parser.parse_args()
    archive = verify_inventory(ARCHIVE)
    manifest = json.loads(MANIFEST.read_bytes())
    assert json.loads(archive["provenance/refinement_cases.json"])["manifest_sha256"] == digest(MANIFEST.read_bytes())
    annotations = {row["pair_id"]: row for row in manifest["candidates"]}
    media = {plan["pair_id"]: receipt["decoded"] for plan, receipt, _ in manifest["media_plan"]}
    rows = []
    for pair in json.loads(archive["exports/run_config.json"])["accepted"]:
        trace = json.loads(archive[f"checkpoints/{pair}.json"])["block"]["branches"]
        for side in "ab":
            branch = side.upper()
            intervals = (annotations[pair]["evidence_intervals"] or {}).get(branch) or []
            row = evaluate(args.media / pair / f"{branch}.mp4", media[pair][side], trace[side], intervals)
            rows.append(dict(pair_id=pair, branch=branch, **row))
    summary = summarize(rows)
    report = dict(refinement_archive_sha256=digest(ARCHIVE.read_bytes()), manifest_sha256=digest(MANIFEST.read_bytes()),
                  threshold=THRESHOLD, fps=FPS,
                  answer_history_frames=6, shared_recent_frames=4, window_budget=3, summary=summary, branches=rows)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
