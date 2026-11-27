"""Recompute the frozen DROID interior coverage criterion from visual review."""
import argparse
from collections import Counter
import json
from pathlib import Path

from droid_span_selector import WINDOWS
from validate_twinpast_diagnostic_evidence import ROOT, digest

REVIEW = ROOT / "data/twinpast/droid_interior_review_v1.json"
INTAKE = ROOT / "data/twinpast/droid_interior_intake_v1.json"
OUTPUT = ROOT / "evidence/audits/droid_interior_cpu_2026-11-28.json"
FIXED = {"first": (0, 1, 2), "center": (4, 5, 6), "last": (7, 8, 9)}


def check(intake_file, review_file):
    intake = json.loads(intake_file.read_bytes())
    candidates = {row["episode_id"]: row for row in intake["intake"]}
    observed = json.loads(review_file.read_bytes())
    assert len(observed) == len(candidates) and {r["episode_id"] for r in observed} == set(candidates)
    windows_covered = dict.fromkeys(WINDOWS, 0)
    seen_scenes, labels, cases = set(), Counter(), []
    previous_scenes = set(map(tuple, intake["excluded_scenes"]))
    for review in observed:
        uid = review["episode_id"]
        source = candidates[uid]
        scene = (source["site"], source["scene_id"])
        assert uid not in intake["excluded_episodes"] and scene not in previous_scenes | seen_scenes
        seen_scenes.add(scene)
        if "camera" not in review:
            assert review["reason"].strip()
            continue
        camera = review["camera"]
        slots = source["grids"][camera]
        video = source["cameras"][camera]
        assert slots == [i*(video["frame_count"]-1)//9 for i in range(10)]
        support = set(review["support_slots"])
        opposite = set(review["opposite_slots"])
        assert support and min(support) > 0 and max(support) < 9
        assert opposite and not support.intersection(opposite) and opposite.issubset(range(10))
        assert len(support) == len(review["support_slots"]) and len(opposite) == len(review["opposite_slots"])
        assert review["question"].strip() and review["event_note"].strip()
        assert review["anchor_frame"] in {slots[i] for i in support}
        label = review["state"]
        assert label in ("open", "closed")
        labels[label] += 1
        hits = {w: len(support.intersection(w)) for w in WINDOWS}
        for window, count in hits.items():
            windows_covered[window] += count > 0
        selected = max(WINDOWS, key=hits.__getitem__)
        cases.append(dict(**review, site=source["site"], scene_id=scene[1], frame_grid=slots, video_source=video,
                          metadata_source=source["metadata_source"], annotated_window=selected,
                          support_counts={**{n: hits[w] for n, w in FIXED.items()}, "annotated": hits[selected]}))
    covered = {n: sum(c["support_counts"][n] > 0 for c in cases) for n in (*FIXED, "annotated")}
    best_count = max(windows_covered.values())
    headroom = covered["annotated"] - best_count
    sites = sorted({r["site"] for r in cases})
    gate = dict(at_least_six_scenes=len(cases) >= 6, at_least_three_sites=len(sites) >= 3,
                at_least_two_per_state=min(labels["open"], labels["closed"]) >= 2,
                at_least_two_more_than_static=headroom >= 2)
    return dict(intake_sha256=digest(intake_file.read_bytes()), review_sha256=digest(review_file.read_bytes()),
                index_candidates=intake["index_candidates"], metadata_requests=intake["metadata_requests"],
                unavailable=intake["unavailable"], previewed=len(candidates), eligible=len(cases), sites=sites,
                labels={name: labels[name] for name in ("open", "closed")}, covered=covered,
                all_static_windows=[dict(slots=w, covered=count) for w, count in windows_covered.items()],
                best_static_coverage=best_count, extra_coverage_over_best_static=headroom,
                both_endpoint_states_opposite=sum({0, 9}.issubset(c["opposite_slots"]) for c in cases),
                prior_episode_ids=len(intake["excluded_episodes"]), prior_scene_ids=len(previous_scenes),
                gate=gate, gpu_feasibility_criterion_met=all(gate.values()), cases=cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", type=Path, default=INTAKE)
    parser.add_argument("--review", type=Path, default=REVIEW)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = check(args.intake, args.review)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: report[k] for k in ("previewed", "eligible", "sites", "labels", "covered", "gate")}, indent=2))
