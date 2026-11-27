"""Recompute the matched DROID event screen using its published receipts."""
import json
from pathlib import Path

from validate_twinpast_diagnostic_evidence import ROOT, digest

INTAKE = ROOT / "data/twinpast/droid_matched_event_intake_v1.json"
REVIEW = ROOT / "data/twinpast/droid_matched_event_review_v1.json"
OUTPUT = ROOT / "evidence/audits/droid_matched_event_cpu_2026-11-28.json"


def audit(intake_file=INTAKE, review_file=REVIEW):
    data = json.loads(intake_file.read_bytes())
    reviews = json.loads(review_file.read_bytes())
    rlds = data["rlds"]
    raw = data["raw"]
    excluded = set(raw["broad"]["excluded_episodes"])
    assert raw["strict"]["excluded_episodes"] == raw["broad"]["excluded_episodes"]
    assert raw["strict"]["excluded_scenes"] == raw["broad"]["excluded_scenes"]
    assert len(reviews) == len(rlds["pairs"]) == raw["broad"]["ranked_pairs"]
    seen, eligible, scenes = set(), 0, set()
    visual_count, qualified_visual = 0, 0
    for pair, review in zip(rlds["pairs"], reviews, strict=True):
        ids = tuple(r["episode_id"] for r in pair["episodes"])
        assert tuple(review["episode_ids"]) == ids and ids not in seen and not excluded.intersection(ids)
        seen.add(ids)
        available = all(uid in rlds["receipts"] for uid in ids)
        assert review["reviewed"] == available and review["reason"].strip()
        visual_count += available
        if not available:
            assert review["status"] == "unavailable"
            continue
        assert review["status"] in {"visual_reject", "provenance_unverified", "eligible"}
        qualified_visual += review["status"] in {"provenance_unverified", "eligible"}
        if review["status"] == "eligible":
            scene = tuple(review["verified_scene"])
            assert scene not in map(tuple, raw["broad"]["excluded_scenes"])
            assert any(tuple(row["scene"]) == scene
                       and set(ids) == {e["episode_id"] for e in row["episodes"]}
                       for row in raw["broad"]["pairs"])
            eligible += 1
            scenes.add(scene)
    attempted = {r["episode_id"] for p in rlds["pairs"] for r in p["episodes"]}
    missing = {r["episode"] for r in rlds["failures"]}
    assert set(rlds["receipts"]).isdisjoint(missing)
    assert set(rlds["receipts"]) | missing == attempted
    record_hashes = set()
    for receipt in rlds["receipts"].values():
        assert receipt["byte_offset"] >= 0 and receipt["bytes"] > 0
        assert receipt["sha256"] not in record_hashes
        record_hashes.add(receipt["sha256"])
        assert len(receipt["sampled_indices"]) == len(set(receipt["sampled_indices"])) == 10
    return dict(intake_sha256=digest(intake_file.read_bytes()), review_sha256=digest(review_file.read_bytes()),
                raw_strict_nominations=raw["strict"]["ranked_pairs"],
                raw_broad_nominations=raw["broad"]["ranked_pairs"],
                raw_pairs_acquired=len(raw["broad"]["pairs"]),
                excluded_prior_episodes=len(excluded), excluded_prior_scenes=len(raw["broad"]["excluded_scenes"]),
                rlds_records_attempted=len(attempted), rlds_records_recovered=len(rlds["receipts"]),
                rlds_pairs_reviewed=visual_count, visually_qualified_pairs=qualified_visual,
                eligible_pairs=eligible, verified_new_scenes=len(scenes),
                gpu_gate_met=eligible >= 4 and len(scenes) >= 3)


if __name__ == "__main__":
    result = json.dumps(audit(), indent=2) + "\n"
    OUTPUT.write_text(result, encoding="utf-8", newline="\n")
    print(result, end="")
