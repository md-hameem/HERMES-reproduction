"""Pin a first within-benchmark confirmation set without reading predictions."""
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANNOTATION = ROOT / "data/streamingbench/streamingbench_realtime.json"
REVIEW = ROOT / "results/failure_pilot/evidence_review.csv"
OUTPUT = ROOT / "data/streamingbench/short_query_confirmation_manifest.json"
SEED = "HERMES-short-query-confirmation-2026-11-21-v1"
SHARDS = ((51, 100, 8708188924, "6a2b3317008e2f2c4cb30046bf74fedb5c8a75e18b6a042a0949c6032224a934"),
          (101, 150, 8218047488, "6fdc44e97d1888b471b2d7f55dddace16291c04fdfcaab10433b35fb315bb8b3"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sample_id(path):
    match = re.search(r"sample_(\d+)_real\.mp4$", path)
    if not match:
        raise ValueError(f"Unrecognized video path: {path}")
    return int(match[1])


def build():
    source = ANNOTATION.read_bytes()
    records = sorted(json.loads(source), key=lambda row: sample_id(row["video_path"]))
    with REVIEW.open(encoding="utf-8-sig", newline="") as handle:
        reviewed = {int(row["case_id"].split("_")[1]) for row in csv.DictReader(handle)}
    shards = []
    for low, high, size, digest in SHARDS:
        eligible = [v for v in records if low <= sample_id(v["video_path"]) <= high
                    and sample_id(v["video_path"]) not in reviewed]
        ranked = sorted(eligible, key=lambda v: sha(f"{SEED}:{sample_id(v['video_path'])}".encode()))
        selected = sorted(ranked[:15], key=lambda v: sample_id(v["video_path"]))
        if len(selected) != 15:
            raise ValueError("Insufficient eligible clips")
        clips = []
        for video in selected:
            questions = video["conversations"]
            times = [float(q["end_time"]) for q in questions]
            if not times or times != sorted(times) or any(not math.isfinite(t) or t <= 0 for t in times):
                raise ValueError("Selected clip has unusable query times")
            clips.append(dict(sample_id=sample_id(video["video_path"]), video_id=video["video_id"],
                              video_basename=video["video_path"].rsplit("/", 1)[-1], question_count=len(questions),
                              last_query_s=times[-1], tasks=dict(sorted(Counter(q["task"] for q in questions).items()))))
        shards.append(dict(shard=f"{low}-{high}", archive=f"Real-Time Visual Understanding_{low}-{high}.zip",
                           archive_bytes=size, archive_sha256=digest, eligible_clips=len(eligible), clips=clips))
    clips = [c for shard in shards for c in shard["clips"]]
    if len({c["video_id"] for c in clips}) != 30 or len({c["sample_id"] for c in clips}) != 30:
        raise ValueError("Confirmation IDs are not unique")
    return dict(
        status="frozen_clip_ids_source_overlap_unverified",
        scope="Within-benchmark confirmation on previously unused diagnostic clip IDs",
        source_dataset="StreamingBench Real-Time Visual Understanding",
        annotation_path=ANNOTATION.relative_to(ROOT).as_posix(), annotation_sha256=sha(source),
        prior_targeted_review_path=REVIEW.relative_to(ROOT).as_posix(),
        prior_targeted_review_sha256=sha(REVIEW.read_bytes()), seed=SEED,
        selection="15 per shard by ascending SHA256(seed:sample_id), excluding targeted review clips",
        source_group_metadata="Original source-channel and overlap metadata unavailable in local annotation",
        arms=["compact_recent4", "latest_two_short_readout", "all_history_short_readout"],
        primary_accuracy_contrast=["all_history_short_readout", "compact_recent4"],
        settings=dict(sample_fps=0.5, recent_frames=4, total_context_frames=20, latest_history_frames=2,
                      model_revision="74dd0bf867a4cda7950c17663794267c60cf4b40"),
        clips=len(clips), questions=sum(c["question_count"] for c in clips), shards=shards,
    )


if __name__ == "__main__":
    manifest = build()
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Frozen {manifest['clips']} clips and {manifest['questions']} questions: {sha(OUTPUT.read_bytes())}")
