"""Freeze a new within-StreamingBench evaluation cohort using metadata only."""
import csv
import json
import math

from make_confirmation_manifest import ANNOTATION, OUTPUT as CONFIRMATION, REVIEW, ROOT, SHARDS, normalized_sha, sample_id, sha

OUTPUT = ANNOTATION.with_name("target_binding_manifest.json")
SEED = "HERMES-target-binding-2026-11-21-v1"
TEXT_TASK = "Text-Rich Understanding"
SHARD_PLAN = (
    (*SHARDS[1], 12, 6),
    (151, 200, 17984397733, "ce4c47e4161a5502f25e9fe9b7ee3929aa839b6916777e3614a81fa31def4472", 12, 3),
    (201, 250, 2230277873, "b7946e8b678fbf704b9816fa708e6febb77f5c37a7ad884084adf858f5d20dbf", 24, 0),
    (251, 300, 18838513104, "776f78a9338aaa72ed2e8c99a10eeced813c8499988bda29f4205c542a8a577a", 12, 4),
)


def eligible(video):
    questions = video["conversations"]
    times = [float(q["end_time"]) for q in questions]
    return (bool(times) and times == sorted(times) and all(math.isfinite(t) and t > 0 for t in times)
            and all(q["question"].strip() and 2 <= len(q["choices"]) <= 8
                    and q["choices"].count(q["answer"]) == 1 for q in questions))


def build():
    previous = json.loads(CONFIRMATION.read_bytes())
    excluded = {c["sample_id"] for shard in previous["shards"] for c in shard["clips"]}
    with REVIEW.open(encoding="utf-8-sig", newline="") as handle:
        excluded.update(int(row["case_id"].split("_")[1]) for row in csv.DictReader(handle))
    records = {sample_id(v["video_path"]): v for v in json.loads(ANNOTATION.read_bytes())}
    shards = []
    for low, high, size, digest, count, quota in SHARD_PLAN:
        pool = [sid for sid in range(low, high + 1)
                if sid in records and sid not in excluded and eligible(records[sid])]
        pool.sort(key=lambda sid: sha(f"{SEED}:{sid}".encode()))
        text = [sid for sid in pool if any(q["task"] == TEXT_TASK for q in records[sid]["conversations"])]
        chosen = text[:quota]
        chosen += [sid for sid in pool if sid not in chosen][:count - quota]
        if len(chosen) != count:
            raise ValueError(f"Insufficient eligible samples for shard {low}-{high}")
        clips = []
        for sid in sorted(chosen):
            video = records[sid]
            questions = video["conversations"]
            clips.append(dict(sample_id=sid, video_id=video["video_id"],
                              video_basename=video["video_path"].rsplit("/", 1)[-1],
                              question_count=len(questions),
                              primary_count=sum(math.ceil(float(q["end_time"]) * .5) >= 8 for q in questions),
                              text_rich_count=sum(q["task"] == TEXT_TASK for q in questions)))
        shards.append(dict(shard=f"{low}-{high}", archive=f"Real-Time Visual Understanding_{low}-{high}.zip",
                           archive_bytes=size, archive_sha256=digest, clips=clips))
    clips = [clip for shard in shards for clip in shard["clips"]]
    if len({clip["video_id"] for clip in clips}) != len(clips):
        raise ValueError("Duplicate benchmark video IDs")
    return dict(annotation_path=ANNOTATION.relative_to(ROOT).as_posix(), annotation_sha256=normalized_sha(ANNOTATION),
                seed=SEED, excluded_diagnostic_sample_ids=sorted(excluded),
                selection="Metadata-only SHA256 rank, reserved text-rich clips by shard, then rank-fill",
                scope="New diagnostic clip IDs within StreamingBench; original source overlap unverified",
                clips=len(clips), questions=sum(c["question_count"] for c in clips),
                primary_questions=sum(c["primary_count"] for c in clips),
                text_rich_questions=sum(c["text_rich_count"] for c in clips), shards=shards)


if __name__ == "__main__":
    manifest = build()
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print({k: manifest[k] for k in ("clips", "questions", "primary_questions", "text_rich_questions")})
