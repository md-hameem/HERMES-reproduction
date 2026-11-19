"""Build a CPU-only, outcome-selected diagnostic casebook from audited predictions.

No video evidence or GPU inference is available to this script. Unobserved failure
causes remain unclassified. Selected videos retain their complete QA history.
"""

import argparse
import ast
import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
QUOTAS = {"hermes_only": 8, "fifo_only": 8, "both_wrong": 10, "both_correct": 4}
REVIEW_FIELDS = [
    "review_status", "reviewer", "support_intervals_seconds_json",
    "support_visible_at_original_sampling", "support_available_before_query",
    "requires_audio", "gold_ambiguity", "evidence_type", "failure_stage",
    "oracle_support_qa_result", "retention_trace_result", "review_notes",
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_predictions(data, annotations):
    """Join by explicit QA identity, validating text, choices, gold and correctness."""
    expected = {(str(v["video_id"]), i): q for v in annotations
                for i, q in enumerate(v["conversations"])}
    result = {}
    for row in csv.DictReader(io.StringIO(data.decode("utf-8-sig"))):
        key = (row["video_id"], int(row["qa_index"]))
        if key in result or key not in expected:
            raise ValueError(f"Duplicate or unexpected prediction key {key}")
        q = expected[key]
        choices = q["choices"]
        gold = "ABCDEFGH"[choices.index(q["answer"])]
        text = row["pred_answer"].strip()
        parsed = text[text.index(")") - 1:text.index(")")] if ")" in text else text[:1]
        correct = int(row["pred_choice"].strip().upper() == gold)
        if not (row["question"].strip() == q["question"].strip()
                and ast.literal_eval(row["choices"]) == choices
                and row["answer"] == str(q["answer"])
                and row["correct_choice"] == gold
                and parsed == row["pred_choice"]
                and float(row["qa_acc"]) == correct * 100
                and int(row["correct"]) == correct
                and row["task"] == q["task"]):
            raise ValueError(f"Prediction/annotation disagreement at {key}")
        result[key] = {**row, "correct": correct}
    if set(result) != set(expected):
        raise ValueError("Prediction keys do not cover annotations exactly")
    return result


def build_rows(annotations, hermes, fifo):
    rows = []
    seen = set()
    for video in annotations:
        vid = str(video["video_id"])
        if vid in seen:
            raise ValueError(f"Duplicate video ID {vid}")
        seen.add(vid)
        sample = int(re.search(r"sample_(\d+)", video["video_path"])[1])
        times = [float(q["end_time"]) for q in video["conversations"]]
        if times != sorted(times):
            raise ValueError(f"Noncausal question chronology for {vid}")
        for i, q in enumerate(video["conversations"]):
            h, f = hermes[(vid, i)], fifo[(vid, i)]
            outcome = ("both_correct" if h["correct"] and f["correct"] else
                       "hermes_only" if h["correct"] else
                       "fifo_only" if f["correct"] else "both_wrong")
            prev_h = hermes.get((vid, i - 1))
            prev_f = fifo.get((vid, i - 1))
            rows.append(dict(
                case_id=f"sample_{sample:03d}_q{i}", sample_id=sample, video_id=vid,
                qa_index=i, video_path=video["video_path"], task=q["task"],
                required_ability=q.get("required_ability", "unknown"),
                query_time_s=q["end_time"], question=q["question"],
                choices_json=json.dumps(q["choices"], ensure_ascii=False),
                gold_answer=q["answer"], gold_choice=h["correct_choice"],
                duplicate_gold_options=q["choices"].count(q["answer"]) > 1,
                hermes_answer=h["pred_answer"], fifo_answer=f["pred_answer"],
                hermes_choice=h["pred_choice"], fifo_choice=f["pred_choice"],
                hermes_correct=h["correct"], fifo_correct=f["correct"],
                outcome=outcome, choices_differ=h["pred_choice"] != f["pred_choice"],
                previous_query_time_s=times[i - 1] if i else "",
                new_time_since_previous_query_s=times[i] - times[i - 1] if i else "",
                previous_hermes_choice=prev_h["pred_choice"] if prev_h else "",
                previous_fifo_choice=prev_f["pred_choice"] if prev_f else "",
                previous_hermes_correct=prev_h["correct"] if prev_h else "",
                previous_fifo_correct=prev_f["correct"] if prev_f else "",
                failure_cause="unclassified",
            ))
    return sorted(rows, key=lambda r: (r["sample_id"], r["qa_index"]))


def select_cases(rows, seed=2024, quotas=None):
    """Task round-robin, hash-ranked within task, one case per source video.

    Outcomes are deliberately selected. This sample cannot estimate accuracy.
    Exclude ambiguous duplicate gold options from the pilot, retaining them in
    the complete paired inventory for audit.
    """
    selected, used_videos = [], set()
    for outcome, count in (quotas or QUOTAS).items():
        buckets = defaultdict(list)
        for row in rows:
            if row["outcome"] == outcome and not row["duplicate_gold_options"]:
                buckets[row["task"]].append(row)
        for task in buckets:
            buckets[task].sort(key=lambda r: sha(f"{seed}:{r['case_id']}".encode()))
        tasks = sorted(buckets, key=lambda t: sha(f"{seed}:{outcome}:{t}".encode()))
        chosen = []
        while len(chosen) < count:
            progress = False
            for task in tasks:
                while buckets[task] and buckets[task][0]["video_id"] in used_videos:
                    buckets[task].pop(0)
                if buckets[task]:
                    row = buckets[task].pop(0)
                    chosen.append(row)
                    used_videos.add(row["video_id"])
                    progress = True
                    if len(chosen) == count:
                        break
            if not progress:
                raise ValueError(f"Insufficient distinct videos for {outcome}: {len(chosen)}/{count}")
        selected.extend(chosen)
    return selected


def write_csv(path, rows, fields=None):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "evidence/archives/HERMES_AutoDL_five_stage_evidence.zip")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/streamingbench/streamingbench_realtime.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/failure_pilot")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    annotation_bytes = args.annotations.read_bytes()
    annotations = json.loads(annotation_bytes)
    source_hashes = {}
    predictions = {}
    with ZipFile(args.archive) as archive:
        manifest = {r["path"]: r for r in csv.DictReader(io.StringIO(
            archive.read("exports/five_stage_manifest_sha256.csv").decode("utf-8-sig")))}
        for method in ("hermes", "fifo"):
            name = f"exports/full_{method}_4000_predictions.csv"
            data = archive.read(name)
            if sha(data) != manifest[name]["sha256"]:
                raise ValueError(f"Archive prediction hash mismatch for {name}")
            source_hashes[name] = sha(data)
            predictions[method] = load_predictions(data, annotations)
    rows = build_rows(annotations, predictions["hermes"], predictions["fifo"])
    if len(rows) != 2495 or len(annotations) != 498:
        raise ValueError("Expected audited full 498-video / 2495-question inventory")
    selected = select_cases(rows, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    # Never replace an existing human annotation file or change its case identity.
    review_path = args.output / "evidence_review.csv"
    if review_path.exists():
        with review_path.open(encoding="utf-8-sig", newline="") as handle:
            existing = list(csv.DictReader(handle))
        if [r["case_id"] for r in existing] != [r["case_id"] for r in selected]:
            raise ValueError("Existing review uses different cases; use a new output directory")
    write_csv(args.output / "paired_inventory.csv", rows)
    write_csv(args.output / "selected_cases.csv", selected)
    if not review_path.exists():
        reviews = [dict(case_id=r["case_id"], **{f: "pending" if f == "review_status" else ""
                                               for f in REVIEW_FIELDS}) for r in selected]
        write_csv(review_path, reviews)
    task_rows = []
    for task in sorted({r["task"] for r in rows}):
        group = [r for r in rows if r["task"] == task]
        task_rows.append(dict(task=task, questions=len(group),
                              hermes_correct=sum(r["hermes_correct"] for r in group),
                              fifo_correct=sum(r["fifo_correct"] for r in group),
                              **{o: sum(r["outcome"] == o for r in group) for o in QUOTAS}))
    write_csv(args.output / "task_outcomes.csv", task_rows)
    selected_ids = {r["video_id"] for r in selected}
    videos = sorted((v for v in annotations if str(v["video_id"]) in selected_ids),
                    key=lambda v: int(re.search(r"sample_(\d+)", v["video_path"])[1]))
    (args.output / "selected_videos.json").write_text(
        json.dumps(videos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    provenance = dict(
        schema_version=1, seed=args.seed, quotas=QUOTAS,
        selection="Outcome-selected task round-robin with SHA256 ranking and globally distinct videos; duplicate gold excluded",
        annotation_sha256=sha(annotation_bytes), archive_sha256=sha(args.archive.read_bytes()),
        prediction_sha256=source_hashes, builder_sha256=sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
        inventory_questions=len(rows), inventory_videos=len(annotations),
        outcome_counts=dict(Counter(r["outcome"] for r in rows)),
        selected_questions=len(selected), selected_videos=len(videos),
        selected_video_full_history_questions=sum(len(v["conversations"]) for v in videos),
        selected_case_ids=[r["case_id"] for r in selected],
        limits=["Development reuse; selected outcomes cannot estimate accuracy or confirm a method",
                "No video inspection, evidence survival measurement or GPU inference performed",
                "Question time is not evidence age; previous wrong answers do not prove feedback harm",
                "Replay complete selected video histories; do not run selected questions in isolation"],
    )
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: provenance[k] for k in (
        "inventory_questions", "outcome_counts", "selected_questions", "selected_videos",
        "selected_video_full_history_questions")}, indent=2))


if __name__ == "__main__":
    main()
