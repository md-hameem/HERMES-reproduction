"""Print a CPU-only paired analysis of the audited StreamingBench 4K runs."""

import csv
import json
from collections import Counter, defaultdict
from math import ceil, comb
from pathlib import PurePosixPath
from zipfile import ZipFile

from build_failure_pilot import ROOT, build_rows, load_predictions


def load_pairs():
    path = ROOT / "data/streamingbench/streamingbench_realtime.json"
    annotations = json.loads(path.read_text(encoding="utf-8"))
    with ZipFile(ROOT / "evidence/archives/HERMES_AutoDL_five_stage_evidence.zip") as archive:
        hermes = load_predictions(archive.read("exports/full_hermes_4000_predictions.csv"), annotations)
        fifo = load_predictions(archive.read("exports/full_fifo_4000_predictions.csv"), annotations)

    rows = build_rows(annotations, hermes, fifo)
    totals = (
        len(rows), len(annotations), sum(row["hermes_correct"] for row in rows),
        sum(row["fifo_correct"] for row in rows),
    )
    if totals != (2495, 498, 1555, 1546):
        raise ValueError(f"Joined predictions do not reproduce the audited totals: {totals}")
    samples = {str(video["video_id"]): PurePosixPath(video["video_path"]).stem for video in annotations}
    durations = {}
    path = ROOT / "results/kaggle/streamingbench_50_video_statistics.csv"
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            video_id = row["video_id"]
            if video_id in durations or samples.get(video_id) != f'{row["sample"]}_real':
                raise ValueError(f"Duplicate or mismatched duration record for video {video_id}")
            durations[video_id] = float(row["duration_sec"])
    if len(durations) != 50 or sum(row["video_id"] in durations for row in rows) != 250:
        raise ValueError("Duration coverage differs from the audited 50 videos and 250 questions")
    return rows, durations


def nearest_rank(values, fraction):
    ordered = sorted(values)
    return ordered[ceil(len(ordered) * fraction) - 1]


def exact_mcnemar(hermes_only, fifo_only):
    discordant = hermes_only + fifo_only
    lower_tail = sum(comb(discordant, count) for count in range(min(hermes_only, fifo_only) + 1))
    return min(1.0, 2 * lower_tail / (2 ** discordant))


def print_breakdown(title, groups):
    print(f"\n### {title}\n")
    print("| Group | Questions | Videos | HERMES | FIFO | Delta (pp) | HERMES only | FIFO only | Exact p |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, rows in groups:
        counts = Counter(row["outcome"] for row in rows)
        questions = len(rows)
        videos = len({row["video_id"] for row in rows})
        both = counts["both_correct"]
        hermes_only, fifo_only = counts["hermes_only"], counts["fifo_only"]
        hermes_accuracy = (both + hermes_only) / questions
        fifo_accuracy = (both + fifo_only) / questions
        difference = 100 * (hermes_only - fifo_only) / questions
        p = exact_mcnemar(hermes_only, fifo_only)
        print(f"| {label} | {questions:,} | {videos} | {hermes_accuracy:.2%} | {fifo_accuracy:.2%} | "
              f"{difference:+.2f} | {hermes_only} | {fifo_only} | {p:.4f} |")


def tertile_groups(rows, value, lower, upper):
    return [
        (f"<= {lower:.6g} s", [row for row in rows if value(row) <= lower]),
        (f"{lower:.6g} < t <= {upper:.6g} s", [row for row in rows if lower < value(row) <= upper]),
        (f"> {upper:.6g} s", [row for row in rows if value(row) > upper]),
    ]


def main():
    rows, durations = load_pairs()
    outcomes = Counter(row["outcome"] for row in rows)
    both, neither = outcomes["both_correct"], outcomes["both_wrong"]
    hermes_only, fifo_only = outcomes["hermes_only"], outcomes["fifo_only"]
    if (both + neither + hermes_only + fifo_only, both + hermes_only, both + fifo_only) != (2495, 1555, 1546):
        raise ValueError("The paired outcome matrix does not reproduce the audited totals")

    print("Integrity PASS - 2,495 joined questions, 498 videos, 1,555 HERMES correct, 1,546 FIFO correct.")
    print("\n### Overall paired outcomes\n")
    print("| | FIFO right | FIFO wrong |\n|---|---:|---:|")
    print(f"| HERMES right | {both:,} | {hermes_only:,} |")
    print(f"| HERMES wrong | {fifo_only:,} | {neither:,} |")
    print_breakdown("Overall accuracy", [("All questions", rows)])

    for field, title in (("task", "Task category"), ("required_ability", "Required ability")):
        groups = defaultdict(list)
        for row in rows:
            label = row[field] or "Unspecified"
            groups[label].append(row)
        print_breakdown(title, sorted(groups.items()))

    questions_by_video = Counter(row["video_id"] for row in rows)
    early, late = [], []
    for row in rows:
        early_count = (questions_by_video[row["video_id"]] + 1) // 2
        group = early if row["qa_index"] < early_count else late
        group.append(row)
    print_breakdown("Question index within video", [("Early (first half rounded up)", early), ("Late", late)])

    def query_time(row):
        return row["query_time_s"]

    times = [query_time(row) for row in rows]
    print_breakdown("Question timestamp tertiles", tertile_groups(
        rows, query_time, nearest_rank(times, 1 / 3), nearest_rank(times, 2 / 3),
    ))
    p90 = nearest_rank(times, 0.9)
    print_breakdown("High question timestamp", [
        (f"Top decile incl. ties (>= {p90:g} s)", [row for row in rows if query_time(row) >= p90]),
        (f"Remaining (< {p90:g} s)", [row for row in rows if query_time(row) < p90]),
    ])

    def duration(row):
        return durations[row["video_id"]]

    covered_rows = [row for row in rows if row["video_id"] in durations]
    duration_values = durations.values()
    print_breakdown("Exploratory video duration tertiles (50 videos, 250 questions)", tertile_groups(
        covered_rows, duration, nearest_rank(duration_values, 1 / 3), nearest_rank(duration_values, 2 / 3),
    ))


if __name__ == "__main__":
    main()
