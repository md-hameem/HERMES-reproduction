import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_elapsed(value):
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"Unsupported elapsed time: {value}")


def outer_video_times(text, video_ids):
    matches = re.findall(r"(?<!\d)(\d{1,2})/50 \[(\d+:\d{2}(?::\d{2})?)<", text)
    elapsed_by_order = {}
    for order, elapsed in matches:
        order = int(order)
        if order == 0:
            continue
        seconds = parse_elapsed(elapsed)
        if order in elapsed_by_order:
            assert elapsed_by_order[order] == seconds
        elapsed_by_order[order] = seconds
    assert sorted(elapsed_by_order) == list(range(1, len(video_ids) + 1))
    cumulative = [elapsed_by_order[i] for i in range(1, len(video_ids) + 1)]
    durations = [cumulative[0]] + [b - a for a, b in zip(cumulative, cumulative[1:])]
    assert all(duration >= 0 for duration in durations)
    return cumulative, durations


def one_match(pattern, segment, label):
    values = re.findall(pattern, segment)
    assert len(values) == 1, f"Expected one {label}, found {len(values)}"
    return values[0]


def extract_run(kv, logs_dir, results_root):
    log_path = logs_dir / f"streamingbench_50_kv{kv}.log"
    result_path = results_root / f"streamingbench_50_kv{kv}" / "1_0.csv"
    text = log_path.read_text(encoding="utf-8", errors="strict")
    results = pd.read_csv(result_path)
    assert len(results) == 250
    results["video_id"] = results.video_id.astype(str)
    results["qa_index"] = results.groupby("video_id", sort=False).cumcount()
    video_ids = list(dict.fromkeys(results.video_id))
    assert len(video_ids) == 50

    answer_markers = list(re.finditer(r"Pred Answer:", text))
    assert len(answer_markers) == len(results)
    rows = []
    start = 0
    for index, marker in enumerate(answer_markers):
        segment = text[start:marker.start()]
        start = marker.end()
        ttft = float(one_match(r"TTFT: ([0-9.]+) seconds", segment, "TTFT"))
        cache_min, cache_max = one_match(
            r"Answering Cache lengths: min=(\d+), max=(\d+)", segment, "cache length"
        )
        chunks = re.findall(r"Encoding frames (\d+) to (\d+)", segment)
        memories = [float(value) for value in re.findall(r"GPU memory usage: ([0-9.]+) GB", segment)]
        assert len(memories) == len(chunks)
        encoded_frames = sum(int(end) - int(begin) + 1 for begin, end in chunks)
        source = results.iloc[index]
        rows.append({
            "kv_budget": kv,
            "video_id": source.video_id,
            "qa_index": int(source.qa_index),
            "task": source.task,
            "qa_acc": float(source.qa_acc),
            "ttft_seconds": ttft,
            "answering_cache_min": int(cache_min),
            "answering_cache_max": int(cache_max),
            "encoded_chunks": len(chunks),
            "encoded_frames": encoded_frames,
            "compression_events": segment.count("Applying KV-Cache compression"),
            "reported_peak_allocated_gb_max": max(memories) if memories else np.nan,
            "memory_readings": len(memories),
        })
    questions = pd.DataFrame(rows)
    cumulative, durations = outer_video_times(text, video_ids)
    video_time = pd.DataFrame({
        "video_id": video_ids,
        "video_order": range(1, len(video_ids) + 1),
        "outer_elapsed_seconds_tqdm": cumulative,
        "video_processing_seconds_tqdm": durations,
    })
    questions = questions.merge(video_time, on="video_id", validate="many_to_one")
    videos = questions.groupby(["kv_budget", "video_id"], sort=False).agg(
        questions=("qa_index", "size"),
        encoded_chunks=("encoded_chunks", "sum"),
        encoded_frames=("encoded_frames", "sum"),
        compression_events=("compression_events", "sum"),
        questions_with_compression=("compression_events", lambda values: int((values > 0).sum())),
        ttft_median_seconds=("ttft_seconds", "median"),
        ttft_mean_seconds=("ttft_seconds", "mean"),
        ttft_max_seconds=("ttft_seconds", "max"),
        answering_cache_min=("answering_cache_max", "min"),
        answering_cache_median=("answering_cache_max", "median"),
        answering_cache_max=("answering_cache_max", "max"),
        reported_peak_allocated_gb_max=("reported_peak_allocated_gb_max", "max"),
        video_order=("video_order", "first"),
        outer_elapsed_seconds_tqdm=("outer_elapsed_seconds_tqdm", "first"),
        video_processing_seconds_tqdm=("video_processing_seconds_tqdm", "first"),
    ).reset_index()
    return log_path, result_path, questions, videos


def build_summary(kv, questions, videos, run_summary):
    runtime = run_summary.loc[run_summary.kv_budget.eq(kv)]
    assert len(runtime) == 1
    runtime = runtime.iloc[0]
    outer_seconds = int(videos.outer_elapsed_seconds_tqdm.max())
    total_seconds = float(runtime.elapsed_min_this_session) * 60
    return {
        "kv_budget": kv,
        "videos": int(questions.video_id.nunique()),
        "questions": len(questions),
        "correct": int(questions.qa_acc.sum() / 100),
        "accuracy_percent": questions.qa_acc.mean(),
        "encoded_chunks": int(questions.encoded_chunks.sum()),
        "encoded_frames": int(questions.encoded_frames.sum()),
        "compression_events": int(questions.compression_events.sum()),
        "questions_with_compression": int((questions.compression_events > 0).sum()),
        "max_reported_peak_allocated_gb": questions.reported_peak_allocated_gb_max.max(),
        "reserved_memory": "not_logged",
        "ttft_count": int(questions.ttft_seconds.notna().sum()),
        "ttft_min_seconds": questions.ttft_seconds.min(),
        "ttft_median_seconds": questions.ttft_seconds.median(),
        "ttft_mean_seconds": questions.ttft_seconds.mean(),
        "ttft_p95_seconds": np.percentile(questions.ttft_seconds, 95),
        "ttft_max_seconds": questions.ttft_seconds.max(),
        "answering_cache_min": int(questions.answering_cache_max.min()),
        "answering_cache_median": questions.answering_cache_max.median(),
        "answering_cache_max": int(questions.answering_cache_max.max()),
        "outer_elapsed_seconds_tqdm": outer_seconds,
        "outer_elapsed_minutes_tqdm": outer_seconds / 60,
        "total_runtime_minutes_notebook": float(runtime.elapsed_min_this_session),
        "runtime_minus_outer_seconds": total_seconds - outer_seconds,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract efficiency metrics from preserved HERMES AutoDL logs")
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--results-root", type=Path, default=Path("HERMES/results"))
    parser.add_argument("--run-summary", type=Path, default=Path("exports/run_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("exports/efficiency"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[4000, 6000, 50000])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_summary = pd.read_csv(args.run_summary)
    question_frames = []
    video_frames = []
    summaries = []
    source_paths = [args.run_summary, Path(__file__)]
    workloads = []
    for kv in args.budgets:
        log_path, result_path, questions, videos = extract_run(kv, args.logs_dir, args.results_root)
        source_paths.extend([log_path, result_path])
        question_frames.append(questions)
        video_frames.append(videos)
        summaries.append(build_summary(kv, questions, videos, run_summary))
        workloads.append(questions[["video_id", "qa_index", "encoded_chunks", "encoded_frames"]])

    reference = workloads[0]
    for other in workloads[1:]:
        pd.testing.assert_frame_equal(reference, other, check_dtype=False)

    question_path = args.output_dir / "autodl_50_question_metrics.csv"
    video_path = args.output_dir / "autodl_50_video_metrics.csv"
    summary_path = args.output_dir / "autodl_50_efficiency_summary.csv"
    provenance_path = args.output_dir / "autodl_50_efficiency_provenance.json"
    pd.concat(question_frames, ignore_index=True).to_csv(question_path, index=False)
    pd.concat(video_frames, ignore_index=True).to_csv(video_path, index=False)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "AutoDL RTX 4080 SUPER; StreamingBench samples 201-250; HERMES KV 4000/6000/50000",
        "source_sha256": {str(path): sha256(path) for path in source_paths},
        "output_sha256": {
            str(question_path): sha256(question_path),
            str(video_path): sha256(video_path),
            str(summary_path): sha256(summary_path),
        },
        "metric_definitions": {
            "ttft_seconds": "Existing LLaVA question_answering timer. It excludes video ingestion and pseudo-query compression and ends after the first selected token is passed through another decoder forward.",
            "reported_peak_allocated_gb": "Existing torch.cuda.max_memory_allocated readings divided by 1024^3. The counter is process-lifetime and is not reset per video or question.",
            "reserved_memory": "Unavailable: the logs contain no torch.cuda.max_memory_reserved measurements.",
            "answering_cache": "Minimum/maximum layer KV length after temporary question-answer cache entries are truncated. Summary statistics use the maximum.",
            "compression_events": "Count of 'Applying KV-Cache compression' messages.",
            "video_processing_seconds_tqdm": "Difference in coarse whole-second outer tqdm cumulative times; includes per-video loading/processing but excludes process and model startup.",
            "total_runtime_minutes_notebook": "Notebook wall time around each inference subprocess from exports/run_summary.csv.",
        },
        "limitations": [
            "Derived from existing logs; no new GPU measurements were collected.",
            "Reserved memory and isolated encoding/compression durations are unavailable.",
            "TTFT is not end-to-end question latency or streaming throughput.",
            "Cross-GPU runtime comparisons are not hardware-independent.",
        ],
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"Wrote {summary_path}, {question_path}, {video_path}, and {provenance_path}")


if __name__ == "__main__":
    main()
