"""Freeze a small metadata-only TwinPast development sample."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
REVISION = "c1f1cbab68bfda2982565c0491f1a97474b73558"
BASE_URL = f"https://huggingface.co/datasets/younght/TwinPast/resolve/{REVISION}"
ANNOTATION_SHA = "f36073ba77565748175741a0e3a11e0bf6536559d6b002ffa43a87016f30542e"
SEED = "HERMES-TwinPast-development-2026-11-23"
OUT = ROOT / "data/twinpast/historical_correction_development.json"


def freeze(data):
    if hashlib.sha256(data).hexdigest() != ANNOTATION_SHA:
        raise ValueError("TwinPast annotation hash differs")
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if len(rows) != 350 or len({row["pair_id"] for row in rows}) != 350:
        raise ValueError("Expected 350 unique TwinPast pairs")
    counts = Counter()
    selected = []
    for task in ("HSR", "AIR"):
        eligible = []
        for row in rows:
            intervals = row["evidence_intervals"]
            if row["task"] != task or not intervals or not all(intervals.get(branch) for branch in ("A", "B")):
                continue
            query = row["t_query"]
            slots = [t for t in range(0, math.floor(query - 8) + 1, 2)
                     if all(any(start <= t <= stop for start, stop in intervals[branch])
                            for branch in ("A", "B"))]
            if len(slots) >= 2:
                eligible.append((row, slots))
        counts[task] = len(eligible)
        ranked = sorted(eligible, key=lambda item: hashlib.sha256(
            f"{SEED}:{item[0]['pair_id']}".encode()).hexdigest())
        if len(ranked) < 6:
            raise ValueError(f"Insufficient annotated pairs in {task}")
        for (row, slots) in ranked[:6]:
            selected.append(dict(pair_id=row["pair_id"], task=task, video_a=row["video_a"], video_b=row["video_b"],
                                 query_time_s=row["t_query"], common_earlier_grid_s=slots,
                                 latest_four_grid_s=[2 * k for k in range(math.ceil(row["t_query"] / 2) - 4,
                                                                         math.ceil(row["t_query"] / 2))]))
    return dict(dataset="younght/TwinPast", revision=REVISION, annotation_sha256=ANNOTATION_SHA,
                seed=SEED, purpose="development_only_exclude_these_pairs_from_confirmation",
                dataset_pairs=len(rows), pairs_with_evidence_intervals=sum(bool(r["evidence_intervals"]) for r in rows),
                eligible_by_task=dict(counts), selected_pairs=selected)


if __name__ == "__main__":
    response = requests.get(f"{BASE_URL}/data.jsonl", timeout=(10, 25))
    response.raise_for_status()
    manifest = freeze(response.content)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Frozen {len(manifest['selected_pairs'])} pairs from {sum(manifest['eligible_by_task'].values())} eligible pairs")
