"""Replay the verifier and prepare a blinded visual-support review from saved outputs."""
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from validate_causal_pilot_evidence import check_manifest
from validate_target_binding_evidence import ARCHIVE, ROOT

SEED = "target-binding-support-review-v1"
OUT = ROOT / "results/target_binding_2026-11-22"
KEY = ROOT / "evidence/local/target_binding_2026-11-22/review_key.csv"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare():
    with ZipFile(ARCHIVE) as archive:
        _, payload = check_manifest(archive)
    blocks = [json.loads(value)["block"] for name, value in payload.items() if name.startswith("checkpoints/")]
    blocks = [block for block in blocks if block["rows"][0]["eligible"]]
    strata = defaultdict(list)
    min_equals_binding = equal_choices = switches = wins = losses = 0
    sign_flips = dict(binding=0, relevance=0, target=0)
    count = 0
    for block in blocks:
        row = block["rows"][0]
        sid, qi = row["sample_id"], row["qa_index"]
        windows = block["windows"]
        count += len(windows)
        best = max(windows, key=lambda w: w["scores"]["binding"])
        equal_choices += best["window"] == next(r["window"] for r in block["rows"] if r["method"] == "conjunction")
        recent = windows[0]
        older = max(windows[1:], key=lambda w: w["scores"]["binding"])
        if recent["scores"]["binding"] < 0 <= older["scores"]["binding"]:
            switches += 1
            wins += bool(older["correct"] and not recent["correct"])
            losses += bool(recent["correct"] and not older["correct"])
        for w in windows:
            min_equals_binding += w["scores"]["binding"] <= w["scores"]["target"]
            for name, (a, b) in w["order_margins"].items():
                sign_flips[name] += a * b < 0
        key = f"{SEED}:{sid}:{qi}".encode()
        strata[(bool(recent["correct"]), any(w["correct"] for w in windows[1:]))].append(
            (hashlib.sha256(key).hexdigest(), block))

    # Fixed per-stratum samples for mechanism discovery. Gold, stratum and correctness
    # stay in the key. Frame timestamps remain visible for temporal validity judgments.
    rows, keys = [], []
    selected = sorted(item for stratum in strata.values() for item in sorted(stratum)[:8])
    for rank, (digest, block) in enumerate(selected, start=1):
        row = block["rows"][0]
        sid, qi = row["sample_id"], row["qa_index"]
        times = json.loads(payload[f"provenance/features_{sid}.json"])["timestamps"]
        windows = list(block["windows"])
        random.Random(digest).shuffle(windows)
        for letter, w in zip("ABCDE", windows):
            indices = w["indices"]
            rid = f"Q{rank:02d}-{letter}"
            rows.append(dict(review_id=rid, sample_id=sid, question_index_0_based=qi,
                             question=row["question"], options=json.dumps(row["choices"], ensure_ascii=False),
                             query_time_s=row["query_time_s"], sampled_indices=json.dumps(indices),
                             source_frame_indices=json.dumps([times[i]["source_frame"] for i in indices]),
                             frame_start_s=json.dumps([times[i]["start_s"] for i in indices]),
                             support_label="", supported_option="", notes=""))
            keys.append(dict(review_id=rid, window=w["window"], model_choice=w["pred_choice"],
                             gold_choice=row["gold_choice"], model_correct=w["correct"],
                             recent_correct=block["windows"][0]["correct"],
                             any_older_correct=int(any(c["correct"] for c in block["windows"][1:]))))
    write_csv(OUT / "visual_support_review.csv", rows)
    write_csv(KEY, keys)
    summary = dict(eligible_questions=len(blocks), eligible_windows=count, minimum_equals_binding=min_equals_binding,
                   binding_only_matches_conjunction=equal_choices, strict_sign_reversals=sign_flips,
                   sign_gate=dict(switches=switches, wins=wins, losses=losses),
                   review_questions=len(selected), review_windows=len(rows))
    (OUT / "verifier_replay.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
