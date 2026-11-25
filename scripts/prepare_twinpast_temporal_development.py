"""Freeze an outcome-blind pool of fresh HSR pairs and pinned media hashes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

import requests

from prepare_twinpast_development import ANNOTATION_SHA, BASE_URL, REVISION, ROOT

DEST = ROOT / "data/twinpast/temporal_window_development.json"
SEED = "HERMES-TwinPast-temporal-window-pilot-v1"
DATASET = "younght/TwinPast"


def get(url):
    response = requests.get(url, timeout=(10, 30))
    response.raise_for_status()
    return response


def media(row):
    pair = row["pair_id"]
    url = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/videos/{pair}"
    files = {f["path"]: f for f in get(url).json()}
    decoded = {}
    paths = {"pair_id": pair}
    for b in "ab":
        name = paths[f"video_{b}"] = row[f"video_{b}"]
        file = files[name]
        assert 0 < file["size"] <= 128 * 1024**2 and file["size"] == file["lfs"]["size"]
        decoded[b] = {"bytes": file["size"], "sha256": file["lfs"]["oid"]}
    return [paths, {"decoded": decoded}, None]


def freeze():
    raw = get(f"{BASE_URL}/data.jsonl").content
    assert hashlib.sha256(raw).hexdigest() == ANNOTATION_SHA
    rows = {row["pair_id"]: row for row in map(json.loads, raw.splitlines())}
    assert len(rows) == 350
    previous = json.loads((ROOT / "data/twinpast/historical_correction_development.json").read_bytes())
    inspected = [p["pair_id"] for p in previous["selected_pairs"]]
    eligible = [r for p, r in rows.items() if p not in inspected and r["task"] == "HSR" and 18 <= r["t_query"] <= 70]
    ranked = sorted(eligible, key=lambda row: hashlib.sha256(f"{SEED}:{row['pair_id']}".encode()).digest())
    selected = ranked[:12]
    assert len(selected) == len(inspected) == 12
    with ThreadPoolExecutor(max_workers=5) as pool:
        media_plan = list(pool.map(media, [*(rows[p] for p in inspected), *selected]))
    return dict(dataset=DATASET, revision=REVISION, annotation_sha256=ANNOTATION_SHA, seed=SEED,
                expected_max_pairs=8, excluded_development_ids=inspected, candidates=selected, media_plan=media_plan,
                source_status="Original source IDs unavailable. Media screening is a duplicate proxy.")


if __name__ == "__main__":
    result = freeze()
    DEST.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Frozen {len(result['candidates'])} candidates and {len(result['excluded_development_ids'])} reference pairs")
