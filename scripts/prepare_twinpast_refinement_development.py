"""Freeze new HSR candidates for budget-matched local refinement."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

from prepare_twinpast_temporal_development import (
    ANNOTATION_SHA, BASE_URL, DATASET, DEST as PREVIOUS, REVISION, ROOT, get, media,
)
from validate_twinpast_temporal_windows_evidence import ARCHIVE, verify_inventory

DEST = ROOT / "data/twinpast/refinement_development.json"
SEED = "HERMES-TwinPast-local-refinement-v1"


def freeze():
    prior = json.loads(PREVIOUS.read_bytes())
    observed = json.loads(verify_inventory(ARCHIVE)["exports/run_config.json"])["accepted"]
    excluded = prior["excluded_development_ids"] + [r["pair_id"] for r in prior["candidates"]]
    references = prior["excluded_development_ids"] + observed
    raw = get(f"{BASE_URL}/data.jsonl").content
    assert hashlib.sha256(raw).hexdigest() == ANNOTATION_SHA
    eligible = (r for r in map(json.loads, raw.splitlines())
                if r["task"] == "HSR" and r["pair_id"] not in excluded and 18 <= r["t_query"] <= 70)
    candidates = sorted(eligible, key=lambda r: hashlib.sha256(f"{SEED}:{r['pair_id']}".encode()).digest())[:12]
    assert len(candidates) == 12
    with ThreadPoolExecutor(max_workers=5) as pool:
        new_media = list(pool.map(media, candidates))
    reference_media = [p for p in prior["media_plan"] if p[0]["pair_id"] in references]
    assert len(references) == len(reference_media) == 20
    return dict(dataset=DATASET, revision=REVISION, annotation_sha256=ANNOTATION_SHA, seed=SEED,
                expected_max_pairs=8, excluded_ids=excluded, reference_ids=references,
                candidates=candidates, media_plan=reference_media+new_media,
                source_status="Original source IDs unavailable. Sampled media screening is a duplicate proxy.")


if __name__ == "__main__":
    frozen = freeze()
    DEST.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Frozen {len(frozen['candidates'])} new candidates against {len(frozen['reference_ids'])} references")
