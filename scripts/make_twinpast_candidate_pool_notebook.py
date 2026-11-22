"""Freeze and embed the candidate-pool diagnostic and its matched references."""
import ast
import json
import math

from validate_twinpast_automatic_evidence import ARCHIVE, ROOT, TEMPLATE, digest, verify_inventory
from validate_twinpast_fixed_recent_evidence import ARCHIVE as FIXED_RECENT

DEST = ROOT / "notebooks/autodl/hermes-twinpast-candidate-pool.ipynb"
INTRO = """# TwinPast candidate-pool capability check

Upload this notebook to the same AutoDL RTX 4090 instance and choose **Restart Kernel →
Run All**. It uses the existing pinned 7B cache and clips.

This diagnostic uses the four historical-state development pairs. Each condition supplies
six earlier frames and the same four recent frames, with both option orders. The arms
replay the previously curated frames, select from the original capped 0.5 FPS pool, or
select from a denser 1 FPS pool. Both pool arms use the same deterministic rule. Spread
selections across frames in the annotated intervals, then fill empty slots from the
earliest remaining frames. The annotations are privileged inputs for this diagnostic.

The curated replay must preserve all archived decisions within 0.05 absolute logit error.
Evidence for expanding the pool requires at least one additional robust paired success
from the denser pool, zero losses relative to the capped pool, and no replay mismatches.
A pair is robust when both branches are correct under both option orders.

When you see **TWINPAST_CANDIDATE_POOL_COMPLETE**, save the executed notebook and download
**HERMES_twinpast_candidate_pool_evidence.zip**.
"""


def notebook():
    automatic = verify_inventory(ARCHIVE)
    fixed = verify_inventory(FIXED_RECENT)
    assert json.loads(automatic["completion.json"]) == {"status": "complete", "pairs": 7}
    auto_spec = json.loads(automatic["provenance/automatic_cases.json"])
    fixed_spec = json.loads(fixed["provenance/fixed_recent_cases.json"])
    assert auto_spec["fixed_recent_archive_sha256"] == digest(FIXED_RECENT.read_bytes())
    media = {r["pair_id"]: r for r in json.loads(fixed["provenance/bundle.json"])["media"]}
    cases = []
    for case in fixed_spec["cases"]:
        if case["cohort"] != "historical_state":
            continue
        annotation = case["annotation"]
        pair = annotation["pair_id"]
        fixed_block = json.loads(fixed[f"checkpoints/{pair}.json"])["block"]
        automatic_block = json.loads(automatic[f"checkpoints/{pair}.json"])["block"]
        older = {b: automatic_block["branches"][b]["sample"]["grid_s"][:-4] for b in "ab"}
        assert older["a"] == older["b"]
        record = media[pair]
        recent = {b: dict(source_frames=record["decoded"][b]["source_frames"][-4:],
                          decoded_rgb_sha256=record["decoded"][b]["decoded_rgb_sha256"][-4:],
                          start_s=record["decoded"][b]["frame_start_s"][-4]) for b in "ab"}
        dense = list(range(math.ceil(min(record["grid_s"][-4], *(r["start_s"] for r in recent.values())))))
        assert set(older["a"]) < set(dense) and len(older["a"]) >= 6
        references = {(r["order"], r["branch"]): r for r in case["backbone_rows"]}
        rows = []
        for r in fixed_block["rows"]["history_with_recent"]:
            earlier = references[r["order"], r["branch"]]
            assert r["history_frame_hashes"] == earlier["frame_hashes"]
            rows.append({**r, **{k: earlier[k] for k in ("frame_indices", "frame_seconds", "frame_hashes")}})
        cases.append(dict(annotation=annotation, curated_rows=rows, recent=recent,
                          recent_grid_s=record["grid_s"][-4:], capped_grid_s=older["a"], dense_grid_s=dense))
    ids = {c["annotation"]["pair_id"] for c in cases}
    assert len(ids) == 4
    spec = dict(model_revision=auto_spec["model_revision"], video_processor=auto_spec["video_processor"],
                media_source=auto_spec["media_source"],
                media_plan=[p for p in auto_spec["media_plan"] if p[0]["pair_id"] in ids],
                source_archives={"automatic": digest(ARCHIVE.read_bytes()), "fixed_recent": digest(FIXED_RECENT.read_bytes())},
                cases=cases)
    book = json.loads(TEMPLATE.read_bytes())
    cells = book["cells"]

    def get(index):
        return "".join(cells[index]["source"])

    def put(index, source):
        cells[index]["source"] = source.splitlines(keepends=True)

    sources = ast.literal_eval(ast.parse(get(4)).body[0].value)
    assert all(s.encode("utf-8") == automatic[n] for n, s in sources.items())
    for name in ("automatic_cases.json", "automatic_runner.py", "evaluation_targets.json"):
        sources.pop(f"provenance/{name}")
    sources["provenance/candidate_pool_cases.json"] = json.dumps(spec, ensure_ascii=False, indent=2) + "\n"
    sources["provenance/candidate_pool_runner.py"] = (
        ROOT / "scripts/autodl_twinpast_candidate_pool.py").read_text(encoding="utf-8")
    hashes = {n: digest(s.encode("utf-8")) for n, s in sources.items()}
    put(0, INTRO)
    put(1, get(1).replace("/hermes_twinpast_automatic", "/hermes_twinpast_candidate_pool")
        .replace("HERMES_TWINPAST_AUTOMATIC_VALIDATE_ONLY", "HERMES_TWINPAST_CANDIDATE_POOL_VALIDATE_ONLY")
        .replace("provenance/automatic_runner.py", "provenance/candidate_pool_runner.py")
        .replace("7 pairs, 6 automatic method outcomes per branch and order", "4 pairs, 48 candidate-pool score rows"))
    put(4, f"SOURCES = {sources!r}\nEXPECTED = {hashes!r}\n" + "\n".join(get(4).split("\n")[2:]))
    put(7, "## 2. Compare the capped and dense candidate pools\n")
    put(8, get(8).replace("HERMES_twinpast_automatic_evidence.zip", "HERMES_twinpast_candidate_pool_evidence.zip")
        .replace("'twinpast_automatic.log'", "'twinpast_candidate_pool.log'"))
    for i, cell in enumerate(cells):
        cell["id"] = f"twinpast-candidate-pool-{i:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    return book


if __name__ == "__main__":
    DEST.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST}")
