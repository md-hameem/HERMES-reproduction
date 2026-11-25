"""Build the matched sub-second refinement pilot from frozen manifests."""
import ast
import json

from prepare_twinpast_refinement_development import DEST as MANIFEST
from validate_twinpast_subsecond_evidence import ARCHIVE, ROOT, TEMPLATE, digest, verify_inventory

DEST = ROOT / "notebooks/autodl/hermes-twinpast-refinement.ipynb"
INTRO = """# TwinPast matched sub-second refinement

Run on an AutoDL RTX 4090 instance with **Restart Kernel → Run All**. On a fresh
container, plan for around 40 GiB of free space on `/root/autodl-tmp`. The notebook
sets up its environment and downloads the pinned 7B model and verified clips as
needed. It can also reuse valid assets from earlier runs on the same container.
It screens twelve new Historical State Recall candidates against prior footage and
evaluates up to eight accepted pairs. Previous development pairs are excluded.
Sampled duplicate screening does not resolve original-source provenance.

Both local refinement arms use the same three query-relevant historical windows.
Each window is decoded at 4 FPS. Uniform refinement picks two fixed sub-second offsets.
Change-guided refinement picks the frames that bracket the largest low-resolution
RGB change. That score is a visual proxy and may respond to camera motion.
Both arms draw from the same 39 decoded history candidates, encode six selected
history frames for answering, and append four matched recent frames.
Ordinary 1 FPS uniform history is the practical reference.

The selector uses no gold answers or evidence intervals. Compare the robust paired
results against local uniform before advancing. The contrast matches window access
and sampled frames, while the change score incurs extra image processing.

After **TWINPAST_REFINEMENT_COMPLETE**, save the executed notebook and download
**HERMES_twinpast_refinement_evidence.zip**.
"""


def notebook():
    evidence = verify_inventory(ARCHIVE)
    assert json.loads(evidence["exports/summary.json"])["development_criterion_met"]
    previous = json.loads(evidence["provenance/subsecond_cases.json"])
    manifest = json.loads(MANIFEST.read_bytes())
    targets, cases = {}, []
    for row in manifest["candidates"]:
        pair = row["pair_id"]
        assert pair not in manifest["excluded_ids"] and row["answer_a"] != row["answer_b"]
        targets[pair] = {k: row[k] for k in ("answer_a", "answer_b")}
        cases.append(dict(pair_id=pair, query_s=row["t_query"], question=row["question"], options=row["options"]))
    assert len(cases) == len(targets) == 12
    spec = dict(model_revision=previous["model_revision"], video_processor=previous["video_processor"],
                media_source={"manifest": {k: manifest[k] for k in ("dataset", "revision")}},
                media_plan=manifest["media_plan"], reference_ids=manifest["reference_ids"],
                excluded_ids=manifest["excluded_ids"], source_status=manifest["source_status"],
                expected_max_pairs=manifest["expected_max_pairs"], cases=cases,
                manifest_sha256=digest(MANIFEST.read_bytes()), subsecond_archive_sha256=digest(ARCHIVE.read_bytes()))
    book = json.loads(TEMPLATE.read_bytes())
    cells = book["cells"]

    def get(i):
        return "".join(cells[i]["source"])

    def put(i, value):
        cells[i]["source"] = value.splitlines(keepends=True)

    sources = ast.literal_eval(ast.parse(get(4)).body[0].value)
    assert all(s.encode() == evidence[n] for n,s in sources.items())
    for name in ("subsecond_cases.json", "subsecond_runner.py"):
        sources.pop(f"provenance/{name}")
    sources["provenance/refinement_cases.json"] = json.dumps(spec, ensure_ascii=False, indent=2) + "\n"
    sources["provenance/evaluation_targets.json"] = json.dumps(targets, sort_keys=True, indent=2) + "\n"
    sources["provenance/refinement_runner.py"] = (ROOT / "scripts/autodl_twinpast_refinement.py").read_text(encoding="utf-8")
    hashes = {n: digest(s.encode()) for n,s in sources.items()}
    new = sum(f["bytes"] for p,m,_ in manifest["media_plan"] if p["pair_id"] in targets for f in m["decoded"].values())
    put(0, INTRO + f"\nNew candidate clips total {new / 1024**2:.0f} MiB.\n")
    put(1, get(1).replace("/hermes_twinpast_subsecond", "/hermes_twinpast_refinement")
        .replace("HERMES_TWINPAST_SUBSECOND_VALIDATE_ONLY", "HERMES_TWINPAST_REFINEMENT_VALIDATE_ONLY")
        .replace("provenance/subsecond_runner.py", "provenance/refinement_runner.py")
        .replace("7 pairs, 56 sub-second evidence score rows", "Up to 8 fresh pairs, 3 matched methods"))
    put(4, f"SOURCES = {sources!r}\nEXPECTED = {hashes!r}\n" + "\n".join(get(4).split("\n")[2:]))
    put(7, "## 2. Screen media and compare the two refinement policies\n")
    put(8, get(8).replace("HERMES_twinpast_subsecond_evidence.zip", "HERMES_twinpast_refinement_evidence.zip")
        .replace("'twinpast_subsecond.log'", "'twinpast_refinement.log'"))
    for i, cell in enumerate(cells):
        cell["id"] = f"twinpast-refinement-{i:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    return book


if __name__ == "__main__":
    DEST.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST}")
