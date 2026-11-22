"""Build the fixed-recent experiment from the audited 7B run."""
import ast
import hashlib
import json

from validate_twinpast_backbone_evidence import ARCHIVE, ROOT, TEMPLATE, verify_inventory

DEST = ROOT / "notebooks/autodl/hermes-twinpast-fixed-recent.ipynb"
AUDIT = ROOT / "evidence/audits/twinpast_backbone_2026-11-23_audit.json"
INTRO = """# TwinPast 7B with fixed recent frames

Upload this notebook alone to the same AutoDL RTX 4090 instance and choose **Restart Kernel
→ Run All**. It reuses the pinned 7B weights, Python environment and clips from the existing
cache. The download cell verifies the cached model files before loading them.

This run uses the same seven pairs as the completed 7B capability test. It replays the
history-only input and compares its logits and answers to the archived scores. It then
prepends each branch's original earlier frames to the exact same four recent frames.
A single recent-only result per pair and option order serves as the shared control.
The longest input has ten frames. The first pair checks GPU fit and deterministic scoring.

The decision rule requires both previously recovered historical-state pairs and at least
two of three positive controls to remain correct on both branches in both option orders,
with the archived history-only decisions reproduced. The run uses supplied evidence.
Adding recent frames also changes sequence length and the position of the question.

When you see **TWINPAST_FIXED_RECENT_COMPLETE**, save the executed notebook and download
**HERMES_twinpast_fixed_recent_evidence.zip**.
"""


def notebook():
    archive_hash = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    audit = json.loads(AUDIT.read_bytes())
    assert archive_hash == audit["archive_sha256"] and audit["development_criterion_met"]
    evidence = verify_inventory(ARCHIVE)
    spec = json.loads(evidence["provenance/backbone_cases.json"])
    cases = []
    recovered = []
    for case in spec["cases"]:
        pair_id = case["annotation"]["pair_id"]
        block = json.loads(evidence[f"checkpoints/{pair_id}.json"])["block"]
        assert block["cohort"] == case["cohort"] and block["pair_id"] == pair_id
        cases.append(dict(annotation=case["annotation"], cohort=case["cohort"],
                          original_prompt_rows=case["original_prompt_rows"], backbone_rows=block["rows"]))
        if block["cohort"] == "historical_state" and block["current"]["robust"]:
            recovered.append(pair_id)
    assert len(cases) == 7 and sorted(recovered) == ["twinpast_0094", "twinpast_0259"]
    frozen = dict(backbone_archive_sha256=archive_hash, recovered_states=recovered, cases=cases,
                  prior_config=json.loads(evidence["exports/run_config.json"]))
    book = json.loads(TEMPLATE.read_bytes())
    cells = book["cells"]

    def get(index):
        return "".join(cells[index]["source"])

    def put(index, value):
        cells[index]["source"] = value.splitlines(keepends=True)

    sources = ast.literal_eval(ast.parse(get(4)).body[0].value)
    assert all(text.encode("utf-8") == evidence[name] for name, text in sources.items())
    sources.pop("provenance/backbone_cases.json")
    sources["provenance/fixed_recent_cases.json"] = json.dumps(frozen, ensure_ascii=False, indent=2) + "\n"
    sources["provenance/fixed_recent_runner.py"] = (ROOT / "scripts/autodl_twinpast_fixed_recent.py").read_text(encoding="utf-8")
    hashes = {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in sources.items()}
    put(0, INTRO)
    put(1, get(1).replace("/hermes_twinpast_backbone", "/hermes_twinpast_fixed_recent")
        .replace("HERMES_TWINPAST_BACKBONE_VALIDATE_ONLY", "HERMES_TWINPAST_FIXED_RECENT_VALIDATE_ONLY")
        .replace("provenance/backbone_runner.py", "provenance/fixed_recent_runner.py")
        .replace("7 pairs, 28 scores on the pinned 7B backbone", "7 pairs, 70 fixed-recent scores on 7B"))
    put(4, f"SOURCES = {sources!r}\nEXPECTED = {hashes!r}\n" + "\n".join(get(4).split("\n")[2:]))
    put(7, "## 2. Score history alone and with fixed recent frames\n")
    put(8, get(8).replace("HERMES_twinpast_backbone_evidence.zip", "HERMES_twinpast_fixed_recent_evidence.zip")
        .replace("'twinpast_backbone.log'", "'twinpast_fixed_recent.log'"))
    for index, cell in enumerate(cells):
        cell["id"] = f"twinpast-fixed-recent-{index:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    return book


if __name__ == "__main__":
    DEST.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST}")
