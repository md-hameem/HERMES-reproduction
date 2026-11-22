"""Build the follow-up notebook from the audited first TwinPast run."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks/autodl/hermes-twinpast-diagnostic.ipynb"
DEST = ROOT / "notebooks/autodl/hermes-twinpast-history-only.ipynb"
ARCHIVE = ROOT / "evidence/archives/HERMES_twinpast_diagnostic_evidence.zip"
AUDIT = ROOT / "evidence/audits/twinpast_diagnostic_2026-11-23_audit.json"
INTRO = """# TwinPast history-only capability check

Upload **this notebook only** to AutoDL JupyterLab and choose **Restart Kernel → Run All**.
Use the same RTX 4090 container if possible. It verifies and reuses the previous TwinPast clips
in place, plus the pinned Python and model cache. A fresh container runs the same pins and
downloads the same 18 clips when absent. No Git clone or extra upload is needed.

The previous experiment reached 3/9 paired successes in both option orders. This diagnostic
scores **the exact two earlier frames alone**, for each of the same nine pairs, both branches,
and the two original option orders. Historical State Recall pairs that still fail in either
order receive one additional six-frame history test. That test retains the two original
frames and adds four sampled across each branch's annotated intervals with the same
relative sampling rule. All frames stop before the recent window. The evidence ZIP
includes contact sheets for those historical-state cases.

The notebook preserves the previous logits and prompt identity for comparison. It uses
annotated earlier evidence and measures capability on development cases. Both visual
content and the prompt's sequence position can differ from the earlier six-frame run.

Watch for **TWINPAST_HISTORY_ONLY_PREFLIGHT_PASS**, **PAIR_DONE** and
**TWINPAST_HISTORY_ONLY_COMPLETE**. Download the evidence ZIP and save the executed notebook.
"""


def notebook():
    raw = ARCHIVE.read_bytes()
    hash_archive = hashlib.sha256(raw).hexdigest()
    audit = json.loads(AUDIT.read_bytes())
    assert hash_archive == audit["archive_sha256"] and audit["pairs"] == 9 and audit["scores"] == 72
    with ZipFile(ARCHIVE) as archive:
        manifest = json.loads(archive.read("evidence_manifest.json"))
        payload = {}
        for entry in manifest:
            content = archive.read(entry["path"])
            assert len(content) == entry["bytes"] and hashlib.sha256(content).hexdigest() == entry["sha256"]
            payload[entry["path"]] = content
    pairs = {}
    for name, content in payload.items():
        if name.startswith("checkpoints/"):
            saved = json.loads(content)
            scores = [r for r in saved["records"] if r["arm"] in {"history_a", "history_b"}]
            assert len(scores) == 4
            pairs[saved["pair_id"]] = scores
    assert len(pairs) == 9
    book = json.loads(TEMPLATE.read_bytes())
    cells = book["cells"]

    def get(index):
        return "".join(cells[index]["source"])

    def put(index, value):
        cells[index]["source"] = value.splitlines(keepends=True)

    import ast

    first = ast.parse(get(4)).body[0]
    sources = ast.literal_eval(first.value)
    sources["provenance/twinpast_base.py"] = sources.pop("provenance/twinpast_runner.py")
    assert sources["provenance/twinpast_base.py"].encode() == payload["provenance/twinpast_runner.py"]
    assert sources["provenance/bundle.json"].encode() == payload["provenance/bundle.json"]
    sources["provenance/prior.json"] = json.dumps(dict(archive_sha256=hash_archive, pairs=pairs,
        annotations=json.loads(payload["provenance/selected_annotations.json"])), ensure_ascii=False, indent=2) + "\n"
    sources["provenance/history_only_runner.py"] = (ROOT / "scripts/autodl_twinpast_history_only.py").read_text(encoding="utf-8")
    hashes = {key: hashlib.sha256(value.encode()).hexdigest() for key, value in sources.items()}
    put(0, INTRO)
    put(1, get(1).replace("/hermes_twinpast_diagnostic", "/hermes_twinpast_history_only").replace(
        "HERMES_TWINPAST_VALIDATE_ONLY", "HERMES_TWINPAST_HISTORY_VALIDATE_ONLY").replace(
        "provenance/twinpast_runner.py", "provenance/history_only_runner.py").replace(
        "9 pairs, 72 forced-option score rows", "9 pairs, 36 required history-only score rows")
        + "PREVIOUS = WORK.with_name('hermes_twinpast_diagnostic')\n")
    put(4, f"SOURCES = {sources!r}\nEXPECTED = {hashes!r}\n" + "\n".join(get(4).split("\n")[2:]))
    put(7, "## 2. Verify clips and run the history-only capability check\n")
    put(8, get(8).replace("HERMES_twinpast_diagnostic_evidence.zip", "HERMES_twinpast_history_only_evidence.zip").replace(
        "'--media-reuse', MEDIA_REUSE]", "'--media-reuse', MEDIA_REUSE, '--previous', PREVIOUS]").replace(
        "'twinpast.log'", "'twinpast_history_only.log'"))
    for index, cell in enumerate(cells):
        cell["id"] = f"twinpast-history-{index:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    return book


if __name__ == "__main__":
    DEST.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST}")
