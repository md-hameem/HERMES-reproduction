"""Build a self-contained 7B AutoDL replay from archived 0.5B frame manifests."""
import ast
import hashlib
import json
from validate_twinpast_diagnostic_evidence import verify_inventory
from validate_twinpast_history_only_evidence import ARCHIVE, ROOT, TEMPLATE

DEST = ROOT / "notebooks/autodl/hermes-twinpast-backbone.ipynb"
AUDIT = ROOT / "evidence/audits/twinpast_history_only_2026-11-23_audit.json"
MODEL_ID = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
MODEL_REVISION = "0d50680527681998e456c7b78950205bedd8a068"
INTRO = """# TwinPast stronger-backbone capability check

Upload this notebook alone and choose **Restart Kernel → Run All**. Use the same RTX 4090
or another CUDA 12.8-compatible GPU with at least 22 GiB of VRAM. The notebook reuses
the verified TwinPast clips and Python environment when available.

The pinned LLaVA-OneVision Qwen2 7B weights are approximately **16.1 GB** and need to be
downloaded once. Have at least **18 GiB free after environment setup** for an uncached
model. Weight files are cached and recoverable on rerun. The first six-frame pair tests
GPU fit and checks the scored options against the native-cache path.

The experiment replays exactly four six-frame Historical State Recall pairs and three
two-frame Attribute and Identity Recall positive controls from the completed 0.5B run.
Both branches and both option orders are evaluated. A pair passes only if both branches
are correct under both orders. The predeclared development criterion is at least two
of four historical-state pairs and two of three positive controls. This tests capability
under supplied historical evidence.

Watch for **TWINPAST_BACKBONE_PREFLIGHT_PASS** and **TWINPAST_BACKBONE_COMPLETE**.
Save the executed notebook and download **HERMES_twinpast_backbone_evidence.zip**.
"""

def notebook():
    history_sha = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    audit = json.loads(AUDIT.read_bytes())
    assert history_sha == audit["archive_sha256"] and audit["dense_pairs"] == 4
    contents = verify_inventory(ARCHIVE)
    history = json.loads(contents["provenance/prior.json"])
    cases = []
    for annotation in history["annotations"]:
        pair_id = annotation["pair_id"]
        checkpoint = json.loads(contents[f"checkpoints/{pair_id}.json"])
        if annotation["task"] == "HSR":
            cohort, rows = "historical_state", checkpoint["history_dense"]
        elif checkpoint["summary"]["two_history_only"]["robust"]:
            cohort, rows = "positive_control", checkpoint["history_two"]
        else:
            continue
        assert len(rows) == 4
        cases.append(dict(annotation=annotation, cohort=cohort, reference_rows=rows,
                          original_prompt_rows=history["pairs"][pair_id]))
    assert sum(c["cohort"] == "historical_state" for c in cases) == 4
    assert sum(c["cohort"] == "positive_control" for c in cases) == 3
    book = json.loads(TEMPLATE.read_bytes())
    cells = book["cells"]

    def get(index):
        return "".join(cells[index]["source"])

    def put(index, value):
        cells[index]["source"] = value.splitlines(keepends=True)

    first = ast.parse(get(4)).body[0]
    sources = ast.literal_eval(first.value)
    sources.pop("provenance/prior.json")
    assert all(text.encode() == contents[name] for name, text in sources.items())
    sources["provenance/backbone_cases.json"] = json.dumps(dict(model_id=MODEL_ID, model_revision=MODEL_REVISION,
        history_only_archive_sha256=history_sha, cases=cases), ensure_ascii=False, indent=2) + "\n"
    sources["provenance/backbone_runner.py"] = (ROOT / "scripts/autodl_twinpast_backbone.py").read_text(encoding="utf-8")
    hashes = {key: hashlib.sha256(value.encode()).hexdigest() for key, value in sources.items()}
    put(0, INTRO)
    put(1, get(1).replace("/hermes_twinpast_history_only", "/hermes_twinpast_backbone").replace(
        "HERMES_TWINPAST_HISTORY_VALIDATE_ONLY", "HERMES_TWINPAST_BACKBONE_VALIDATE_ONLY").replace(
        "provenance/history_only_runner.py", "provenance/backbone_runner.py").replace(
        "9 pairs, 36 required history-only score rows", "7 pairs, 28 scores on the pinned 7B backbone").replace(
        "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", MODEL_ID).replace(
        "74dd0bf867a4cda7950c17663794267c60cf4b40", MODEL_REVISION))
    put(3, get(3).replace(">= 15000", ">= 22528").replace("at least 15 GiB VRAM", "at least 22 GiB VRAM"))
    put(4, f"SOURCES = {sources!r}\nEXPECTED = {hashes!r}\n" + "\n".join(get(4).split("\n")[2:]))
    put(6, (ROOT / "scripts/twinpast_backbone_model_cell.py").read_text(encoding="utf-8"))
    put(7, "## 2. Replay the frozen windows on the pinned 7B model\n")
    put(8, get(8).replace("HERMES_twinpast_history_only_evidence.zip", "HERMES_twinpast_backbone_evidence.zip").replace(
        "'twinpast_history_only.log'", "'twinpast_backbone.log'"))
    for index, cell in enumerate(cells):
        cell["id"] = f"twinpast-backbone-{index:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    return book


if __name__ == "__main__":
    DEST.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST}")
