"""Build the matched natural-history capability notebook from frozen frame receipts."""
import ast
import json

from make_twinpast_boundary_notebook import DEST as TEMPLATE
from validate_twinpast_diagnostic_evidence import ROOT, digest

MANIFEST = ROOT / "data/twinpast/droid_natural_history_v1.json"
DEST = ROOT / "notebooks/autodl/hermes-droid-natural-history.ipynb"
FRAME_ZIP = "DROID_natural_history_frames.zip"
INTRO = """# DROID natural-history capability

Upload this notebook and **DROID_natural_history_frames.zip** to the same
JupyterLab folder on the same AutoDL container used for the previous tests.
Use at least 22 GiB of GPU VRAM. Select **Restart Kernel → Run All**. The
notebook checks the frame archive before GPU setup and reuses verified
environment and model caches if they are available.

Each of the seven development pairs contains two real episodes filmed in
one recorded scene. Each history includes opposite initial and terminal
states of the same object. Both histories are followed by four identical
real frames from an unrelated activity. Compare six uniformly spaced source
frames with six frames selected to include at least three terminal-state
observations. Use the same direct scorer and both option orders.

The robust-pair criterion is at least two recoveries without a regression.
The audit stores 56 choice scores and 28 exact uniform repeat checks.
These are development scenes from four collection sites.

The resized frames come from [DROID](https://droid-dataset.github.io/),
Khazatsky et al. and the DROID Dataset Team, under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

After **DROID_NATURAL_HISTORY_COMPLETE**, save this executed notebook and
download **HERMES_droid_natural_history_evidence.zip**. If it fails after
launch, run the final download cell manually to recover evidence.
"""


def notebook():
    book = json.loads(TEMPLATE.read_bytes())
    cells = book["cells"]

    def get(i):
        return "".join(cells[i]["source"])

    def put(i, source):
        cells[i]["source"] = source.splitlines(keepends=True)

    manifest = json.loads(MANIFEST.read_bytes())
    sources = ast.literal_eval(ast.parse(get(4)).body[0].value)
    prior = json.loads(sources.pop("provenance/boundary_cases.json"))
    for name in ("boundary_runner.py", "twinpast_boundary_policy.py", "evaluation_targets.json"):
        sources.pop(f"provenance/{name}")
    cases, targets = [], {}
    recent = manifest["cases"][0]["recent_frames"]
    for row in manifest["cases"]:
        label = row["pair_id"]
        targets[label] = {b: trace["gold"] for b, trace in row["branches"].items()}
        assert set(targets[label].values()) == {"A", "B"} and row["recent_frames"] == recent
        case = {k: row[k] for k in ("pair_id", "source_site", "scene_id", "question", "options", "recent_frames")}
        case["branches"] = {b: {"frames": trace["frames"]} for b, trace in row["branches"].items()}
        cases.append(case)
    assert len(cases) == 7 and len({c["source_site"] for c in cases}) == 4
    spec = dict(model_revision=prior["model_revision"], video_processor=prior["video_processor"],
                manifest_sha256=digest(MANIFEST.read_bytes()), frame_zip_sha256=manifest["frame_zip_sha256"],
                frame_receipts=manifest["frame_receipts"], cases=cases)
    sources.update({
        "provenance/droid_natural_manifest.json": MANIFEST.read_text(encoding="utf-8"),
        "provenance/droid_natural_cases.json": json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
        "provenance/droid_natural_targets.json": json.dumps(targets, indent=2) + "\n",
        "provenance/droid_natural_runner.py": (ROOT / "scripts/autodl_droid_natural_history.py").read_text(encoding="utf-8"),
    })
    hashes = {name: digest(src.encode()) for name, src in sources.items()}
    put(0, INTRO)
    put(1, get(1).replace("hermes_twinpast_boundary", "hermes_droid_natural_history")
        .replace("HERMES_TWINPAST_BOUNDARY_VALIDATE_ONLY", "HERMES_DROID_NATURAL_HISTORY_VALIDATE_ONLY")
        .replace("provenance/boundary_runner.py", "provenance/droid_natural_runner.py")
        .replace("8 archived pairs, 3 policies, frozen CPU rule", "7 natural DROID pairs, 2 matched frame policies")
        .replace("PREVIOUS = REUSE\n", "") + f"""
FRAME_PACKAGE = Path.cwd() / {FRAME_ZIP!r}
if not VALIDATE_ONLY:
    if not FRAME_PACKAGE.is_file():
        raise FileNotFoundError(f'Upload the companion frame ZIP alongside this notebook: {{FRAME_PACKAGE}}')
    with FRAME_PACKAGE.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != {manifest['frame_zip_sha256']!r}:
            raise ValueError(f'Frame ZIP has the wrong SHA-256: {{FRAME_PACKAGE}}')
    print('DROID_FRAME_PACKAGE_VERIFIED', FRAME_PACKAGE, flush=True)
""")
    put(4, f"SOURCES = {sources!r}\nEXPECTED = {hashes!r}\n" + "\n".join(get(4).split("\n")[2:]))
    put(7, "## 2. Compare uniform and annotated natural histories\n")
    put(8, get(8).replace("HERMES_twinpast_boundary_evidence.zip", "HERMES_droid_natural_history_evidence.zip")
        .replace("'--previous', PREVIOUS", "'--frames', FRAME_PACKAGE")
        .replace("'twinpast_boundary.log'", "'droid_natural_history.log'"))
    for i, cell in enumerate(cells):
        cell["id"] = f"droid-natural-{i:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    return book


if __name__ == "__main__":
    DEST.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST}")
