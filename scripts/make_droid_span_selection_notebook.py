"""Build the AutoDL notebook for the frozen source-disjoint span cohort."""
import ast
import json

from freeze_droid_span_cohort import FRAME_ZIP, MANIFEST
from validate_droid_terminal_attention_evidence import ARCHIVE, ROOT, TEMPLATE, digest, verify_inventory

DEST = ROOT / "notebooks/autodl/hermes-droid-span-selection.ipynb"
INTRO = """# DROID automatic three-frame span selection

Open this notebook on the same GPU container as the terminal-attention run.
Place **DROID_span_cohort_frames.zip** beside it and select **Restart Kernel → Run All**.
It verifies the frame receipts and reuses the pinned 7B model and environment when available.

Each fresh scene contributes two opposite-state episodes. Score a first-view
and last-view question on each episode, in both option orders. Keep the ten
visual frames, their order and token positions fixed for all conditions.

Compare unrestricted decoding, last-three recency, a first/last rule,
endpoint-contrast selection and a window chosen from the visual annotations.
The selector sees the question and visual features. A full causal explicit
mask checks that the masking implementation reproduces native predictions.

Save the executed notebook and download **HERMES_droid_span_selection_evidence.zip**
after **DROID_SPAN_SELECTION_COMPLETE**. Run the final cell manually to
retrieve failure evidence if the runner stops.

Frames are resized from [DROID](https://droid-dataset.github.io/) under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
"""


def notebook():
    book = json.loads(TEMPLATE.read_bytes())
    cells = book["cells"]

    def get(i):
        return "".join(cells[i]["source"])

    def put(i, source):
        cells[i]["source"] = source.splitlines(keepends=True)

    archived = verify_inventory(ARCHIVE)
    sources = ast.literal_eval(ast.parse(get(4)).body[0].value)
    assert all(source.encode() == archived[name] for name, source in sources.items())
    prior = json.loads(sources.pop("provenance/droid_attention_cases.json"))
    sources.pop("provenance/droid_natural_targets.json")
    sources.pop("provenance/droid_natural_manifest.json")
    spec = json.loads(MANIFEST.read_bytes())
    assert len(spec["cases"]) >= 6 and len({r["source_site"] for r in spec["cases"]}) >= 3
    spec.update(model_revision=prior["model_revision"], video_processor=prior["video_processor"],
                previous_evidence_sha256=digest(ARCHIVE.read_bytes()))
    sources.update({
        "provenance/droid_span_cohort.json": json.dumps(spec, indent=2) + "\n",
        "provenance/droid_span_selector.py": (ROOT / "scripts/droid_span_selector.py").read_text(encoding="utf-8"),
        "provenance/droid_span_runner.py": (
            ROOT / "scripts/autodl_droid_span_selection.py"
        ).read_text(encoding="utf-8"),
    })
    hashes = {name: digest(src.encode()) for name, src in sources.items()}
    put(0, INTRO)
    put(1, get(1).replace("hermes_droid_terminal_attention", "hermes_droid_span_selection")
        .replace("HERMES_DROID_TERMINAL_ATTENTION_VALIDATE_ONLY", "HERMES_DROID_SPAN_SELECTION_VALIDATE_ONLY")
        .replace("provenance/droid_attention_runner.py", "provenance/droid_span_runner.py")
        .replace("DROID_natural_history_frames.zip", FRAME_ZIP)
        .replace(prior["frame_zip_sha256"], spec["frame_zip_sha256"])
        .replace("7 DROID pairs, 4 attention conditions", f"{len(spec['cases'])} DROID scene pairs, 5 reported conditions"))
    put(4, f"SOURCES = {sources!r}\nEXPECTED = {hashes!r}\n" + "\n".join(get(4).split("\n")[2:]))
    put(7, "## 2. Score the fresh scenes against matched recency\n")
    put(8, get(8).replace("HERMES_droid_terminal_attention_evidence.zip", "HERMES_droid_span_selection_evidence.zip")
        .replace("'droid_terminal_attention.log'", "'droid_span_selection.log'"))
    for i, cell in enumerate(cells):
        cell["id"] = f"droid-span-selection-{i:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    return book


if __name__ == "__main__":
    DEST.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST}")
