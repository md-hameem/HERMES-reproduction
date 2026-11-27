"""Build the same-input DROID competing-state masking comparison."""
import ast
import json

from validate_droid_natural_history_evidence import ARCHIVE, MANIFEST, ROOT, TEMPLATE, digest, verify_inventory

PLAN = ROOT / "data/twinpast/droid_state_interference_v1.json"
DEST = ROOT / "notebooks/autodl/hermes-droid-state-interference.ipynb"
INTRO = """# DROID earlier-state interference

Use the same AutoDL GPU container with at least 22 GiB VRAM. Put this notebook
in the same JupyterLab folder as the existing **DROID_natural_history_frames.zip**,
then select **Restart Kernel → Run All**. The frame pack is the same one used
for the natural-history run.

Compare unmodified annotated inputs, a mask on the earlier target region and
an equal-area mask on a different region. The masks change only the first two
history frames. The three labeled terminal frames and four real recent frames
remain in the same positions. Score both option orders and check all 28 archived
unmodified annotated scores before reporting the comparison.

These are seven reviewed development pairs. The exploratory threshold requires
two target-mask-only robust recoveries, with no losses against either comparator.

Frames are resized from [DROID](https://droid-dataset.github.io/), Khazatsky et al.
and the DROID Dataset Team, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

When **DROID_STATE_INTERFERENCE_COMPLETE** appears, save the executed notebook
and download **HERMES_droid_state_interference_evidence.zip**. If replay fails
after launch, run the final download cell manually to retrieve its evidence.
"""


def notebook():
    book = json.loads(TEMPLATE.read_bytes())
    cells = book["cells"]

    def get(i):
        return "".join(cells[i]["source"])

    def put(i, source):
        cells[i]["source"] = source.splitlines(keepends=True)

    evidence = verify_inventory(ARCHIVE)
    mask_plan = json.loads(PLAN.read_bytes())
    assert mask_plan["natural_history_archive_sha256"] == digest(ARCHIVE.read_bytes())
    assert mask_plan["natural_history_manifest_sha256"] == digest(MANIFEST.read_bytes())
    sources = ast.literal_eval(ast.parse(get(4)).body[0].value)
    assert all(src.encode() == evidence[name] for name, src in sources.items())
    spec = json.loads(sources.pop("provenance/droid_natural_cases.json"))
    originals = {b["pair_id"]: b for b in json.loads(evidence["exports/pair_scores.json"])}
    masks = {row["pair_id"]: row for row in mask_plan["cases"]}
    assert len(masks) == len(originals) == len(spec["cases"]) == 7
    for case in spec["cases"]:
        pair = case["pair_id"]
        case["mask"] = masks[pair]
        for side, branch in case["branches"].items():
            branch["archived_annotated"] = {
                order: {k: row["annotated"][k] for k in ("logits", "prediction", "prompt_sha256", "visual_tokens", "lm_tokens")}
                for order, row in originals[pair]["branches"][side]["answers"].items()
            }
    spec.update(mask_plan_sha256=digest(PLAN.read_bytes()), mask_fill_rgb=mask_plan["fill_rgb"],
                natural_history_archive_sha256=mask_plan["natural_history_archive_sha256"])
    sources.update({
        "provenance/droid_interference_cases.json": json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
        "provenance/droid_interference_mask_plan.json": PLAN.read_text(encoding="utf-8"),
        "provenance/droid_interference_runner.py": (ROOT / "scripts/autodl_droid_interference.py").read_text(encoding="utf-8"),
        "provenance/natural_history_runner.py": sources.pop("provenance/droid_natural_runner.py"),
        "provenance/droid_state_masks.py": (ROOT / "scripts/droid_state_masks.py").read_text(encoding="utf-8"),
    })
    hashes = {name: digest(src.encode()) for name, src in sources.items()}
    put(0, INTRO)
    put(1, get(1).replace("hermes_droid_natural_history", "hermes_droid_state_interference")
        .replace("HERMES_DROID_NATURAL_HISTORY_VALIDATE_ONLY", "HERMES_DROID_STATE_INTERFERENCE_VALIDATE_ONLY")
        .replace("provenance/droid_natural_runner.py", "provenance/droid_interference_runner.py")
        .replace("7 natural DROID pairs, 2 matched frame policies", "7 DROID pairs, 3 matched masking conditions"))
    put(4, f"SOURCES = {sources!r}\nEXPECTED = {hashes!r}\n" + "\n".join(get(4).split("\n")[2:]))
    put(7, "## 2. Replay the annotated baseline and compare masks\n")
    put(8, get(8).replace("HERMES_droid_natural_history_evidence.zip", "HERMES_droid_state_interference_evidence.zip")
        .replace("'droid_natural_history.log'", "'droid_state_interference.log'"))
    for i, cell in enumerate(cells):
        cell["id"] = f"droid-interference-{i:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    return book


if __name__ == "__main__":
    DEST.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST}")
