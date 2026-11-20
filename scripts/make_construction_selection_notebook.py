"""Build the self-contained training-free history selector pilot."""
import hashlib
import json
from zipfile import ZipFile

import make_short_query_notebook as short
from validate_causal_pilot_evidence import check_manifest
from validate_short_query_evidence import ARCHIVE, ARCHIVE_SHA

ROOT = short.ROOT
DEST = ROOT / "notebooks/autodl/hermes-construction-selection.ipynb"
ARTIFACT = "HERMES_construction_selection_evidence.zip"

INTRO = """# HERMES construction history selection

Open this notebook in the **same AutoDL container** and choose **Restart Kernel → Run All**.
Defaults reuse the model and video features under `/root/autodl-tmp/hermes_causal_pilot`.
Output goes to `/root/autodl-tmp/hermes_construction_selection`. No new images or annotations are needed.

This pilot asks whether two selected older frames can supply the context of the full
older window when encoding the recent frames. A first-layer, visual-only score picks
the pair greedily to preserve the full-history attention output. The answer decoder
uses only the prompt and recent K/V in every arm.

| Arm | Older frames used for recent encoding |
|---|---|
| Selected two | Greedy attention-output matching |
| Latest two | Two nearest older frames |
| Attention-mass two | Two older frames with most first-layer attention mass |
| No history | No older frames, original positions preserved |
| All history | Entire available older window |
| Compact recent-four | No older frames, ordinary compact positions |

Runs **240 answers on the same 40 development questions**. The two-frame arms
all use the same construction budget. The runner saves selected frame indices,
positions, the scorer cost, construction time, extraction time and answer latency.
Shared history preparation is reported separately. The mass baseline is charged the
shared scorer cost, so its standalone cost could be lower. Single-pass timings
are exploratory. All-history and compact recent-four must replay their prior choices.

Watch for `SHORT_QUERY_PREFLIGHT_PASS` and `QUESTION_BLOCK_DONE 1/40`.
At `CONSTRUCTION_SELECTION_COMPLETE`, save the executed notebook and download
**HERMES_construction_selection_evidence.zip**. This is a development test before
source-separated confirmation.
"""


def reference_data():
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == ARCHIVE_SHA
    with ZipFile(ARCHIVE) as archive:
        manifest, payload = check_manifest(archive)
    assert manifest["status"] == "complete"
    ref = json.loads(payload["reference.json"])
    arms = {"short_contextual": "all_history", "compact_recent4": "compact_recent4"}
    rows = [dict(sample_id=row["sample_id"], qa_index=row["qa_index"], arm=arms[row["method"]],
                 choice=row["pred_choice"])
            for n, blob in payload.items() if n.startswith("checkpoints/")
            for row in json.loads(blob)["rows"] if row["method"] in arms]
    expected = {(v["sample_id"], qi, arm) for v in ref["videos"]
                for qi in range(len(v["conversations"])) for arm in arms.values()}
    assert len(rows) == len(expected) == 80
    assert {(r["sample_id"], r["qa_index"], r["arm"]) for r in rows} == expected
    return dict(videos=ref["videos"], source_sha256=ref["source_sha256"], baseline_choices=rows,
                short_query_archive_sha256=ARCHIVE_SHA)


def notebook():
    nb = short.notebook()
    cells = nb["cells"]
    for i, cell in enumerate(cells):
        cell["id"] = f"selection-{i:02d}"

    def replace(i, source):
        if cells[i]["cell_type"] == "code":
            compile(source, cells[i]["id"], "exec")
        cells[i]["source"] = source.splitlines(keepends=True)

    replace(0, INTRO)
    config = "".join(cells[1]["source"]).replace("hermes_short_query", "hermes_construction_selection").replace(
        "HERMES_SHORT_QUERY_VALIDATE_ONLY", "HERMES_SELECTION_VALIDATE_ONLY").replace(
        "REPEATS = 2\n", "").replace("160 answers, 320 timed calls by default", "240 construction-selection answers")
    replace(1, config)
    sources = dict(short.matched.SOURCES, RUNNER_SOURCE="autodl_construction_selection.py",
                   SHORT_BASE_SOURCE="autodl_short_query.py", MATCHED_BASE_SOURCE="autodl_matched_boundary.py")
    parts = []
    for variable, name in sources.items():
        code = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "'''" not in code
        digest = hashlib.sha256(code.encode()).hexdigest()
        parts.extend([variable + " = r'''" + code + "'''\n", f"ast.parse({variable})\n",
                      f"assert hashlib.sha256({variable}.encode()).hexdigest() == {digest!r}\n"])
    ref = json.dumps(reference_data(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    parts.append(f"REFERENCE_JSON = {ref!r}\nassert hashlib.sha256(REFERENCE_JSON.encode()).hexdigest() == {hashlib.sha256(ref.encode()).hexdigest()!r}\n")
    embed = short.matched.stage.EMBED.replace("('stage_runner.py',RUNNER_SOURCE),",
        "('stage_runner.py',RUNNER_SOURCE),('provenance/stage_base.py',STAGE_BASE_SOURCE),"
        "('provenance/matched_base.py',MATCHED_BASE_SOURCE),('provenance/short_base.py',SHORT_BASE_SOURCE),").replace(
        "            target=WORK/name\n", "            target=WORK/name\n            target.parent.mkdir(parents=True,exist_ok=True)\n")
    replace(8, "".join(parts) + embed)
    replace(9, "## 3. Validate and run the construction-selection pilot\n")
    run = "".join(cells[10]["source"]).replace(",repeats=REPEATS", "").replace(
        "'--repeats',str(REPEATS),", "").replace("'short_query.log'", "'selection.log'")
    replace(10, run)
    replace(12, "".join(cells[12]["source"]).replace("HERMES_short_query_evidence.zip", ARTIFACT))
    return nb


if __name__ == "__main__":
    nb = notebook()
    DEST.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST} ({len(nb['cells'])} cells)")
