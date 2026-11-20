"""Generate the self-contained matched-boundary AutoDL notebook."""
import hashlib
import json
from zipfile import ZipFile

import make_stage_attribution_notebook as stage
from validate_causal_pilot_evidence import check_manifest
from validate_stage_attribution_evidence import ARCHIVE, ARCHIVE_SHA, ARMS

ROOT = stage.ROOT
DEST = ROOT / "notebooks/autodl/hermes-matched-boundary.ipynb"
SOURCES = {
    "RUNNER_SOURCE": "autodl_matched_boundary.py",
    "STAGE_BASE_SOURCE": "autodl_stage_attribution.py",
    "CONTEXT_SOURCE": "autodl_context_diagnostic.py",
    "INPUT_SOURCE": "autodl_input_diagnostic.py",
    "UTILS_SOURCE": "autodl_failure_pilot.py",
}

INTRO = """# HERMES matched-boundary control

Upload this notebook to the **same AutoDL container**, open it in JupyterLab and choose
**Restart Kernel → Run All**. Use the default settings. The notebook reuses the verified
assets under `/root/autodl-tmp/hermes_causal_pilot` and writes to
`/root/autodl-tmp/hermes_matched_boundary`.

**Question:** Does older history help build the recent cache when both construction arms
encode the same recent frames in the same groups and with the same positions?
The previous experiment mixed construction visibility and grouping on 36 of 40 questions.

| Arm | Older history visible while recent frames are encoded? | Older history visible directly while answering? |
|---|---|---|
| A matched | Yes | Yes |
| B | No | Yes |
| C matched | Yes | No |
| D | No | No |
| Compact recent-4 | No history slots | No history slots |

**Primary pair: C matched vs D.** They share the original older cache prefix and
ingest recent frames in identical groups with explicit causal masks from that prefix. Only access to older
tokens during recent encoding differs. Both hide older slots throughout answering.
B, D and compact recent-4 must replay all 120 previous choices. A matched and C
matched are also compared to their previous original-partition counterparts.

The run covers the same **40 questions and 200 answers**. It keeps the same 0.5 FPS
sampled frames, model, prompts and cache positions, and excludes the separate
long-prefix case. There is no new frame curation or annotation. The evidence ZIP
includes frame indices, question-level pairs, comparisons to the previous run,
preflight metrics, source, checkpoints and closed logs.

The runner verifies the existing FP32 physical reference and uses the GPU-tested
FP16 same-shape invariance policy. FP16 logit differences against the physical
reference are recorded, with matching top tokens required. Preflight and
historical replay failures stop the run and preserve evidence.

Watch for `MATCHED_PREFLIGHT_PASS` and `QUESTION_BLOCK_DONE 1/40`. When it
finishes, download **HERMES_matched_boundary_evidence.zip** and save the executed
notebook. With reuse, the earlier experiment took 56 seconds after model load.
Setup, fresh downloads and this new control's timing can differ.
"""


def reference_data():
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == ARCHIVE_SHA
    with ZipFile(ARCHIVE) as archive:
        manifest, payload = check_manifest(archive)
    assert manifest["status"] == "complete"
    previous = json.loads(payload["reference.json"])
    rows = []
    for name, blob in payload.items():
        if name.startswith("checkpoints/main/"):
            row = json.loads(blob)["row"]
            assert row["status"] == "answered" and row["method"] in ARMS
            rows.append({k: row[k] for k in ("sample_id", "qa_index", "method", "pred_choice", "correct")})
    expected = {(v["sample_id"], qi, arm) for v in previous["videos"]
                for qi in range(len(v["conversations"])) for arm in ARMS}
    assert len(rows) == len(expected) == 200
    assert {(r["sample_id"], r["qa_index"], r["method"]) for r in rows} == expected
    return dict(videos=previous["videos"], source_sha256=previous["source_sha256"],
                stage_archive_sha256=ARCHIVE_SHA, stage_rows=rows)


def notebook():
    base = stage.base
    ref = reference_data()
    cells = []

    def cell(kind, source):
        value = dict(cell_type=kind, id=f"matched-{len(cells):02d}",
                     metadata={}, source=source.splitlines(keepends=True))
        if kind == "code":
            compile(source, value["id"], "exec")
            value.update(execution_count=None, outputs=[])
        cells.append(value)

    config = stage.config().replace("hermes_stage_attribution_v2", "hermes_matched_boundary").replace(
        "HERMES_STAGE_VALIDATE_ONLY", "HERMES_MATCHED_VALIDATE_ONLY").replace(
        "RUN_LONG_CASE = True", "RUN_LONG_CASE = False").replace(
        "200 main stage answers plus four separate long-case answers if enabled", "200 matched-boundary answers")
    cell("markdown", INTRO)
    cell("code", config)
    cell("markdown", "## 1. Prepare pinned source, environment and model\n")
    cell("code", base.helpers())
    paths = list(ref["source_sha256"]) + ["data/streamingbench/streamingbench_realtime.json"]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for name in paths}
    assert all(hashes[name] == digest for name, digest in ref["source_sha256"].items())
    cell("code", "SOURCE_HASHES = " + repr(hashes) + "\n" + base.CLONE)
    cell("code", base.install())
    cell("code", base.old.MODEL.replace("WORK/'cache/huggingface/hub'", "Path(ENV['HF_HOME'])/'hub'"))

    cell("markdown", "## 2. Materialize trusted runner sources and audited reference data\n")
    parts = []
    for variable, name in SOURCES.items():
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "'''" not in source
        parts.extend([variable + " = r'''" + source + "'''\n", f"ast.parse({variable})\n",
                      f"assert hashlib.sha256({variable}.encode()).hexdigest() == {hashlib.sha256(source.encode()).hexdigest()!r}\n"])
    ref_json = json.dumps(ref, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    parts.extend(["REFERENCE_JSON = " + repr(ref_json) + "\n",
                  f"assert hashlib.sha256(REFERENCE_JSON.encode()).hexdigest() == {hashlib.sha256(ref_json.encode()).hexdigest()!r}\n"])
    embed = stage.EMBED.replace("('stage_runner.py',RUNNER_SOURCE),",
        "('stage_runner.py',RUNNER_SOURCE),('provenance/stage_base.py',STAGE_BASE_SOURCE),").replace(
        "            target=WORK/name\n", "            target=WORK/name\n            target.parent.mkdir(parents=True,exist_ok=True)\n")
    cell("code", "".join(parts) + embed)
    cells[-1]["metadata"]["jupyter"] = {"source_hidden": True}

    cell("markdown", "## 3. Validate and run the matched-boundary matrix\n")
    cell("code", stage.RUN.replace("'stage.log'", "'matched.log'"))
    cell("markdown", "## 4. View results and download the evidence ZIP\n"
         "If the run stops, run this cell manually to download the failure evidence.\n")
    cell("code", stage.FINISH.replace("HERMES_stage_attribution_evidence.zip", "HERMES_matched_boundary_evidence.zip").replace(
        "A matching rerun resumes checkpoints; failed setup must be fixed first.",
        "Time-budget stops can resume. Inspect failed gates before restarting."))
    return dict(cells=cells, metadata=dict(
        kernelspec=dict(display_name="Python 3 (ipykernel)", language="python", name="python3"),
        language_info=dict(name="python", version="3.12.0")), nbformat=4, nbformat_minor=5)


if __name__ == "__main__":
    value = notebook()
    DEST.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST} ({len(value['cells'])} cells)")
