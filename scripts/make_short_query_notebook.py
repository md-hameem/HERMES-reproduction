"""Build the short query notebook from the verified AutoDL setup templates."""
import hashlib
import json
from zipfile import ZipFile

import make_matched_boundary_notebook as matched
from validate_causal_pilot_evidence import check_manifest
from validate_matched_boundary_evidence import ARCHIVE, ARCHIVE_SHA

ROOT = matched.ROOT
DEST = ROOT / "notebooks/autodl/hermes-short-query-cache.ipynb"
INTRO = """# HERMES short query-cache control

Upload this notebook to the **same AutoDL container** and choose **Restart Kernel → Run All**.
Keep the defaults. It reuses the assets under `/root/autodl-tmp/hermes_causal_pilot` and
writes to `/root/autodl-tmp/hermes_short_query`. No additional visual inputs or annotations are needed.

| Arm | Recent representations | Answer cache |
|---|---|---|
| Full contextual | Encoded with older history | Older and recent slots visible |
| Masked contextual | Same contextual cache | Older slots retained but masked |
| Short contextual | Same contextual recent K/V | Older slots physically removed; recent logical positions preserved |
| Compact recent-four | Encoded without older history | Recent slots at ordinary compact positions |

The primary comparison is **short versus masked contextual**. The runner records actual
positions and physical lengths during question prefill and every decode call. It checks
FP32 reference agreement and saves the FP16 differences. A short-path choice change
is an outcome. All three reference arms must replay the prior matched run.

The same 40 development questions give **160 arm-level answers** and **320 timed calls**
with the default two interleaved repeats. Initial warmup is separate. The export records
query K/V bytes, answer latency and the one-time extraction cost separately. A shorter
query cache does not by itself reduce total live memory. This run is for feasibility
before confirmation on source-separated data.

Watch for `SHORT_QUERY_PREFLIGHT_PASS` and `QUESTION_BLOCK_DONE 1/40`. At
`SHORT_QUERY_COMPLETE`, download **HERMES_short_query_evidence.zip** and save the
executed notebook. If the run stops, the last cell can download available failure evidence.
"""


def reference_data():
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == ARCHIVE_SHA
    with ZipFile(ARCHIVE) as zipped:
        manifest, data = check_manifest(zipped)
    assert manifest["status"] == "complete"
    prior = json.loads(data["reference.json"])
    arms = {"A_history_both": "full_contextual", "C_hide_history_readout": "masked_contextual",
            "compact_recent4": "compact_recent4"}
    rows = []
    for name, payload in data.items():
        if name.startswith("checkpoints/main/"):
            row = json.loads(payload)["row"]
            if row["method"] in arms:
                rows.append(dict(sample_id=row["sample_id"], qa_index=row["qa_index"],
                                 arm=arms[row["method"]], choice=row["pred_choice"]))
    expected = {(v["sample_id"], qi, arm) for v in prior["videos"]
                for qi in range(len(v["conversations"])) for arm in arms.values()}
    assert len(rows) == len(expected) == 120
    assert {(r["sample_id"], r["qa_index"], r["arm"]) for r in rows} == expected
    return dict(videos=prior["videos"], source_sha256=prior["source_sha256"],
                baseline_choices=rows, matched_archive_sha256=ARCHIVE_SHA)


def notebook():
    result = matched.notebook()
    cells = result["cells"]
    for i, cell in enumerate(cells):
        cell["id"] = f"short-query-{i:02d}"

    def replace(i, source):
        if cells[i]["cell_type"] == "code":
            compile(source, cells[i]["id"], "exec")
        cells[i]["source"] = source.splitlines(keepends=True)

    replace(0, INTRO)
    config = "".join(cells[1]["source"]).replace("hermes_matched_boundary", "hermes_short_query").replace(
        "HERMES_MATCHED_VALIDATE_ONLY", "HERMES_SHORT_QUERY_VALIDATE_ONLY").replace(
        "RUN_LONG_CASE = False", "REPEATS = 2").replace("200 matched-boundary answers", "160 answers, 320 timed calls by default")
    replace(1, config)
    files = dict(matched.SOURCES, RUNNER_SOURCE="autodl_short_query.py", MATCHED_BASE_SOURCE="autodl_matched_boundary.py")
    parts = []
    for variable, name in files.items():
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "'''" not in source
        parts.extend([variable + " = r'''" + source + "'''\n", f"ast.parse({variable})\n",
                      f"assert hashlib.sha256({variable}.encode()).hexdigest() == {hashlib.sha256(source.encode()).hexdigest()!r}\n"])
    ref = json.dumps(reference_data(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    parts.append(f"REFERENCE_JSON = {ref!r}\nassert hashlib.sha256(REFERENCE_JSON.encode()).hexdigest() == {hashlib.sha256(ref.encode()).hexdigest()!r}\n")
    embed = matched.stage.EMBED.replace("('stage_runner.py',RUNNER_SOURCE),",
        "('stage_runner.py',RUNNER_SOURCE),('provenance/stage_base.py',STAGE_BASE_SOURCE),"
        "('provenance/matched_base.py',MATCHED_BASE_SOURCE),").replace(
        "            target=WORK/name\n", "            target=WORK/name\n            target.parent.mkdir(parents=True,exist_ok=True)\n")
    replace(8, "".join(parts) + embed)
    replace(9, "## 3. Validate positions and compare query caches\n")
    run = "".join(cells[10]["source"]).replace("long_case=RUN_LONG_CASE", "repeats=REPEATS").replace(
        "'--long-case',str(int(RUN_LONG_CASE))", "'--repeats',str(REPEATS)").replace("'matched.log'", "'short_query.log'")
    replace(10, run)
    replace(12, "".join(cells[12]["source"]).replace("HERMES_matched_boundary_evidence.zip", "HERMES_short_query_evidence.zip"))
    return result


if __name__ == "__main__":
    value = notebook()
    DEST.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST} ({len(value['cells'])} cells)")
