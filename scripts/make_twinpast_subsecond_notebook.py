"""Freeze the native-timestamp evidence capability check from the endpoint run."""
import ast
import json

from validate_twinpast_temporal_windows_evidence import ARCHIVE, MANIFEST, ROOT, TEMPLATE, digest, verify_inventory

DEST = ROOT / "notebooks/autodl/hermes-twinpast-subsecond.ipynb"
INTRO = """# TwinPast sub-second evidence check

Run this notebook on the same AutoDL RTX 4090 instance with **Restart Kernel → Run All**.
It reuses the existing 7B model and the temporal-window media cache.

This diagnostic covers the seven evaluated pairs with evidence annotations. It replays
the archived six uniform history frames and compares them with up to six frames decoded
at actual frame timestamps within the annotated intervals. If fewer than six qualify,
the remaining slots use frames from the archived uniform selection. Both arms use the
same four recent frames, the same prompts, and both option orders.

The evidence ZIP includes score records and contact sheets. Verify visible evidence in the
rendered frames before interpreting the result. Timestamp coverage alone does not prove
that the necessary event was observable.

Consider a new sampler only if the annotation-guided condition recovers at least two
robust pairs and the uniform reference replays within 0.05 absolute logit error with
the same predictions. Robust means both branches correct under both option orders.
This is a development capability control with privileged annotations.

After **TWINPAST_SUBSECOND_COMPLETE**, save the executed notebook and download
**HERMES_twinpast_subsecond_evidence.zip**.
"""


def notebook():
    evidence = verify_inventory(ARCHIVE)
    previous = json.loads(evidence["provenance/temporal_cases.json"])
    manifest = json.loads(MANIFEST.read_bytes())
    assert previous["manifest_sha256"] == digest(MANIFEST.read_bytes())
    accepted = json.loads(evidence["exports/run_config.json"])["accepted"]
    annotations = {a["pair_id"]: a for a in manifest["candidates"]}
    cases = []
    for pair in accepted:
        annotation = annotations[pair]
        intervals = annotation["evidence_intervals"]
        if not intervals or not all(intervals.get(b) for b in "AB"):
            continue
        block = json.loads(evidence[f"checkpoints/{pair}.json"])["block"]
        baseline = {b: dict(sample=t["sample"], selection=t["selection"]["uniform"],
                            scores={o: a["logits"]["uniform"] for o,a in t["answers"].items()})
                    for b,t in block["branches"].items()}
        cases.append(dict(annotation=annotation, baseline=baseline))
    ids = {c["annotation"]["pair_id"] for c in cases}
    assert len(ids) == len(cases) == 7
    spec = dict(model_revision=previous["model_revision"], video_processor=previous["video_processor"],
                media_source=previous["media_source"], source_archive_sha256=digest(ARCHIVE.read_bytes()),
                media_plan=[p for p in previous["media_plan"] if p[0]["pair_id"] in ids], cases=cases)
    book = json.loads(TEMPLATE.read_bytes())
    cells = book["cells"]

    def get(i):
        return "".join(cells[i]["source"])

    def put(i, value):
        cells[i]["source"] = value.splitlines(keepends=True)

    sources = ast.literal_eval(ast.parse(get(4)).body[0].value)
    assert all(s.encode("utf-8") == evidence[n] for n,s in sources.items())
    sources.pop("provenance/temporal_cases.json")
    sources.pop("provenance/evaluation_targets.json")
    sources["provenance/subsecond_cases.json"] = json.dumps(spec, ensure_ascii=False, indent=2) + "\n"
    sources["provenance/subsecond_runner.py"] = (ROOT / "scripts/autodl_twinpast_subsecond.py").read_text(encoding="utf-8")
    hashes = {n: digest(s.encode("utf-8")) for n,s in sources.items()}
    put(0, INTRO)
    put(1, get(1).replace("/hermes_twinpast_temporal_windows", "/hermes_twinpast_subsecond")
        .replace("HERMES_TWINPAST_TEMPORAL_VALIDATE_ONLY", "HERMES_TWINPAST_SUBSECOND_VALIDATE_ONLY")
        .replace("provenance/temporal_runner.py", "provenance/subsecond_runner.py")
        .replace("Up to 8 fresh HSR pairs", "7 pairs, 56 sub-second evidence score rows")
        .replace("WORK.with_name('hermes_twinpast_diagnostic')", "WORK.with_name('hermes_twinpast_temporal_windows')"))
    put(4, f"SOURCES = {sources!r}\nEXPECTED = {hashes!r}\n" + "\n".join(get(4).split("\n")[2:]))
    put(7, "## 2. Replay uniform history and score native-timestamp evidence\n")
    put(8, get(8).replace("HERMES_twinpast_temporal_windows_evidence.zip", "HERMES_twinpast_subsecond_evidence.zip")
        .replace("'twinpast_temporal_windows.log'", "'twinpast_subsecond.log'"))
    for i, cell in enumerate(cells):
        cell["id"] = f"twinpast-subsecond-{i:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    return book


if __name__ == "__main__":
    DEST.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST}")
