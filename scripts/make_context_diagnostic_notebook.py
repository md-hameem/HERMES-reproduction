"""Generate an upload-alone context intervention notebook from tested source modules."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import make_input_diagnostic_notebook as base
import autodl_context_diagnostic as runner

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/"notebooks/autodl/hermes-context-diagnostic.ipynb"


def reference_data():
    value = base.reference_data()
    selected = {sid for sid,qi in runner.CASES}
    value["videos"] = [v for v in value["videos"] if v["sample_id"] in selected]
    value["predictions"] = [r for r in value["predictions"] if int(r["sample_id"]) in selected]
    archive = ROOT/"evidence/archives/HERMES_input_diagnostic_evidence.zip"
    expected = "f184271bd8929f2ca5dead71a1ec60e4e1b555f33f2ed66d9909e3d7e0ffa1e8"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected
    from validate_causal_pilot_evidence import check_manifest
    with ZipFile(archive) as z:
        _,payload = check_manifest(z)
    value["input_archive_sha256"] = expected
    value["input_rows"] = [json.loads(data)["row"] for name,data in payload.items() if name.startswith("checkpoints/")
                           and (json.loads(data)["row"]["sample_id"],json.loads(data)["row"]["qa_index"]) in runner.CASES]
    assert len(value["input_rows"]) == 35
    return value


INTRO = """# HERMES context/content diagnostic — AutoDL

Upload **this notebook alone**, then **Restart Kernel → Run All**. Defaults need no edits. Prefer the same container: it reuses the verified environment, videos and feature tensors from `/root/autodl-tmp/hermes_causal_pilot`. Results go to the new `/root/autodl-tmp/hermes_context_diagnostic` directory. A fresh container is supported with extra setup/download/vision work. No earlier experimental files are overwritten.

**Question:** do earlier visual inputs change a usable recent answer because of their content, or are positions/chunking/numerics responsible? This is an outcome-selected diagnostic, not a new method, independent benchmark, or publication-quality gain claim.

**Seven cases:** 204/q0, 203/q2, 204/q3 (recoveries), 209/q0 and 208/q4 (recent-window regressions), 202/q0 and 208/q0 (unrecovered controls). The screen has **12 arms × 7 cases = 84 case records**. Under the default 16,384-visual-token cap it expects up to 74 answers: five unavailable support candidates and five too-long full-prefix arms are explicit skips; actual context/VRAM limits can add skips. The first three replay controls must answer all seven cases for complete status.

Arms: question-only; recent-4/recent-20/full prefix with original chunk boundaries; full prefix framewise; recent-20/full prefix with fixed four-frame batches; and same-length earlier-context replacements using either the already available opening frame or the first of the retained recent four frames as repeated filler. Replacement arms preserve the final four source frames and their token positions exactly. Filler repetition/content changes are confounds; neither filler is assumed neutral. When a prefix has only four frames, replacement is an identical-input control.

Two **candidate-support** probes use 204/q0 sampled indices [6,7] and 204/q3 [69], based on saved-sheet inspection. These are analyst-selected diagnostic candidates, **not certified human oracles**. Original-resolution candidate frames and complete original sampled-prefix pages for all seven cases are exported for review. No invented support is supplied for the other five cases.

**Recommended:** RTX 4090 / 24 GB, CUDA 12.8-compatible driver, 18 GiB free for fresh setup. Same pinned 0.5B model, 0.5-fps source frames, prompt/parser and greedy 16-token answers. Every arm resets state with no compression or past-answer history. All natural/counterfactual source frames are strictly within the original query prefix; repeated input slots are logged explicitly.

Stage/per-arm output and a 30-second heartbeat show liveness. Check `WORK/progress.json` from another terminal. Default `MAX_HOURS=0.75` is a soft post-model guard checked between case blocks; it is not a runtime forecast and excludes setup/download/model load. Rerun All with the same settings/WORK to resume validated checkpoints. Change WORK when changing code or the visual-token cap. No total runtime is guaranteed.

Download **`HERMES_context_diagnostic_evidence.zip`** and save the executed notebook at the end. The ZIP includes all arm maps, chunk groups, statuses, replay/contrast tables, sources/provenance, logs, original review pages and candidate images; it excludes weights/videos/features. Preserve the existing asset cache. Local CPU checks do not replace the mandatory GPU preflight or establish a complete AutoDL run.
"""


def config():
    value = base.CONFIG.replace("hermes_input_diagnostic","hermes_context_diagnostic")
    value = value.replace("HERMES_DIAGNOSTIC_VALIDATE_ONLY","HERMES_CONTEXT_VALIDATE_ONLY")
    value = value.replace("RUNNER = WORK/'diagnostic_runner.py'","RUNNER = WORK/'context_runner.py'")
    value = value.replace("RUN_NATIVE = True\n", "").replace("MAKE_REVIEW_PAGES = True\n", "")
    value = value.replace("NATIVE_MAX_VISUAL_TOKENS = 8192  # no truncation: larger prefixes are skipped explicitly", "MAX_VISUAL_TOKENS = 16384  # no truncation; larger full-prefix arms are explicitly skipped")
    value = value.replace("MAX_HOURS = 1.0", "MAX_HOURS = 0.75")
    value = value.replace("SUPPORT_ITEMS = []              # optional reviewed cases, see schema in the next Markdown cell\n", "")
    value = value.replace("NATIVE_MAX_VISUAL_TOKENS >= 196","MAX_VISUAL_TOKENS >= 3920")
    value = value.replace("print('Plan: 8 videos, 40 questions, 120 mandatory answers; optional native prefix and verified support')", "print('Plan: 7 selected cases x 12 arms = 84 records; up to 74 default answers plus explicit skips')")
    return value


EMBED = '''if not VALIDATE_ONLY:
    import fcntl
    session_lock = (WORK/'.notebook.lock').open('a')
    try:
        fcntl.flock(session_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for filename, source in [('context_runner.py', RUNNER_SOURCE), ('context_input_utils.py', INPUT_SOURCE), ('diagnostic_pilot_utils.py', UTILS_SOURCE)]:
            target = WORK/filename
            if target.exists() and target.read_text(encoding='utf-8') != source:
                raise RuntimeError('Embedded source changed: use a new WORK directory')
            target.write_text(source, encoding='utf-8', newline='\\n')
        target = WORK/'reference.json'
        if target.exists() and target.read_text(encoding='utf-8') != REFERENCE_JSON:
            raise RuntimeError('Reference changed: use a new WORK directory')
        target.write_text(REFERENCE_JSON, encoding='utf-8', newline='\\n')
    except Exception:
        session_lock.close()
        raise
print('EMBEDDED_CONTEXT_SOURCES_PASS')
'''

RUN = '''if not VALIDATE_ONLY:
    (WORK/'launch.json').write_text(json.dumps(dict(started_utc=datetime.now(timezone.utc).isoformat(),
        repository_revision=BASE_REVISION, runner_sha256=hashlib.sha256(RUNNER_SOURCE.encode()).hexdigest(),
        reuse_candidate=str(REUSE), max_hours=MAX_HOURS, max_visual_tokens=MAX_VISUAL_TOKENS,
        gpu_driver=smi.stdout.strip()), indent=2), encoding='utf-8')
    try:
        logged([PYTHON, '-u', RUNNER, '--repo', REPO, '--work', WORK, '--model', MODEL_PATH,
                '--reuse', REUSE, '--max-visual-tokens', str(MAX_VISUAL_TOKENS), '--max-hours', str(MAX_HOURS)], 'context.log', ENV, REPO)
    finally:
        try:
            if (WORK/'exports/completion.json').is_file():
                repack = "import sys,json; from pathlib import Path; sys.path.insert(0,sys.argv[1]); import context_runner as c; w=Path(sys.argv[1]); c.package(w,json.loads((w/'exports/completion.json').read_text())['status'])"
                result = probe([PYTHON, '-c', repack, WORK], ENV, REPO)
                print(result.stdout, flush=True)
                assert result.returncode == 0, result.stderr[-2000:]
        finally:
            fcntl.flock(session_lock, fcntl.LOCK_UN)
            session_lock.close()
else:
    print('VALIDATION_PASS: every context notebook cell executed; no network/install/GPU or filesystem mutation')
'''

FINISH = base.FINISH.replace("HERMES_input_diagnostic_evidence.zip","HERMES_context_diagnostic_evidence.zip")
FINISH = FINISH.replace("'| Mandatory answers:', completion.get('mandatory_answered', 'see checkpoints')", "'| Answered:', completion.get('answered', 'see checkpoints')")


def main():
    reference = reference_data()
    sources = {"RUNNER_SOURCE": "scripts/autodl_context_diagnostic.py", "INPUT_SOURCE": "scripts/autodl_input_diagnostic.py", "UTILS_SOURCE": "scripts/autodl_failure_pilot.py"}
    cells = []
    def cell(kind,source):
        value = dict(cell_type=kind,id=f"context-{len(cells):02d}",metadata={},source=source.splitlines(keepends=True))
        if kind == "code":
            compile(source,f"context_cell_{len(cells)}","exec")
            value.update(execution_count=None,outputs=[])
        cells.append(value)
    cell("markdown",INTRO)
    cell("code",config())
    cell("markdown","## 1. Pinned source, network, environment and model\nReuse is verified; missing assets are downloaded/prepared separately. Setup and run subprocesses report liveness while silent.\n")
    cell("code",base.helpers())
    paths = list(reference["source_sha256"])+["data/streamingbench/streamingbench_realtime.json"]
    hashes = {name:hashlib.sha256((ROOT/name).read_bytes().replace(b"\r\n",b"\n")).hexdigest() for name in paths}
    assert all(hashes[name] == expected for name,expected in reference["source_sha256"].items())
    cell("code","SOURCE_HASHES = "+repr(hashes)+"\n"+base.CLONE)
    cell("code",base.install())
    cell("code",base.old.MODEL.replace("WORK/'cache/huggingface/hub'","Path(ENV['HF_HOME'])/'hub'"))
    cell("markdown","## 2. Embedded source and trusted historical reference\nHistorical code is never executed from the uploaded ZIP; only verified comparison data is embedded. Candidate support is explicitly unconfirmed for publication.\n")
    text = ""
    for variable,path in sources.items():
        source = (ROOT/path).read_text(encoding="utf-8").replace("\r\n","\n")
        assert "'''" not in source
        text += variable+" = r'''"+source+"'''\nast.parse("+variable+")\n"
        text += f"assert hashlib.sha256({variable}.encode()).hexdigest() == {hashlib.sha256(source.encode()).hexdigest()!r}\n"
    reference_json = json.dumps(reference,ensure_ascii=False,sort_keys=True,separators=(",",":"))
    text += "REFERENCE_JSON = "+repr(reference_json)+"\n"
    text += f"assert hashlib.sha256(REFERENCE_JSON.encode()).hexdigest() == {hashlib.sha256(reference_json.encode()).hexdigest()!r}\n"
    cell("code",text+EMBED)
    cells[-1]["metadata"]["jupyter"] = {"source_hidden":True}
    cell("markdown","## 3. Run seven case blocks\nAll answers reset state. Exact-content chunking controls come before semantic interpretation. The report flags historical replay mismatches; repeated fillers and candidate support are diagnostic privileges, not proposed methods.\n")
    cell("code",RUN)
    cell("markdown","## 4. Download evidence\nSave this ZIP and the executed notebook. If a runner failure stopped Run All, execute this cell manually when failure evidence exists. Setup failures can precede archive creation; inspect WORK/logs.\n")
    cell("code",FINISH)
    value = dict(cells=cells,metadata=dict(kernelspec=dict(display_name="Python 3 (ipykernel)",language="python",name="python3"),
                 language_info=dict(name="python",version="3.12.0")),nbformat=4,nbformat_minor=5)
    DEST.write_text(json.dumps(value,ensure_ascii=False,indent=1)+"\n",encoding="utf-8",newline="\n")
    print(f"Wrote {DEST} ({len(cells)} cells; {DEST.stat().st_size/1024:.0f} KiB)")


if __name__ == "__main__":
    main()
