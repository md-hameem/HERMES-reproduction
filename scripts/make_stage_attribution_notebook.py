"""Build the self-contained stage-attribution AutoDL notebook."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import make_input_diagnostic_notebook as base
from validate_causal_pilot_evidence import check_manifest

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/"notebooks/autodl/hermes-stage-attribution.ipynb"


def reference_data():
    value=base.reference_data()
    input_archive=ROOT/"evidence/archives/HERMES_input_diagnostic_evidence.zip"
    context_archive=ROOT/"evidence/archives/HERMES_context_diagnostic_evidence.zip"
    assert hashlib.sha256(input_archive.read_bytes()).hexdigest()=="f184271bd8929f2ca5dead71a1ec60e4e1b555f33f2ed66d9909e3d7e0ffa1e8"
    assert hashlib.sha256(context_archive.read_bytes()).hexdigest()=="fee3165e323c0f7106688a37cb185ec390d4f831e9debdfbf0352bd72881b774"
    with ZipFile(input_archive) as z:_,data=check_manifest(z)
    value["input_rows"]=[json.loads(blob)["row"] for name,blob in data.items() if name.startswith("checkpoints/")]
    assert len(value["input_rows"])==200
    with ZipFile(context_archive) as z:_,data=check_manifest(z)
    value["long_reference_choice"]=json.loads(data["checkpoints/full_prefix_original/sample_204_q3.json"])["row"]["pred_choice"]
    value["input_archive_sha256"]=hashlib.sha256(input_archive.read_bytes()).hexdigest()
    value["context_archive_sha256"]=hashlib.sha256(context_archive.read_bytes()).hexdigest()
    return value


INTRO="""# HERMES stage attribution — recent construction versus history readout

Upload **this notebook alone**, preferably into your existing AutoDL container, then **Restart Kernel → Run All**. Defaults need no edits. It reuses verified assets under `/root/autodl-tmp/hermes_causal_pilot` and writes to `/root/autodl-tmp/hermes_stage_attribution_v2`, preserving the first run. A fresh container is supported with extra setup/download/feature preparation.

**Main experiment: 40 questions × five conditions = 200 answers.** Each input uses the last up to 20 original sampled frames: older H (up to 16) plus recent R (last up to four). All answers reset state and use no eviction, past-answer guidance or gold-dependent policy.

| Arm | H visible while R is encoded? | H directly visible during question/answer? |
|---|---|---|
| A | Yes | Yes |
| B | No | Yes |
| C | Yes | No |
| D | No | No |
| Compact recent-4 | No H present | No H present; ordinary compact positions |

A–D retain identical physical H/R cache slots and logical positions. B/D hide H during R encoding, validated against physically independent R construction at the same RoPE positions. C/D mask H at **every layer for question prefill and every generated token**. C can still carry history indirectly through R. Compact recent-4 checks the position/sink-distance difference. Questions with no H are null controls, not extra independent history interventions.

The default also runs **four separate A–D answers for 204/q3's longer prefix**, kept separate from the main table. Set `RUN_LONG_CASE=False` before a new run if you want only the 200-answer matrix. This is a bounded diagnostic/existing-mechanism comparator, not a new method or independent confirmation. B is not a complete DSCache reproduction.

**Mandatory gates:** feature/native-attention preflight; explicit-mask versus native logits; independent-construction K/V agreement at fixed positions; masked readout versus physically removed H through prefill and two decode steps; zero hidden-H/future attention at all layers; source-cache preservation. A and compact must replay their earlier choices on every question or the experiment stops and exports failure evidence. Do not bypass a mismatch by rerunning.

The two-frame stage preflight checks mask and position equivalence with FP32 math attention, restores the original model precision, then repeats the comparisons in FP16. The evidence ZIP includes per-layer K/V errors in `exports/stage_preflight_diagnostics.json`.

Recommended: **RTX 4090 / 24 GB**, CUDA 12.8-compatible driver, 18 GiB free for fresh setup. Same pinned 0.5B model, source sampling, prompt/parser and greedy 16-token policy as before. The kernel never imports torch. Shared A/B/compact cache construction is recorded separately; held control snapshots and transient buffers are included in observed allocation, so per-answer times are not standalone algorithm efficiency.

Setup and inference have stage output, per-arm results, progress JSON and 30-second liveness messages. `MAX_HOURS=1.0` is a soft post-model budget checked between question blocks, excluding setup/download/model loading/final packaging; it is not a runtime estimate. Rerun with the same WORK/settings to resume validated checkpoints; change WORK for changed code or long-case settings.

At completion download **`HERMES_stage_attribution_evidence.zip`** and save the executed notebook. It includes source/config/feature provenance, all masks/positions/groups, preflight metrics, checkpoints, paired summaries and closed logs. Weights, videos and feature tensors are excluded. Keep those assets. Local CPU tests cannot certify a complete GPU execution; failed gates stop QA.
"""


def config():
    value=base.CONFIG.replace("hermes_input_diagnostic","hermes_stage_attribution").replace("HERMES_DIAGNOSTIC_VALIDATE_ONLY","HERMES_STAGE_VALIDATE_ONLY")
    value=value.replace("RUNNER = WORK/'diagnostic_runner.py'","RUNNER = WORK/'stage_runner.py'")
    value=value.replace("WORK = Path('/root/autodl-tmp/hermes_stage_attribution')","WORK = Path('/root/autodl-tmp/hermes_stage_attribution_v2')")
    value=value.replace("RUN_NATIVE = True","RUN_LONG_CASE = True")
    value=value.replace("NATIVE_MAX_VISUAL_TOKENS = 8192  # no truncation: larger prefixes are skipped explicitly\n","")
    value=value.replace("MAKE_REVIEW_PAGES = True\n","")
    value=value.replace("SUPPORT_ITEMS = []              # optional reviewed cases, see schema in the next Markdown cell\n","")
    value=value.replace("assert MAX_HOURS >= 0 and NATIVE_MAX_VISUAL_TOKENS >= 196","assert MAX_HOURS >= 0")
    value=value.replace("print('Plan: 8 videos, 40 questions, 120 mandatory answers; optional native prefix and verified support')","print('Plan: 200 main stage answers plus four separate long-case answers if enabled')")
    return value


EMBED='''if not VALIDATE_ONLY:
    import fcntl
    session_lock=(WORK/'.notebook.lock').open('a')
    try:
        fcntl.flock(session_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name,source in [('stage_runner.py',RUNNER_SOURCE),('stage_context_utils.py',CONTEXT_SOURCE),('context_input_utils.py',INPUT_SOURCE),('diagnostic_pilot_utils.py',UTILS_SOURCE)]:
            target=WORK/name
            if target.exists() and target.read_text(encoding='utf-8')!=source:
                raise RuntimeError('Embedded source changed: use a new WORK directory')
            target.write_text(source,encoding='utf-8',newline='\\n')
        target=WORK/'reference.json'
        if target.exists() and target.read_text(encoding='utf-8')!=REFERENCE_JSON:
            raise RuntimeError('Reference changed: use a new WORK directory')
        target.write_text(REFERENCE_JSON,encoding='utf-8',newline='\\n')
    except Exception:
        session_lock.close()
        raise
print('EMBEDDED_STAGE_SOURCES_PASS')
'''

RUN='''if not VALIDATE_ONLY:
    (WORK/'launch.json').write_text(json.dumps(dict(started_utc=datetime.now(timezone.utc).isoformat(),
        repository_revision=BASE_REVISION,runner_sha256=hashlib.sha256(RUNNER_SOURCE.encode()).hexdigest(),
        reuse_candidate=str(REUSE),max_hours=MAX_HOURS,long_case=RUN_LONG_CASE,gpu_driver=smi.stdout.strip()),indent=2),encoding='utf-8')
    try:
        logged([PYTHON,'-u',RUNNER,'--repo',REPO,'--work',WORK,'--model',MODEL_PATH,'--reuse',REUSE,
                '--long-case',str(int(RUN_LONG_CASE)),'--max-hours',str(MAX_HOURS)],'stage.log',ENV,REPO)
    finally:
        try:
            if (WORK/'exports/completion.json').is_file():
                repack="import sys,json; from pathlib import Path; sys.path.insert(0,sys.argv[1]); import stage_runner as s; w=Path(sys.argv[1]); s.package(w,json.loads((w/'exports/completion.json').read_text())['status'])"
                result=probe([PYTHON,'-c',repack,WORK],ENV,REPO)
                print(result.stdout,flush=True)
                assert result.returncode==0,result.stderr[-2000:]
        finally:
            fcntl.flock(session_lock,fcntl.LOCK_UN)
            session_lock.close()
else:
    print('VALIDATION_PASS: every stage notebook cell executed; network/install/GPU and filesystem mutations skipped')
'''

FINISH=base.FINISH.replace("HERMES_input_diagnostic_evidence.zip","HERMES_stage_attribution_evidence.zip")
FINISH=FINISH.replace("'| Mandatory answers:', completion.get('mandatory_answered', 'see checkpoints')","'| Main answers:', completion.get('main_answers', 'see checkpoints')")


def main():
    reference=reference_data();cells=[]
    def cell(kind,source):
        value=dict(cell_type=kind,id=f"stage-{len(cells):02d}",metadata={},source=source.splitlines(keepends=True))
        if kind=="code":compile(source,f"stage_cell_{len(cells)}","exec");value.update(execution_count=None,outputs=[])
        cells.append(value)
    cell("markdown",INTRO);cell("code",config())
    cell("markdown","## 1. Pinned source, network, environment and model\nExisting assets are verified before reuse. Missing assets are fetched/prepared separately.\n")
    cell("code",base.helpers())
    paths=list(reference["source_sha256"])+["data/streamingbench/streamingbench_realtime.json"]
    hashes={name:hashlib.sha256((ROOT/name).read_bytes().replace(b"\r\n",b"\n")).hexdigest() for name in paths}
    assert all(hashes[n]==v for n,v in reference["source_sha256"].items())
    cell("code","SOURCE_HASHES = "+repr(hashes)+"\n"+base.CLONE)
    cell("code",base.install());cell("code",base.old.MODEL.replace("WORK/'cache/huggingface/hub'","Path(ENV['HF_HOME'])/'hub'"))
    cell("markdown","## 2. Embedded runner and immutable references\nSource modules come from this tested checkout; uploaded evidence supplies comparison data only.\n")
    text=""
    for variable,name in [("RUNNER_SOURCE","autodl_stage_attribution.py"),("CONTEXT_SOURCE","autodl_context_diagnostic.py"),
                          ("INPUT_SOURCE","autodl_input_diagnostic.py"),("UTILS_SOURCE","autodl_failure_pilot.py")]:
        source=(ROOT/"scripts"/name).read_text(encoding="utf-8").replace("\r\n","\n")
        assert "'''" not in source
        text+=variable+" = r'''"+source+"'''\nast.parse("+variable+")\n"
        text+=f"assert hashlib.sha256({variable}.encode()).hexdigest() == {hashlib.sha256(source.encode()).hexdigest()!r}\n"
    ref=json.dumps(reference,ensure_ascii=False,sort_keys=True,separators=(",",":"))
    text+="REFERENCE_JSON = "+repr(ref)+"\n"+f"assert hashlib.sha256(REFERENCE_JSON.encode()).hexdigest() == {hashlib.sha256(ref.encode()).hexdigest()!r}\n"
    cell("code",text+EMBED);cells[-1]["metadata"]["jupyter"]={"source_hidden":True}
    cell("markdown","## 3. GPU gates, 200 main answers and optional separate long case\nMasks cover all layers and every decode call. Stop on a failed preflight or historical replay mismatch. Liveness is not proof of successful progress.\n")
    cell("code",RUN)
    cell("markdown","## 4. Download evidence\nAfter a runner error stops Run All, run this final cell manually if failure evidence exists. Earlier setup errors are in WORK/logs.\n")
    cell("code",FINISH)
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name="Python 3 (ipykernel)",language="python",name="python3"),
        language_info=dict(name="python",version="3.12.0")),nbformat=4,nbformat_minor=5)
    DEST.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",encoding="utf-8",newline="\n")
    print(f"Wrote {DEST} ({len(cells)} cells; {DEST.stat().st_size/1024:.0f} KiB)")


if __name__=="__main__":main()
