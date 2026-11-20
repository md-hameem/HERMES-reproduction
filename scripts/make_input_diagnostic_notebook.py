"""Build an upload-alone notebook without executing historical evidence code."""
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import make_failure_pilot_notebook as old

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "notebooks/autodl/hermes-input-diagnostic.ipynb"
EVIDENCE_SHA = "af4b7392e90bd3f433ad20182a9058d8c8027a8ede936ed5d759572be04e0361"


def reference_data():
    archive = ROOT/"evidence/archives/HERMES_causal_scorer_pilot_evidence.zip"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == EVIDENCE_SHA
    with ZipFile(archive) as z:
        manifest = json.loads(z.read("evidence_manifest.json"))
        for item in manifest["files"]:
            blob = z.read(item["path"])
            assert len(blob) == item["bytes"] and hashlib.sha256(blob).hexdigest() == item["sha256"]
        config = json.loads(z.read("exports/run_config.json"))
        videos = json.loads(z.read("exports/replay_annotations.json"))
        for video in videos:
            sid = int(Path(video["video_path"]).stem.split("_")[1])
            video["sample_id"] = sid
            video["video_sha256"] = config["assets"][str(sid)]
            video["feature_meta"] = json.loads(z.read(f"features/sample_{sid}/complete.json"))
            video["timestamps"] = list(csv.DictReader(io.StringIO(z.read(f"features/sample_{sid}/sampled_timestamps.csv").decode())))
        return dict(archive_sha256=EVIDENCE_SHA, source_sha256=config["source_sha256"], videos=videos,
                    predictions=list(csv.DictReader(io.StringIO(z.read("exports/predictions.csv").decode()))))


INTRO = """# HERMES input/capability diagnostic — AutoDL

Upload **this notebook alone**, then **Restart Kernel → Run All**. Defaults need no edits. It runs a separate diagnostic on the **same eight videos and 40 questions** as the completed scorer pilot. No new method or acceptance claim is implied.

**Mandatory:** question-only, recent 4 frames and recent 20 frames: **120 new answers**. Each question starts from a fresh initial prompt, with no past answers and no compression. **Optional by default enabled:** native uncompressed sampled prefix, limited to 8,192 visual tokens and checked model context/VRAM; infeasible cases are explicitly skipped, never truncated. Native and recent controls change input/history as well as compression; they do not isolate eviction policy by themselves.

**Recommended:** RTX 4090 / 24 GB, CUDA 12.8-compatible driver, 18 GiB free for a fresh container. The kernel never imports torch; one isolated subprocess owns the GPU. Uses the same pinned 0.5B model, inference checkout, source sampling, prompt, parser and greedy 16-token answer policy as the previous pilot.

**Reuse:** if `/root/autodl-tmp/hermes_causal_pilot` survives, the notebook checks its environment and reuses exact SHA-verified videos/features. It writes results to a separate directory. If those assets are unavailable, it downloads the eight videos and prepares features. Rejected old features are regenerated separately. The old notebook/results are not overwritten. Imported features are tied to the archived manifest; regenerated features have explicit new provenance.

**Progress:** stage output, per-question results, and a 30-second runner heartbeat; setup subprocesses also report liveness while silent. `WORK/progress.json` is readable from a second terminal. A heartbeat means alive, not guaranteed useful progress. The default one-hour soft experiment limit is checked between video blocks (after the first block); setup/downloads/model load are excluded. Rerun All to resume a matching configuration. No runtime guarantee is made.

**Support review:** the ZIP includes complete original sampled-prefix pages for seven priority questions and a pending review table. No support is guessed automatically. Verified-support oracle runs are disabled until you supply reviewed sampled-frame indices in `SUPPORT_ITEMS`; these are privileged diagnostic inputs. To add support or change experimental settings after a run, use a **new WORK directory**. The old directory can remain the read-only asset source.

At the end download **`HERMES_input_diagnostic_evidence.zip`** and save the executed notebook. The ZIP excludes videos, model weights and feature tensors. Save the container assets if you want to reuse them. Local CPU tests and headless cell validation cannot prove a fresh AutoDL GPU execution; mandatory GPU preflight stops QA on failure.
"""

CONFIG = '''from pathlib import Path
import os, sys, subprocess, json, hashlib, shutil, time, zipfile, ast
from datetime import datetime, timezone

WORK = Path('/root/autodl-tmp/hermes_input_diagnostic')
REUSE = Path('/root/autodl-tmp/hermes_causal_pilot')
RUN_NATIVE = True
NATIVE_MAX_VISUAL_TOKENS = 8192  # no truncation: larger prefixes are skipped explicitly
MAKE_REVIEW_PAGES = True
MAX_HOURS = 1.0                 # soft post-model budget; 0 disables; rerun resumes
SUPPORT_ITEMS = []              # optional reviewed cases, see schema in the next Markdown cell
REPO_URL = 'https://github.com/md-hameem/HERMES-reproduction.git'
BASE_REVISION = '2a1385e0abc94debb64761b0465713c799a81c1c'
MODEL_ID = 'llava-hf/llava-onevision-qwen2-0.5b-ov-hf'
MODEL_SHA = '74dd0bf867a4cda7950c17663794267c60cf4b40'
TF_SHA = '66bc4def9505fa7c7fe4aa7a248c34a026bb552b'
VALIDATE_ONLY = os.environ.get('HERMES_DIAGNOSTIC_VALIDATE_ONLY') == '1'
assert VALIDATE_ONLY or (WORK.is_absolute() and REUSE.is_absolute())
assert WORK != REUSE and not WORK.is_relative_to(REUSE) and not REUSE.is_relative_to(WORK)
assert MAX_HOURS >= 0 and NATIVE_MAX_VISUAL_TOKENS >= 196
REPO = WORK/'repository'
VENV = WORK/'venv'
PYTHON = VENV/'bin/python'
RUNNER = WORK/'diagnostic_runner.py'
print('Plan: 8 videos, 40 questions, 120 mandatory answers; optional native prefix and verified support')
print('WORK:', WORK, '\\nReuse candidate:', REUSE)
print('Mode:', 'CPU notebook validation (no network/GPU)' if VALIDATE_ONLY else 'AutoDL execution')
'''

LOGGED = '''def logged(command, name, env=None, cwd=None):
    import queue, threading
    logfile = WORK/'logs'/name
    logfile.parent.mkdir(parents=True, exist_ok=True)
    print('SETUP_STAGE', name, flush=True)
    with logfile.open('a', encoding='utf-8') as handle:
        proc = subprocess.Popen([str(x) for x in command], env=env, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1)
        lines = queue.Queue()
        def collect():
            try:
                for line in proc.stdout:
                    lines.put(line)
            finally:
                lines.put(None)
        reader = threading.Thread(target=collect, daemon=True)
        reader.start()
        started = time.monotonic()
        while True:
            try:
                line = lines.get(timeout=30)
            except queue.Empty:
                line = f'COMMAND_ALIVE log={name} elapsed_min={(time.monotonic()-started)/60:.1f} (liveness only)\\n'
            if line is None:
                break
            print(line, end='', flush=True)
            handle.write(line)
            handle.flush()
        reader.join(timeout=5)
        status = proc.wait()
        proc.stdout.close()
    if status:
        raise RuntimeError(f'Command failed ({status}); see {logfile}. Failure evidence, if created, stays in WORK.')

'''

def helpers():
    source = old.HELPERS
    begin, end = source.index('def logged('), source.index('def probe(')
    source = source[:begin] + LOGGED + source[end:]
    # Prefer persistent shared download caches without altering old execution artifacts.
    source = source.replace("HF_HOME=str(WORK/'cache/huggingface')", "HF_HOME=str((REUSE if (REUSE/'cache/huggingface').is_dir() else WORK)/'cache/huggingface')")
    return source


CLONE = '''if not VALIDATE_ONLY:
    if not (REPO/'.git').is_dir():
        assert not REPO.exists() or not list(REPO.iterdir()), 'Non-Git source directory has files; use a new WORK'
        REPO.mkdir(parents=True, exist_ok=True)
        logged(['git', 'init', REPO], 'clone.log', ENV)
        logged(['git', 'remote', 'add', 'origin', REPO_URL], 'clone.log', ENV, REPO)
    remote = probe(['git', 'remote', 'get-url', 'origin'], ENV, REPO)
    assert remote.returncode == 0 and remote.stdout.strip() == REPO_URL, 'Unexpected source repository origin'
    dirty = probe(['git', 'status', '--porcelain', '--untracked-files=no'], ENV, REPO)
    assert dirty.returncode == 0 and not dirty.stdout.strip(), 'Pinned checkout has local edits; preserve them and use a new WORK'
    if probe(['git', 'cat-file', '-e', BASE_REVISION+'^{commit}'], ENV, REPO).returncode:
        for attempt in range(4):
            try:
                logged(['git', 'fetch', '--filter=blob:none', '--depth', '1', 'origin', BASE_REVISION], 'clone.log', ENV, REPO)
                break
            except RuntimeError:
                if attempt == 3:
                    raise
                print('GITHUB_FETCH_RETRY', attempt+1, flush=True)
                time.sleep(3*(attempt+1))
    logged(['git', 'sparse-checkout', 'init', '--cone'], 'clone.log', ENV, REPO)
    logged(['git', 'sparse-checkout', 'set', 'inference', 'video_qa', 'scripts', 'data/streamingbench'], 'clone.log', ENV, REPO)
    for attempt in range(4):
        try:
            logged(['git', 'checkout', '--detach', BASE_REVISION], 'clone.log', ENV, REPO)
            break
        except RuntimeError:
            if attempt == 3:
                raise
            print('GITHUB_CHECKOUT_RETRY', attempt+1, flush=True)
            time.sleep(3*(attempt+1))
    assert probe(['git', 'rev-parse', 'HEAD'], ENV, REPO).stdout.strip() == BASE_REVISION
    for rel, expected in SOURCE_HASHES.items():
        assert hashlib.sha256((REPO/rel).read_bytes().replace(b'\\r\\n', b'\\n')).hexdigest() == expected, 'Pinned source mismatch: '+rel
    print('PINNED_SOURCE_PASS', BASE_REVISION, flush=True)
'''


def install():
    source = old.INSTALL
    return source.replace("    PIP_ENV = package_environment(ENV)", """    candidate_python = REUSE/'venv/bin/python'
    if candidate_python.is_file() and probe([candidate_python, '-c', ENV_PROBE], ENV).returncode == 0:
        PYTHON = candidate_python
        print('REUSING_VERIFIED_PINNED_ENVIRONMENT', PYTHON, flush=True)
    PIP_ENV = package_environment(ENV)""")


EMBED = '''if not VALIDATE_ONLY:
    # Do not overwrite embedded source/reference while another notebook run is active.
    import fcntl
    session_lock = (WORK/'.notebook.lock').open('a')
    try:
        fcntl.flock(session_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        session_lock.close()
        raise RuntimeError('Another run owns WORK. Wait for it or choose a different WORK directory.')
    for filename, source in [('diagnostic_runner.py', RUNNER_SOURCE), ('diagnostic_pilot_utils.py', UTILS_SOURCE)]:
        target = WORK/filename
        if target.exists() and target.read_text(encoding='utf-8') != source:
            raise RuntimeError('Embedded source differs from previous run: use a new WORK directory')
        target.write_text(source, encoding='utf-8', newline='\\n')
    reference_path = WORK/'reference.json'
    if reference_path.exists() and reference_path.read_text(encoding='utf-8') != REFERENCE_JSON:
        raise RuntimeError('Reference differs from previous run: use a new WORK directory')
    reference_path.write_text(REFERENCE_JSON, encoding='utf-8', newline='\\n')
    support_path = WORK/'support_oracle.json'
    support_text = json.dumps(SUPPORT_ITEMS, sort_keys=True)
    if support_path.exists() and json.loads(support_path.read_text()) != SUPPORT_ITEMS:
        raise RuntimeError('Support annotations changed: choose a new WORK directory')
    support_path.write_text(support_text, encoding='utf-8')
print('EMBEDDED_SOURCES_AND_REFERENCE_PASS')
'''

RUN = '''if not VALIDATE_ONLY:
    (WORK/'launch.json').write_text(json.dumps(dict(started_utc=datetime.now(timezone.utc).isoformat(),
        repository_revision=BASE_REVISION, runner_sha256=hashlib.sha256(RUNNER_SOURCE.encode()).hexdigest(),
        reuse_candidate=str(REUSE), max_hours=MAX_HOURS, native=RUN_NATIVE,
        native_max_visual_tokens=NATIVE_MAX_VISUAL_TOKENS, gpu_driver=smi.stdout.strip()), indent=2), encoding='utf-8')
    try:
        logged([PYTHON, '-u', RUNNER, '--repo', REPO, '--work', WORK, '--model', MODEL_PATH,
                '--reuse', REUSE, '--native', str(int(RUN_NATIVE)), '--native-max-tokens', str(NATIVE_MAX_VISUAL_TOKENS),
                '--review', str(int(MAKE_REVIEW_PAGES)), '--max-hours', str(MAX_HOURS)], 'diagnostic.log', ENV, REPO)
    finally:
        # The log is now closed: package its final status/traceback, including failure before QA.
        if (WORK/'exports/completion.json').is_file():
            repack = "import sys,json; from pathlib import Path; sys.path.insert(0,sys.argv[1]); import diagnostic_runner as d; w=Path(sys.argv[1]); d.package(w,json.loads((w/'exports/completion.json').read_text())['status'])"
            result = probe([PYTHON, '-c', repack, WORK], ENV, REPO)
            print(result.stdout, flush=True)
            assert result.returncode == 0, result.stderr[-2000:]
        fcntl.flock(session_lock, fcntl.LOCK_UN)
        session_lock.close()
else:
    print('VALIDATION_PASS: all cells executed; network, installs, filesystem mutations and GPU skipped')
'''

FINISH = '''if not VALIDATE_ONLY:
    from IPython.display import display, Markdown, FileLink
    completion = json.loads((WORK/'exports/completion.json').read_text(encoding='utf-8'))
    report = WORK/'exports/report.md'
    if report.exists():
        display(Markdown(report.read_text(encoding='utf-8')))
    print('Status:', completion['status'], '| Mandatory answers:', completion.get('mandatory_answered', 'see checkpoints'))
    if completion['status'] != 'complete':
        print('Partial results are preserved. A matching rerun resumes checkpoints; failed setup must be fixed first.')
    artifact = WORK/'HERMES_input_diagnostic_evidence.zip'
    link = Path.cwd()/artifact.name
    if link.resolve() != artifact.resolve():
        if link.is_symlink():
            link.unlink()
        # Always refresh the small evidence copy, never link/copy videos or weights here.
        shutil.copy2(artifact, link)
    display(FileLink(artifact.name))
    print('Evidence:', artifact, '| MiB:', round(artifact.stat().st_size/1024**2, 2))
    print('Send this ZIP and the saved executed notebook back for analysis. Preserve asset caches for reuse.')
'''


def main():
    reference = reference_data()
    sources = {"RUNNER_SOURCE": (ROOT/"scripts/autodl_input_diagnostic.py").read_text(encoding="utf-8").replace("\r\n", "\n"),
               "UTILS_SOURCE": (ROOT/"scripts/autodl_failure_pilot.py").read_text(encoding="utf-8").replace("\r\n", "\n")}
    reference_json = json.dumps(reference, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    cells = []
    def cell(kind, source):
        value = dict(cell_type=kind, id=f"input-diagnostic-{len(cells):02d}", metadata={}, source=source.splitlines(keepends=True))
        if kind == "code":
            compile(source, f"diagnostic_cell_{len(cells)}", "exec")
            value.update(execution_count=None, outputs=[])
        cells.append(value)
    cell("markdown", INTRO)
    cell("code", CONFIG)
    cell("markdown", "## Optional verified-support schema\nLeave `SUPPORT_ITEMS=[]` for the initial run. After reviewing actual frames, add objects with integer `sample_id`, integer `qa_index`, sorted unique `sampled_indices` (1–20 indices), `verified: True`, and nonempty `notes`. Indices refer to the original 0.5-fps sampled sequence, not source-video frame numbers. Only pre-query indices are accepted. Use a new WORK directory when changing annotations/configuration. There is no prefilled example with invented evidence.\n")
    cell("markdown", "## 1. Network, disk, pinned source and environment\nSetup can take longer than QA. GitHub/Hugging Face acceleration is kept separate from domestic package traffic. A silent subprocess prints liveness every 30 seconds.\n")
    cell("code", helpers())
    paths = list(reference["source_sha256"])+["data/streamingbench/streamingbench_realtime.json"]
    hashes = {x:hashlib.sha256((ROOT/x).read_bytes().replace(b"\r\n",b"\n")).hexdigest() for x in paths}
    assert all(hashes[x] == expected for x,expected in reference["source_sha256"].items()), "Inference source changed since audited pilot"
    cell("code", "SOURCE_HASHES = "+repr(hashes)+"\n"+CLONE)
    cell("code", install())
    model_source = old.MODEL.replace("WORK/'cache/huggingface/hub'", "Path(ENV['HF_HOME'])/'hub'")
    cell("code", model_source)
    cell("markdown", "## 2. Install embedded diagnostic sources and audited reference\nNo old experiment code is loaded from uploaded evidence. The reference contains trusted hashes, source annotations, sampling metadata and old predictions as comparison data.\n")
    embed = ""
    for variable, source in sources.items():
        assert "'''" not in source
        embed += variable+" = r'''"+source+"'''\nast.parse("+variable+")\n"
        embed += f"assert hashlib.sha256({variable}.encode()).hexdigest() == {hashlib.sha256(source.encode()).hexdigest()!r}\n"
    embed += "REFERENCE_JSON = "+repr(reference_json)+"\n"
    embed += f"assert hashlib.sha256(REFERENCE_JSON.encode()).hexdigest() == {hashlib.sha256(reference_json.encode()).hexdigest()!r}\n"
    cell("code", embed+EMBED)
    cells[-1]["metadata"]["jupyter"] = {"source_hidden": True}
    cell("markdown", "## 3. Run matched input controls and build review pages\nMandatory GPU preflight verifies feature ingestion and the checked native attention path. Each answer logs selected sampled indices/timestamps and fresh cache lengths. Skips remain visible; native/oracle scores are compared to historical predictions on exactly the answered identities.\n")
    cell("code", RUN)
    cell("markdown", "## 4. Download the evidence ZIP\nA failure stops Run All. To download failure evidence, run this final cell manually if the runner created an archive. Installation failures are in WORK/logs and may precede archive creation.\n")
    cell("code", FINISH)
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3 (ipykernel)", language="python", name="python3"),
                    language_info=dict(name="python", version="3.12.0")), nbformat=4, nbformat_minor=5)
    DEST.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+"\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST} ({len(cells)} cells; {DEST.stat().st_size/1024:.0f} KiB)")


if __name__ == "__main__":
    main()
