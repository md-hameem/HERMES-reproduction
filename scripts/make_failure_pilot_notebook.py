"""Generate the self-contained AutoDL notebook from the tested runner source."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "notebooks/autodl/hermes-causal-scorer-pilot.ipynb"


INTRO = """# HERMES causal-scorer pilot — fresh AutoDL container

Upload **this notebook alone**, choose **Kernel → Restart Kernel and Run All Cells**, and leave it running. Setup clones only the needed repository paths and installs an isolated pinned Python environment. No old container or evidence ZIP upload is required.

**Recommended:** one RTX 4090 / 24 GB, a recent CUDA 12.8-compatible driver, and at least 18 GiB free under `/root/autodl-tmp`. The notebook checks these before installing. The Jupyter kernel can use the container's existing Python; inference runs in its own Python 3.12 environment. Do not manually upgrade Transformers.

The default fast screen uses **8 videos × all 5 questions × 3 methods = 120 answers** from the small 201–250 archive. It compares FIFO 4K, current HERMES 4K and a fresh native causal scorer. This is an outcome-selected development diagnostic, not a confirmatory benchmark or a new method claim. It does not rerun the full 498-video benchmark.

Speed comes from selective ZIP downloads where supported, one model load, shared video features, and per-method/video checkpoints. Startup depends on network speed. A soft two-hour inference budget is checked after complete three-method video blocks, after at least two videos. It excludes installation/download time and can overrun by one block. A valid partial result is packaged if the budget is reached; rerun All to continue. Actual timing estimates appear after the first block.

At the end download **`HERMES_causal_scorer_pilot_evidence.zip`** using the displayed link. It contains predictions, paired summaries, scorer preflight, asset/source hashes, logs, sampled timestamps, contact sheets and a review sheet. It excludes model weights, video files and feature tensors. Evidence-support annotation remains a human task.

This notebook has CPU numerical/safety tests and a headless validation mode. The complete fresh-container GPU execution is pending your run. Any failed correctness preflight stops the experiments and preserves a diagnostic package.
"""

CONFIG = '''# Defaults are ready for a new container. Change only these if desired.
from pathlib import Path
import os, sys, subprocess, json, hashlib, shutil, time, zipfile, ast
from datetime import datetime, timezone

WORK = Path('/root/autodl-tmp/hermes_causal_pilot')
PER_OUTCOME = 2       # 1 = 4 videos / 60 answers; 2 = 8 / 120; 3 = 12 / 180
# If changing PER_OUTCOME after a run, also choose a new WORK directory.
MAX_HOURS = 2.0      # soft inference budget; 0 disables it; rerun resumes completed blocks
REPO_URL = 'https://github.com/md-hameem/HERMES-reproduction.git'
BASE_REVISION = '2a1385e0abc94debb64761b0465713c799a81c1c'
MODEL_ID = 'llava-hf/llava-onevision-qwen2-0.5b-ov-hf'
MODEL_SHA = '74dd0bf867a4cda7950c17663794267c60cf4b40'
TF_SHA = '66bc4def9505fa7c7fe4aa7a248c34a026bb552b'
VALIDATE_ONLY = os.environ.get('HERMES_PILOT_VALIDATE_ONLY') == '1'
assert PER_OUTCOME in (1, 2, 3) and MAX_HOURS >= 0
REPO = WORK / 'repository'
VENV = WORK / 'venv'
PYTHON = VENV / 'bin/python'
RUNNER = WORK / 'pilot_runner.py'
print('Plan:', PER_OUTCOME * 4, 'videos;', PER_OUTCOME * 20, 'questions per method; 3 methods')
print('Mode:', 'CPU notebook validation' if VALIDATE_ONLY else 'AutoDL execution')
'''

HELPERS = '''def network_environment():
    env = os.environ.copy()
    if Path('/etc/network_turbo').is_file():
        proc = subprocess.run(['bash', '-lc', 'source /etc/network_turbo >/dev/null 2>&1; env -0'], capture_output=True)
        if proc.returncode == 0:
            for item in proc.stdout.split(b'\\0'):
                if b'=' in item:
                    key, value = item.split(b'=', 1)
                    env[key.decode()] = value.decode()
    env.update(HF_HOME=str(WORK/'cache/huggingface'), HF_HUB_DISABLE_XET='1',
               HF_HUB_ENABLE_HF_TRANSFER='0', TOKENIZERS_PARALLELISM='false',
               CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='8', GIT_LFS_SKIP_SMUDGE='1',
               UV_CACHE_DIR=str(WORK/'cache/uv'), UV_LINK_MODE='copy', TMPDIR=str(WORK),
               UV_DEFAULT_INDEX='https://pypi.tuna.tsinghua.edu.cn/simple',
               PYTORCH_ALLOC_CONF='expandable_segments:True')
    env.pop('PYTHONHOME', None)
    env['PYTHONPATH'] = ''
    return env

def logged(command, name, env=None, cwd=None):
    logfile = WORK/'logs'/name
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('a', encoding='utf-8') as handle:
        proc = subprocess.Popen([str(x) for x in command], env=env, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1)
        for line in proc.stdout:
            print(line, end='', flush=True)
            handle.write(line)
            handle.flush()
        status = proc.wait()
    if status:
        raise RuntimeError(f'Command failed ({status}); full log: {logfile}')

def probe(command, env=None, cwd=None):
    return subprocess.run([str(x) for x in command], env=env, cwd=cwd,
                          text=True, capture_output=True)

if not VALIDATE_ONLY:
    assert sys.platform.startswith('linux'), 'Run this notebook in an AutoDL Linux container'
    WORK.mkdir(parents=True, exist_ok=True)
    ENV = network_environment()
    free = shutil.disk_usage(WORK).free / 1024**3
    assert free >= 18, f'Need at least 18 GiB free on data disk; found {free:.1f} GiB'
    smi = probe(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], ENV)
    assert smi.returncode == 0 and smi.stdout.strip(), 'CUDA GPU is missing'
    print('GPU:', smi.stdout.strip(), '\\nFree GiB:', round(free, 1))
    import re
    first_gpu = smi.stdout.splitlines()[0]
    assert float(re.search(r'(\\d+(?:\\.\\d+)?)\\s*MiB', first_gpu)[1]) >= 15000, 'Use a GPU with at least 15 GiB VRAM'
    # Never print the environment or proxy values.
'''

CLONE = '''if not VALIDATE_ONLY:
    if not (REPO/'.git').is_dir():
        assert not REPO.exists(), f'Non-Git directory already exists: {REPO}'
        logged(['git', 'clone', '--filter=blob:none', '--no-checkout', '--depth', '1', REPO_URL, REPO],
               'clone.log', ENV)
    # Reject local source edits rather than overwriting them.
    if (REPO/'.git/index').exists():
        dirty = probe(['git', 'status', '--porcelain', '--untracked-files=no'], ENV, REPO)
        assert dirty.returncode == 0 and not dirty.stdout.strip(), 'Repository has local changes; preserve them and use a new WORK directory'
    else:
        assert not [p for p in REPO.iterdir() if p.name != '.git'], 'Incomplete clone has unexpected files'
    logged(['git', 'fetch', '--depth', '1', 'origin', BASE_REVISION], 'clone.log', ENV, REPO)
    logged(['git', 'sparse-checkout', 'init', '--cone'], 'clone.log', ENV, REPO)
    logged(['git', 'sparse-checkout', 'set', 'inference', 'video_qa', 'scripts',
            'data/streamingbench', 'results/failure_pilot'], 'clone.log', ENV, REPO)
    logged(['git', 'checkout', '--detach', BASE_REVISION], 'clone.log', ENV, REPO)
    assert probe(['git', 'rev-parse', 'HEAD'], ENV, REPO).stdout.strip() == BASE_REVISION
    for rel, expected in SOURCE_HASHES.items():
        actual = hashlib.sha256((REPO/rel).read_bytes().replace(b'\\r\\n', b'\\n')).hexdigest()
        assert actual == expected, f'Pinned source mismatch: {rel}'
    print('Pinned sparse source checkout verified; paper PDFs and old evidence archives were not cloned.')
'''

INSTALL = '''ENV_PROBE = r"""
import sys, torch, transformers, tokenizers, huggingface_hub, pandas, decord, logzero
import json, importlib.metadata as md
assert sys.version_info[:2] == (3, 12)
assert torch.__version__ == '2.10.0+cu128'
assert transformers.__version__ == '4.45.0.dev0'
origin = json.loads(md.distribution('transformers').read_text('direct_url.json') or '{}')
assert '66bc4def9505fa7c7fe4aa7a248c34a026bb552b' in str(origin), 'Transformers source revision differs'
assert tokenizers.__version__ == '0.19.1' and huggingface_hub.__version__ == '0.34.4'
assert torch.cuda.is_available(), 'PyTorch cannot use CUDA; check the container driver'
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
import inspect
assert 'seq_len' in inspect.signature(Qwen2RotaryEmbedding.forward).parameters
x = torch.ones((16, 16), device='cuda')
assert float((x @ x)[0, 0]) == 16
print('PINNED_ENV_PASS', torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))
"""
if not VALIDATE_ONLY:
    if not PYTHON.is_file() or probe([PYTHON, '-c', ENV_PROBE], ENV).returncode:
        if probe([sys.executable, '-m', 'uv', '--version'], ENV).returncode:
            attempt = probe([sys.executable, '-m', 'pip', 'install', '-q', 'uv', '-i',
                             'https://pypi.tuna.tsinghua.edu.cn/simple'], ENV)
            if attempt.returncode:
                logged([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], 'setup.log', ENV)
        UV = [sys.executable, '-m', 'uv']
        if not PYTHON.is_file():
            logged(UV + ['venv', VENV, '--python', '3.12', '--seed'], 'setup.log', ENV)
        def install(packages, extra=()):
            logged(UV + ['pip', 'install', '--python', PYTHON] + list(extra) + list(packages), 'setup.log', ENV)
        install(['torch @ https://download-r2.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl'])
        install(['numpy==1.26.4', 'tokenizers==0.19.1', 'huggingface-hub==0.34.4',
                 'accelerate==0.34.2', 'decord==0.6.0', 'logzero', 'pillow',
                 'sentencepiece', 'safetensors', 'packaging', 'filelock', 'regex',
                 'requests', 'pyyaml', 'tqdm', 'fsspec', 'protobuf', 'einops', 'psutil', 'pandas'])
        # Install the exact Transformers revision used by the audited reproduction.
        install([f'https://codeload.github.com/huggingface/transformers/zip/{TF_SHA}'], ['--no-deps'])
    logged([PYTHON, '-c', ENV_PROBE], 'environment.log', ENV)
    ENV['PYTHONPATH'] = str(REPO) + os.pathsep + str(REPO/'scripts')
    logged([PYTHON, '-m', 'pip', 'freeze'], 'environment_freeze.txt', ENV)
'''

MODEL = '''MODEL_DOWNLOAD = r"""
import json, sys
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors import safe_open
path = Path(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], cache_dir=sys.argv[3],
    allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.vocab', 'tokenizer.*', 'merges.txt'],
    ignore_patterns=['onnx/*', '*.onnx', '*.bin', '*.msgpack'], max_workers=4))
assert path.name == sys.argv[2] and (path/'config.json').is_file()
files = list(path.glob('*.safetensors'))
assert files
for file in files:
    with safe_open(str(file), framework='pt', device='cpu') as handle:
        assert list(handle.keys())
Path(sys.argv[4]).write_text(json.dumps(str(path)), encoding='utf-8')
print('MODEL_REVISION_PASS', path.name)
"""
if not VALIDATE_ONLY:
    pointer = WORK/'model_path.json'
    failures = []
    for host in ('https://hf-mirror.com', 'https://huggingface.co'):
        try:
            logged([PYTHON, '-c', MODEL_DOWNLOAD, MODEL_ID, MODEL_SHA,
                    WORK/'cache/huggingface/hub', pointer], 'model_download.log', {**ENV, 'HF_ENDPOINT': host})
            break
        except RuntimeError as exc:
            failures.append(type(exc).__name__)
            print('Model endpoint failed; trying fallback', flush=True)
    else:
        raise RuntimeError('Model download failed on both endpoints; rerun to resume cached files')
    MODEL_PATH = Path(json.loads(pointer.read_text(encoding='utf-8')))
    assert MODEL_PATH.name == MODEL_SHA
'''

RUN = '''# This subprocess owns the GPU. The notebook kernel never imports torch.
if not VALIDATE_ONLY:
    started = datetime.now(timezone.utc).isoformat()
    (WORK/'launch.json').write_text(json.dumps(dict(started_utc=started, repository_revision=BASE_REVISION,
        runner_sha256=hashlib.sha256(RUNNER_SOURCE.encode()).hexdigest(), per_outcome=PER_OUTCOME,
        max_hours=MAX_HOURS, gpu_driver=smi.stdout.strip()), indent=2), encoding='utf-8')
    try:
        logged([PYTHON, RUNNER, '--repo', REPO, '--work', WORK, '--model', MODEL_PATH,
                '--per-outcome', str(PER_OUTCOME), '--max-hours', str(MAX_HOURS)], 'pilot.log', ENV, REPO)
    finally:
        if (WORK/'exports/completion.json').is_file() and (WORK/'HERMES_causal_scorer_pilot_evidence.zip').is_file():
            # Repack after the live log closes, including its final traceback/status.
            repack = "import sys,json; from pathlib import Path; sys.path.insert(0,sys.argv[1]); import pilot_runner as p; w=Path(sys.argv[1]); p.package(w,json.loads((w/'exports/completion.json').read_text())['status'])"
            result = probe([PYTHON, '-c', repack, WORK], ENV, REPO)
            print(result.stdout, flush=True)
            assert result.returncode == 0, result.stderr[-2000:]
else:
    print('VALIDATION_PASS: every notebook cell executed; network/install/GPU steps deliberately skipped')
'''

FINISH = '''if not VALIDATE_ONLY:
    from IPython.display import display, Markdown, FileLink
    completion = json.loads((WORK/'exports/completion.json').read_text(encoding='utf-8'))
    display(Markdown((WORK/'exports/report.md').read_text(encoding='utf-8')))
    print('Status:', completion['status'])
    if completion['status'] != 'complete':
        print('A partial screen is preserved. Run All again to resume completed checkpoints.')
    # Expose the artifact under this notebook's Jupyter directory without copying weights/videos.
    artifact = WORK/'HERMES_causal_scorer_pilot_evidence.zip'
    link = Path.cwd()/artifact.name
    if link.resolve() != artifact.resolve():
        if link.is_symlink():
            link.unlink()
        if not link.exists():
            try:
                os.link(artifact, link)
            except OSError:
                shutil.copy2(artifact, link)
        elif hashlib.sha256(link.read_bytes()).digest() != hashlib.sha256(artifact.read_bytes()).digest():
            shutil.copy2(artifact, link)
    print('Download this evidence ZIP and send it back for analysis:')
    display(FileLink(artifact.name))
    print('Original:', artifact, '\\nMiB:', round(artifact.stat().st_size/1024**2, 2))
    print('Sparse contact sheets are inside the ZIP. They do not replace reviewing the actual support.')
'''


def main():
    cells = []
    def cell(kind, source):
        value = dict(cell_type=kind, id=f"pilot-{len(cells):02d}", metadata={}, source=source.splitlines(keepends=True))
        if kind == "code":
            value.update(execution_count=None, outputs=[])
            compile(source, f"cell_{len(cells)}", "exec")
        cells.append(value)
    runner = (ROOT / "scripts/autodl_failure_pilot.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    paths = ["inference/llavaov_hermes.py", "inference/abstract_hermes.py", "inference/reindex_1d.py",
             "video_qa/base.py", "video_qa/hermes_vqa.py", "scripts/build_failure_pilot.py",
             "data/streamingbench/streamingbench_realtime.json", "results/failure_pilot/paired_inventory.csv"]
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths}
    cell("markdown", INTRO)
    cell("code", CONFIG)
    cell("markdown", "## 1. Network acceleration, disk and GPU checks\nProxy values are inherited privately from AutoDL's network helper when available.\n")
    cell("code", HELPERS)
    cell("code", "SOURCE_HASHES = " + repr(hashes) + "\n" + CLONE)
    cell("markdown", "## 2. Isolated pinned environment and model\nThis may be the slowest step in a new container. Completed package/model downloads are cached for reruns.\n")
    cell("code", INSTALL)
    cell("code", MODEL)
    cell("markdown", "## 3. Install the embedded, source-checked experimental runner\nThe complete runner is embedded so this notebook can be uploaded by itself. Original inference source files are unchanged.\n")
    assert "'''" not in runner, "Runner needs a different embedding delimiter"
    cell("code", "RUNNER_SOURCE = r'''" + runner + "'''\n" +
         "ast.parse(RUNNER_SOURCE)\n" +
         f"assert hashlib.sha256(RUNNER_SOURCE.encode()).hexdigest() == {hashlib.sha256(runner.encode()).hexdigest()!r}\n" +
         "if not VALIDATE_ONLY:\n    RUNNER.write_text(RUNNER_SOURCE, encoding='utf-8', newline='\\n')\n" +
         "print('EMBEDDED_RUNNER_HASH_PASS')\n")
    cells[-1]["metadata"]["jupyter"] = {"source_hidden": True}
    cell("markdown", "## 4. Download, validate and run\nThe mandatory GPU preflight compares cached-feature ingestion against the original wrapper and layerwise causal attention against the native decoder before and after real pruning. A failure stops QA. After each video block, timing estimates and summaries are saved.\n")
    cell("code", RUN)
    cell("markdown", "## 5. Download results\nDo not delete the container before downloading the evidence ZIP. A rerun revalidates assets and resumes matching completed checkpoints.\n")
    cell("code", FINISH)
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3 (ipykernel)", language="python", name="python3"),
                    language_info=dict(name="python", version="3.12.0")), nbformat=4, nbformat_minor=5)
    DEST.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST} ({len(cells)} cells; embedded runner SHA256 {hashlib.sha256(runner.encode()).hexdigest()})")


if __name__ == "__main__":
    main()
