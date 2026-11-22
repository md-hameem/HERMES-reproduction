"""Build the self-contained TwinPast AutoDL notebook from audited inputs."""
import ast
import csv
import hashlib
import json
import subprocess
from pathlib import Path

from autodl_twinpast_diagnostic import ARTIFACT, cases

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "notebooks/autodl/hermes-twinpast-diagnostic.ipynb"
TEMPLATE = ROOT / "notebooks/autodl/hermes-target-binding.ipynb"
MODEL_SOURCE = ("inference/llavaov_hermes.py", "inference/abstract_hermes.py", "inference/reindex_1d.py")
INTRO = """# HERMES TwinPast historical-evidence diagnostic

Upload **this notebook only** to your AutoDL JupyterLab and choose **Restart Kernel → Run All**.
The same RTX 4090 container is fastest because it can reuse the validated causal-pilot Python
environment and model cache. A fresh CUDA 12.8-compatible container can bootstrap the same pins.
The notebook embeds the pinned model wrapper and experiment files. It does not need a Git clone.

The run tests nine frozen development pairs with two option orders. Each pair has four conditions
per order: question-only, recent-four, two earlier A plus recent-four, and two earlier B plus
recent-four. The history arms share exactly the same recent features, question and answer prompt
within an option order. Their scores are mapped to canonical answer identities.

Only 18 short MP4s are needed. Individual clips use bounded requests and verified SHA-256 hashes,
and completed pairs are checkpointed. If the exact clips already live on AutoDL, set MEDIA_REUSE
to a directory containing <pair_id>/A.mp4 and <pair_id>/B.mp4. Otherwise leave the defaults.
The first pair runs GPU preflight checks before the diagnostic proceeds.

Watch for MEDIA_READY, TWINPAST_PREFLIGHT_PASS and PAIR_DONE. At TWINPAST_DIAGNOSTIC_COMPLETE,
download the evidence ZIP and save the executed notebook. This is a diagnostic with annotated
earlier frames. No automatic retrieval method is evaluated.
"""


def notebook():
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    cells = []

    def ref(i):
        return "".join(template["cells"][i]["source"])

    def add(kind, source):
        cell = dict(cell_type=kind, id=f"twinpast-{len(cells):02d}", metadata={}, source=source.splitlines(True))
        if kind == "code":
            compile(source, cell["id"], "exec")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    def load(path):
        return json.loads((ROOT / path).read_text(encoding="utf-8"))

    with (ROOT / "results/twinpast_development_2026-11-23/pair_visual_screen.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        visual = list(csv.DictReader(handle))
    bundle = dict(manifest=load("data/twinpast/historical_correction_development.json"), visual=visual,
                  media=load("results/twinpast_development_2026-11-23/media_check.json"))
    selected = cases(bundle)
    total_bytes = sum(meta["bytes"] for _, row, _ in selected for meta in row["decoded"].values())
    add("markdown", INTRO + f"\nVerified media total: {total_bytes / 1024**2:.1f} MiB.\n")
    config = ref(1).replace("hermes_target_binding", "hermes_twinpast_diagnostic").replace(
        "HERMES_TARGET_BINDING_VALIDATE_ONLY", "HERMES_TWINPAST_VALIDATE_ONLY")
    config = "\n".join(line for line in config.splitlines()
                       if not line.startswith(("MAX_HOURS", "assert MAX_HOURS", "REPO_URL"))) + "\n"
    config = config.replace("WORK/'repository'", "WORK/'pinned_source'").replace(
        "WORK/'stage_runner.py'", "WORK/'provenance/twinpast_runner.py'").replace(
        "1,800 selector outcomes on 60 frozen clips", "9 pairs, 72 forced-option score rows")
    add("code", config + "MEDIA_REUSE = Path('/root/autodl-tmp/twinpast-development')\n")
    add("markdown", "## 1. Verify runtime and materialize the experiment\n")
    setup = ref(3).replace("assert free >= 18, f'Need at least 18 GiB free on data disk; found {free:.1f} GiB'",
        "minimum_free = 1 if (REUSE/'venv/bin/python').is_file() and (REUSE/'cache/huggingface').is_dir() else 18\n"
        "    assert free >= minimum_free, f'Need {minimum_free} GiB free on the data disk; found {free:.1f} GiB'")
    add("code", setup)
    revision = "2a1385e0abc94debb64761b0465713c799a81c1c"
    hashes = ast.literal_eval(ref(4).splitlines()[0].split(" = ", 1)[1])
    source = {}
    for name in MODEL_SOURCE:
        raw = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT).replace(b"\r\n", b"\n")
        assert hashlib.sha256(raw).hexdigest() == hashes[name]
        source[f"pinned_source/{name}"] = raw.decode("utf-8")
    source["provenance/twinpast_runner.py"] = (ROOT / "scripts/autodl_twinpast_diagnostic.py").read_text(encoding="utf-8")
    source["provenance/bundle.json"] = json.dumps(bundle, ensure_ascii=False, indent=2) + "\n"
    integrity = {key: hashlib.sha256(value.encode()).hexdigest() for key, value in source.items()}
    add("code", f"SOURCES = {source!r}\nEXPECTED = {integrity!r}\n" + """
for relative, contents in SOURCES.items():
    raw = contents.encode('utf-8')
    assert hashlib.sha256(raw).hexdigest() == EXPECTED[relative]
    if relative.endswith('.py'):
        ast.parse(contents, filename=relative)
    if not VALIDATE_ONLY:
        target = WORK / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.read_bytes() != raw:
            raise ValueError(f'Existing file differs at {target}. Preserve it and choose a fresh WORK directory.')
        if not target.is_file():
            target.write_bytes(raw)
print('EMBEDDED_SOURCE_AND_MANIFEST_PASS', BASE_REVISION)
""")
    add("code", ref(5))
    add("code", ref(6).replace("    failures = []\n    for host", """    local_model = Path(ENV['HF_HOME'])/'hub'/'models--llava-hf--llava-onevision-qwen2-0.5b-ov-hf'/'snapshots'/MODEL_SHA
    local_env = {**ENV, 'HF_HUB_OFFLINE': '1'}
    use_cache = local_model.is_dir() and probe(
        [PYTHON, '-c', MODEL_DOWNLOAD, MODEL_ID, MODEL_SHA, Path(ENV['HF_HOME'])/'hub', pointer], local_env
    ).returncode == 0
    if use_cache:
        print('REUSING_PINNED_MODEL', local_model)
    failures = []
    for host""").replace("        try:\n            logged([PYTHON", "        if use_cache:\n            break\n        try:\n            logged([PYTHON"))
    add("markdown", "## 2. Acquire clips, run the preflight and score the nine pairs\n")
    add("code", f"ARTIFACT = {ARTIFACT!r}\n" + """
if not VALIDATE_ONLY:
    (WORK/'launch.json').write_text(json.dumps(dict(started_utc=datetime.now(timezone.utc).isoformat(),
        source_revision=BASE_REVISION, model_revision=MODEL_SHA, transformers_revision=TF_SHA,
        embedded_sha256=EXPECTED, gpu=smi.stdout.strip()), indent=2), encoding='utf-8')
    env = {**ENV, 'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1'}
    logged([PYTHON, '-u', RUNNER, '--work', WORK, '--model', MODEL_PATH, '--media-reuse', MEDIA_REUSE],
           'twinpast.log', env, REPO)
else:
    print('VALIDATION_PASS: setup and embedded sources checked without GPU, network or file writes')
""")
    add("markdown", "## 3. Download the evidence ZIP\n\nAfter a runner error, run this cell manually to retrieve failure evidence.\n")
    add("code", """
if not VALIDATE_ONLY:
    from IPython.display import FileLink, display
    artifact = WORK/ARTIFACT
    if artifact.is_file():
        link = Path.cwd()/ARTIFACT
        if artifact.resolve() != link.resolve():
            shutil.copyfile(artifact, link)
        print((WORK/'completion.json').read_text(encoding='utf-8'))
        display(FileLink(ARTIFACT))
    else:
        print('No runner ZIP yet. Inspect setup logs under', WORK/'logs')
""")
    return dict(cells=cells, metadata=template["metadata"], nbformat=4, nbformat_minor=5)


if __name__ == "__main__":
    DEST.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST}")
