"""Build the self-contained CPU notebook for the blinded historical evidence review."""
import base64
import csv
import hashlib
import io
import json
from zipfile import ZIP_DEFLATED, ZipFile

from prepare_target_binding_review import ARCHIVE, OUT, ROOT
from validate_causal_pilot_evidence import check_manifest

DEST = ROOT / "notebooks/autodl/hermes-blind-support-review.ipynb"
INTRO = """# HERMES blinded visual-support review

Upload this notebook to the same AutoDL instance and run all cells in CPU/no-GPU mode.
It uses the existing target-binding video cache. This notebook needs neither a fresh
checkout nor a model, dataset download, or GPU. It checks the clip hashes and saved
decoder timestamps before rendering 147 windows across 32 questions.

Download **HERMES_blind_support_review.zip**, unzip it, and open `review.html`.
The worksheet uses the same window IDs. Record the supported option, the sampled
frame IDs in the notes, and whether the evidence applies at the question time.
If the evidence is unreadable or ambiguous, mark the window as uncertain.
Each reviewer should use a separate copy of the worksheet.
"""

CONFIG = """from pathlib import Path
import base64, csv, hashlib, os, shutil, subprocess, sys
from zipfile import ZipFile

BASE = Path('/root/autodl-tmp')
MEDIA = BASE / 'hermes_target_binding/videos'
WORK = BASE / 'hermes_blind_support_review'
VALIDATE_ONLY = os.environ.get('HERMES_BLIND_REVIEW_VALIDATE_ONLY') == '1'
OUTPUT_NAME = 'HERMES_blind_support_review.zip'
"""

RUN = """for relative, (encoded, digest) in PAYLOAD.items():
    payload = base64.b64decode(encoded, validate=True)
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f'Embedded review input is corrupt: {relative}')
    if relative.endswith('.py'):
        compile(payload, relative, 'exec')
    if not VALIDATE_ONLY:
        target = WORK / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != payload:
            raise RuntimeError(f'Review input changed: {target}. Use a new WORK directory.')
        if not target.exists():
            target.write_bytes(payload)

if VALIDATE_ONLY:
    print('BLIND_REVIEW_VALIDATION_PASS')
else:
    from IPython.display import FileLink, display

    python_choices = (
        BASE / 'hermes_target_binding/venv/bin/python',
        BASE / 'hermes_causal_pilot/venv/bin/python',
        Path(sys.executable),
    )
    python = next((p for p in python_choices if p.is_file() and subprocess.run(
        [str(p), '-c', 'import decord, PIL'], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL).returncode == 0), None)
    if python is None:
        raise RuntimeError('The experiment Python with Decord and Pillow is missing.')

    artifact = WORK / OUTPUT_NAME
    worksheet = WORK / 'visual_support_review.csv'
    if not artifact.exists():
        command = [str(python), '-u', str(WORK / 'render_target_binding_review.py'),
                   '--worksheet', str(worksheet), '--evidence', str(WORK / 'review_reference.zip'),
                   '--videos', str(MEDIA), '--output', str(artifact)]
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            for line in proc.stdout:
                print(line.rstrip(), flush=True)
            if proc.wait():
                raise RuntimeError('Review rendering failed. See the clip or environment error above.')
    with ZipFile(artifact) as rendered:
        with worksheet.open(encoding='utf-8', newline='') as handle:
            ids = {row['review_id'] for row in csv.DictReader(handle)}
        expected = {'review.html', worksheet.name} | {f'frames/{rid}.jpg' for rid in ids}
        if (len(ids) != 147 or set(rendered.namelist()) != expected or rendered.testzip() is not None
                or rendered.read(worksheet.name) != worksheet.read_bytes()):
            raise ValueError('The rendered ZIP differs from the blinded worksheet. Use a new WORK directory.')
    link = Path.cwd() / OUTPUT_NAME
    if link.resolve() != artifact.resolve():
        if link.exists():
            if hashlib.sha256(link.read_bytes()).digest() != hashlib.sha256(artifact.read_bytes()).digest():
                raise FileExistsError(f'Choose a clean notebook directory. A different packet exists at {link}.')
        else:
            shutil.copyfile(artifact, link)
    print('BLIND_REVIEW_PACKET_READY | 147 windows | 32 questions')
    display(FileLink(link.name))
"""


def reference(worksheet):
    ids = {int(row["sample_id"]) for row in csv.DictReader(io.StringIO(worksheet.decode("utf-8")))}
    assert len(ids) == 28
    with ZipFile(ARCHIVE) as archive:
        manifest, payload = check_manifest(archive)
    assert manifest["status"] == "complete"
    assets = json.loads(payload["asset_manifest.json"])["videos"]
    output = io.BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("asset_manifest.json", json.dumps(dict(videos=[
            dict(sample_id=a["sample_id"], sha256=a["sha256"]) for a in assets if a["sample_id"] in ids])))
        for sid in sorted(ids):
            name = f"provenance/features_{sid}.json"
            archive.writestr(name, json.dumps(dict(timestamps=json.loads(payload[name])["timestamps"])))
    return output.getvalue()


def notebook():
    worksheet = (OUT / "visual_support_review.csv").read_bytes()
    files = {"visual_support_review.csv": worksheet, "review_reference.zip": reference(worksheet)}
    for name in ("render_target_binding_review.py", "autodl_confirmation_visual_review.py"):
        files[name] = (ROOT / "scripts" / name).read_bytes().replace(b"\r\n", b"\n")
    payload = {name: (base64.b64encode(blob).decode("ascii"), hashlib.sha256(blob).hexdigest())
               for name, blob in files.items()}
    cells = []
    for i, (kind, source) in enumerate((("markdown", INTRO), ("code", CONFIG),
                                        ("code", "PAYLOAD = " + repr(payload) + "\n" + RUN))):
        cell = dict(id=f"blind-support-{i}", cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
        if kind == "code":
            compile(source, cell["id"], "exec")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3")),
                nbformat=4, nbformat_minor=5)


if __name__ == "__main__":
    DEST.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST}")
