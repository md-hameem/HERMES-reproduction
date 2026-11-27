"""Package the source-verified DROID capability test as an AutoDL notebook."""
import ast
import base64
from io import BytesIO
import json

from decord import VideoReader
from PIL import Image

from freeze_droid_state_control import DEST as MANIFEST
from make_twinpast_boundary_notebook import DEST as TEMPLATE
from prepare_droid_state_control import WORK
from validate_twinpast_diagnostic_evidence import ROOT, digest

DEST = ROOT / "notebooks/autodl/hermes-droid-state-control.ipynb"
INTRO = """# DROID controlled historical-state capability

Use the same AutoDL GPU container as the TwinPast runs, with at least 22 GiB
VRAM. Upload this notebook, then select **Restart Kernel → Run All**. The
notebook includes its 12 source frames and can reuse the pinned 7B environment
and model. A fresh container needs to download the environment and weights.

The six open/closed pairs come from different DROID collection sites. Their
timelines are controlled edits, with a real camera frame held for 0.5 seconds
and shared neutral filler. Compare an aligned cue, a cue missed by uniform
sampling, and the same off-grid cue supplied by annotation, all with ten
answer frames. A one-frame cue-only condition is a diagnostic. Both option
orders are scored. This pilot measures capability under controlled inputs.

Frames are adapted from the [DROID dataset](https://droid-dataset.github.io/),
Khazatsky et al. and the DROID Dataset Team, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The notebook embeds resized frames only. Its manifest records the raw sources.

When **DROID_STATE_CONTROL_COMPLETE** appears, save the executed notebook and
download **HERMES_droid_state_control_evidence.zip**.
"""

MATERIALIZE = """
import base64

for filename, encoded in INPUT_FRAMES.items():
    assert Path(filename).name == filename and filename.endswith('.png')
    raw = base64.b64decode(encoded, validate=True)
    assert hashlib.sha256(raw).hexdigest() == FRAME_HASHES[filename]
    if not VALIDATE_ONLY:
        folder = WORK / 'input_frames'
        folder.mkdir(exist_ok=True)
        path = folder / filename
        if path.is_file() and path.read_bytes() != raw:
            raise ValueError(f'Input frame differs: {path}')
        if not path.is_file():
            path.write_bytes(raw)
print('DROID_INPUT_FRAMES_VERIFIED', len(INPUT_FRAMES), flush=True)
"""


def inputs(cases):
    images, receipts, targets, questions = {}, {}, {}, []
    for case in cases:
        pair = case["pair_id"]
        targets[pair] = {}
        questions.append({key: case[key] for key in ("pair_id", "source_group", "question", "options")})
        path = WORK / case["source_episode"] / f"{case['camera']}.mp4"
        assert digest(path.read_bytes()) == case["video_source"]["sha256"]
        reader = VideoReader(str(path), num_threads=1)
        for side, branch in case["branches"].items():
            rgb = reader.get_batch([branch["source_frame"]]).asnumpy()[0]
            assert digest(rgb.tobytes()) == branch["decoded_rgb_sha256"]
            image = Image.fromarray(rgb).resize((640, 360), Image.Resampling.BILINEAR)
            stream = BytesIO()
            image.save(stream, format="PNG")
            raw = stream.getvalue()
            name = f"{pair}_{side}.png"
            images[name] = base64.b64encode(raw).decode("ascii")
            receipts[name] = dict(png_sha256=digest(raw), rgb_sha256=digest(image.tobytes()), shape=[360, 640, 3],
                                  source_frame=branch["source_frame"], source_episode=case["source_episode"])
            targets[pair][side] = branch["gold"]
    return images, receipts, targets, questions


def notebook():
    book = json.loads(TEMPLATE.read_bytes())
    cells = book["cells"]

    def get(i):
        return "".join(cells[i]["source"])

    def put(i, source):
        cells[i]["source"] = source.splitlines(keepends=True)

    manifest = json.loads(MANIFEST.read_bytes())
    images, receipts, targets, questions = inputs(manifest["cases"])
    sources = ast.literal_eval(ast.parse(get(4)).body[0].value)
    prior = json.loads(sources.pop("provenance/boundary_cases.json"))
    for name in ("boundary_runner.py", "twinpast_boundary_policy.py", "evaluation_targets.json"):
        sources.pop(f"provenance/{name}")
    spec = dict(model_revision=prior["model_revision"], video_processor=prior["video_processor"],
                timeline=manifest["timeline"], manifest_sha256=digest(MANIFEST.read_bytes()), cases=questions)
    sources.update({
        "provenance/droid_source_manifest.json": MANIFEST.read_text(encoding="utf-8"),
        "provenance/droid_cases.json": json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
        "provenance/droid_frame_receipts.json": json.dumps(receipts, indent=2) + "\n",
        "provenance/droid_targets.json": json.dumps(targets, indent=2) + "\n",
        "provenance/droid_runner.py": (ROOT / "scripts/autodl_droid_state_control.py").read_text(encoding="utf-8"),
    })
    hashes = {name: digest(src.encode()) for name, src in sources.items()}
    put(0, INTRO)
    put(1, get(1).replace("hermes_twinpast_boundary", "hermes_droid_state_control")
        .replace("HERMES_TWINPAST_BOUNDARY_VALIDATE_ONLY", "HERMES_DROID_STATE_CONTROL_VALIDATE_ONLY")
        .replace("provenance/boundary_runner.py", "provenance/droid_runner.py")
        .replace("8 archived pairs, 3 policies, frozen CPU rule", "6 source-separated DROID pairs, controlled frame inputs")
        .replace("PREVIOUS = REUSE\n", ""))
    put(4, f"SOURCES = {sources!r}\nEXPECTED = {hashes!r}\n" + "\n".join(get(4).split("\n")[2:]))
    put(7, "## 2. Run the controlled DROID capability comparison\n")
    put(8, get(8).replace("HERMES_twinpast_boundary_evidence.zip", "HERMES_droid_state_control_evidence.zip")
        .replace(", '--previous', PREVIOUS", "").replace("'twinpast_boundary.log'", "'droid_state_control.log'"))
    frame_hashes = {name: row["png_sha256"] for name, row in receipts.items()}
    cells.insert(5, dict(cell_type="code", execution_count=None, metadata={}, outputs=[],
                         source=(f"INPUT_FRAMES = {images!r}\nFRAME_HASHES = {frame_hashes!r}\n"
                                 + MATERIALIZE).splitlines(keepends=True)))
    for i, cell in enumerate(cells):
        cell["id"] = f"droid-control-{i:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    return book


if __name__ == "__main__":
    DEST.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST}")
