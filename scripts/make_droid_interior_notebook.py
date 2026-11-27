"""Package the audited DROID interior cases and generate a standalone AutoDL notebook."""
import ast
import json
from zipfile import ZipFile, ZipInfo

from check_droid_interior_coverage import INTAKE, OUTPUT, REVIEW, check
from freeze_droid_natural_history import encode, reader_for
from prepare_droid_interior import WORK
from validate_droid_terminal_attention_evidence import ARCHIVE, ROOT, TEMPLATE, digest, verify_inventory

DEST = ROOT / "notebooks/autodl/hermes-droid-interior.ipynb"
FRAME_ZIP = "DROID_interior_frames.zip"
INTRO = """# DROID event-relative interior evidence

Use the GPU container from the span-selection run. Put **DROID_interior_frames.zip**
beside this notebook and select **Restart Kernel → Run All**.

On seven reviewed DROID clips, compare native decoding, the strongest fixed
three-frame window, last-three recency, and an annotated three-frame interior
window. All arms keep the original ten frames and token positions. Run both
option orders and verify native parity with an explicit full causal mask.
This experiment tests whether the 7B model can use the annotated interior evidence.

After **DROID_INTERIOR_COMPLETE**, save the executed notebook and download
**HERMES_droid_interior_evidence.zip**. Run the final cell manually to obtain
the evidence ZIP after a runner failure.

Source frames are from [DROID](https://droid-dataset.github.io/) under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
"""


def frame_spec():
    report = check(INTAKE, REVIEW)
    assert json.loads(json.dumps(report)) == json.loads(OUTPUT.read_bytes())
    assert report["gpu_feasibility_criterion_met"]
    intake = {r["episode_id"]: r for r in json.loads(INTAKE.read_bytes())["intake"]}
    assets = {}
    for i, case in enumerate(report["cases"]):
        reader = reader_for(WORK, intake[case["episode_id"]], case["camera"])
        names = encode(reader, case["frame_grid"], f"interior_{i:02}", assets)
        case["frames"] = list(names.values())
        case["id"] = f"interior_{i:02}"
        case["timestamps_s"] = reader.get_frame_timestamp(case["frame_grid"])[:, 0].tolist()
    path = WORK / FRAME_ZIP
    with ZipFile(path, "w") as archive:
        for name, (raw, _) in sorted(assets.items()):
            archive.writestr(ZipInfo(name), raw)
    static = max(report["all_static_windows"], key=lambda r: (r["covered"], -abs(r["slots"][1] - 5)))
    return dict(cases=report["cases"], static_window=static["slots"], review_sha256=report["review_sha256"],
                coverage_audit_sha256=digest(OUTPUT.read_bytes()), frame_zip_sha256=digest(path.read_bytes()),
                frame_receipts={n: dict(png_sha256=digest(raw), rgb_sha256=h) for n, (raw, h) in assets.items()})


def notebook(spec):
    book = json.loads(TEMPLATE.read_bytes())
    cells = book["cells"]

    def get(i):
        return "".join(cells[i]["source"])

    def put(i, source):
        cells[i]["source"] = source.splitlines(keepends=True)

    archived = verify_inventory(ARCHIVE)
    sources = ast.literal_eval(ast.parse(get(4)).body[0].value)
    assert all(src.encode() == archived[name] for name, src in sources.items())
    prior = json.loads(sources.pop("provenance/droid_attention_cases.json"))
    for path in ("provenance/droid_natural_targets.json", "provenance/droid_natural_manifest.json"):
        sources.pop(path)
    spec.update(model_revision=prior["model_revision"], video_processor=prior["video_processor"])
    for file, source in {
        "provenance/droid_interior_runner.py": "autodl_droid_interior.py",
        "provenance/droid_span_runner.py": "autodl_droid_span_selection.py",
        "provenance/droid_span_selector.py": "droid_span_selector.py",
    }.items():
        sources[file] = (ROOT / "scripts" / source).read_text(encoding="utf-8")
    sources["provenance/droid_interior_cases.json"] = json.dumps(spec, indent=2) + "\n"
    hashes = {name: digest(src.encode()) for name, src in sources.items()}
    put(0, INTRO)
    put(1, get(1).replace("hermes_droid_terminal_attention", "hermes_droid_interior")
        .replace("HERMES_DROID_TERMINAL_ATTENTION_VALIDATE_ONLY", "HERMES_DROID_INTERIOR_VALIDATE_ONLY")
        .replace("provenance/droid_attention_runner.py", "provenance/droid_interior_runner.py")
        .replace("DROID_natural_history_frames.zip", FRAME_ZIP)
        .replace(prior["frame_zip_sha256"], spec["frame_zip_sha256"])
        .replace("7 DROID pairs, 4 attention conditions", "7 DROID clips, 4 attention conditions"))
    put(4, f"SOURCES = {sources!r}\nEXPECTED = {hashes!r}\n" + "\n".join(get(4).split("\n")[2:]))
    put(7, "## 2. Score event-relative evidence on the frozen inputs\n")
    put(8, get(8).replace("HERMES_droid_terminal_attention_evidence.zip", "HERMES_droid_interior_evidence.zip")
        .replace("'droid_terminal_attention.log'", "'droid_interior.log'"))
    for i, cell in enumerate(cells):
        cell["id"] = f"droid-interior-{i:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    return book


if __name__ == "__main__":
    DEST.write_text(json.dumps(notebook(frame_spec()), ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8", newline="\n")
    print(f"Wrote {DEST}")
