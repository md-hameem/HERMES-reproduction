"""Embed the frozen fresh-pair manifest and matched temporal-selector experiment."""
import ast
import json

from validate_twinpast_automatic_evidence import ARCHIVE, ROOT, TEMPLATE, digest, verify_inventory
from prepare_twinpast_temporal_development import DEST as MANIFEST

DEST = ROOT / "notebooks/autodl/hermes-twinpast-temporal-windows.ipynb"
INTRO = """# TwinPast temporal endpoint selection

Use **Restart Kernel → Run All** on the same AutoDL RTX 4090 instance. The notebook reuses
the pinned 7B model, environment and any verified prior videos. This run must download
new evaluation clips. It freezes twelve fresh historical-state candidates in hash order
and evaluates up to eight that pass the media screen. All twelve inspected development
pairs are excluded and used as duplicate-screen references.

TwinPast does not provide original source IDs at this revision. The notebook excludes
sampled near-duplicate footage and checks exact equality of the four recent frames.
Treat results as a fresh-pair development screen with unresolved source provenance.
Passing the screen cannot establish source-disjoint confirmation.

Each branch uses its own causal 1 FPS prefix. The matched comparison scores up to twelve
three-second windows, using both yes/no orders and the question alone. The endpoint method
keeps the ends of three non-overlapping windows, maximizing summed relevance. Individual
ranking uses the same probe scores but can split windows. Both use six history frames.
Uniform history uses the same 1 FPS pool and six-frame budget. All three add the same four
recent frames and use direct forced-choice scoring under both label orders.

Gold answers are loaded after the predictions are saved. The pilot asks whether endpoint
selection produces more robust pairs than both matched dense baselines. A robust pair
requires both branches correct under both option orders. Inspect the cost and regressions
before advancing. Keep these development pairs out of later confirmation.

After **TWINPAST_TEMPORAL_WINDOWS_COMPLETE**, save the executed notebook and download
**HERMES_twinpast_temporal_windows_evidence.zip**.
"""


def notebook():
    prior = verify_inventory(ARCHIVE)
    auto = json.loads(prior["provenance/automatic_cases.json"])
    manifest = json.loads(MANIFEST.read_bytes())
    targets, cases = {}, []
    for annotation in manifest["candidates"]:
        pair = annotation["pair_id"]
        assert pair not in manifest["excluded_development_ids"] and annotation["task"] == "HSR"
        assert annotation["answer_a"] != annotation["answer_b"]
        cases.append(dict(pair_id=pair, query_s=annotation["t_query"], question=annotation["question"],
                          options=annotation["options"]))
        targets[pair] = {k: annotation[k] for k in ("answer_a", "answer_b")}
    spec = dict(model_revision=auto["model_revision"], video_processor=auto["video_processor"], cases=cases,
                manifest_sha256=digest(MANIFEST.read_bytes()), automatic_archive_sha256=digest(ARCHIVE.read_bytes()),
                source_status=manifest["source_status"], expected_max_pairs=manifest["expected_max_pairs"],
                excluded=manifest["excluded_development_ids"], media_plan=manifest["media_plan"],
                media_source={"manifest": {k: manifest[k] for k in ("dataset", "revision")}})
    book = json.loads(TEMPLATE.read_bytes())
    cells = book["cells"]

    def get(i):
        return "".join(cells[i]["source"])

    def put(i, value):
        cells[i]["source"] = value.splitlines(keepends=True)

    sources = ast.literal_eval(ast.parse(get(4)).body[0].value)
    assert all(s.encode() == prior[n] for n, s in sources.items())
    sources.pop("provenance/automatic_cases.json")
    sources["provenance/temporal_cases.json"] = json.dumps(spec, ensure_ascii=False, indent=2) + "\n"
    sources["provenance/evaluation_targets.json"] = json.dumps(targets, sort_keys=True, indent=2) + "\n"
    sources["provenance/temporal_runner.py"] = (
        ROOT / "scripts/autodl_twinpast_temporal_windows.py").read_text(encoding="utf-8")
    hashes = {n: digest(s.encode()) for n, s in sources.items()}
    total = sum(x["bytes"] for _, media, _ in manifest["media_plan"] for x in media["decoded"].values())
    new_media = sum(x["bytes"] for plan, media, _ in manifest["media_plan"]
                    if plan["pair_id"] not in manifest["excluded_development_ids"] for x in media["decoded"].values())
    put(0, INTRO + f"\nThe twelve new candidates total {new_media / 1024**2:.0f} MiB. "
        f"With no prior clips cached, the full media plan totals {total / 1024**2:.0f} MiB.\n")
    put(1, get(1).replace("/hermes_twinpast_automatic", "/hermes_twinpast_temporal_windows")
        .replace("HERMES_TWINPAST_AUTOMATIC_VALIDATE_ONLY", "HERMES_TWINPAST_TEMPORAL_VALIDATE_ONLY")
        .replace("provenance/automatic_runner.py", "provenance/temporal_runner.py")
        .replace("7 pairs, 6 automatic method outcomes per branch and order", "Up to 8 fresh HSR pairs"))
    put(4, f"SOURCES = {sources!r}\nEXPECTED = {hashes!r}\n" + "\n".join(get(4).split("\n")[2:]))
    put(7, "## 2. Screen clips and score the matched temporal selectors\n")
    put(8, get(8).replace("HERMES_twinpast_automatic_evidence.zip", "HERMES_twinpast_temporal_windows_evidence.zip")
        .replace("'twinpast_automatic.log'", "'twinpast_temporal_windows.log'"))
    for i, cell in enumerate(cells):
        cell["id"] = f"twinpast-temporal-{i:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    return book


if __name__ == "__main__":
    DEST.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST}")
