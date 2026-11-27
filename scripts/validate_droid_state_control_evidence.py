"""Audit the DROID source frame receipts and recompute the paired GPU outcomes."""
import ast
import base64
from collections import Counter
from io import BytesIO
import json
import math
from pathlib import Path
import sys

from PIL import Image

from validate_twinpast_diagnostic_evidence import LETTERS, ORDERS, ROOT, digest, verify_inventory

ARCHIVE = ROOT / "evidence/archives/HERMES_droid_state_control_evidence.zip"
NOTEBOOK = ROOT / "notebooks/executed/2026-11-27_hermes-droid-state-control.ipynb"
TEMPLATE = ROOT / "notebooks/autodl/hermes-droid-state-control.ipynb"
MANIFEST = ROOT / "data/twinpast/droid_state_control_v1.json"
METHODS = ("aligned_uniform", "off_grid_uniform", "off_grid_annotated", "cue_only")


def audit(archive, notebook):
    saved = verify_inventory(archive)

    def load(name):
        return json.loads(saved[name])

    cells = json.loads(notebook.read_bytes())["cells"]
    template = json.loads(TEMPLATE.read_bytes())["cells"]
    assert [(c["cell_type"], c["source"]) for c in cells] == [(c["cell_type"], c["source"]) for c in template]
    for i, cell in enumerate((c for c in cells if c["cell_type"] == "code"), 1):
        assert cell["execution_count"] == i and all(o["output_type"] != "error" for o in cell["outputs"])

    source, = ("".join(c["source"]) for c in cells if "".join(c["source"]).startswith("SOURCES = "))
    embedded = ast.literal_eval(ast.parse(source).body[0].value)
    assert all(raw.encode() == saved[name] for name, raw in embedded.items())
    launch, config = load("launch.json"), load("exports/run_config.json")
    assert launch["embedded_sha256"] == {n: digest(saved[n]) for n in embedded}
    manifest = json.loads(MANIFEST.read_bytes())
    assert saved["provenance/droid_source_manifest.json"] == MANIFEST.read_bytes()
    spec = load("provenance/droid_cases.json")
    assert config["manifest_sha256"] == spec["manifest_sha256"] == digest(MANIFEST.read_bytes())
    assert config["model_revision"] == spec["model_revision"] == launch["model_revision"]
    assert spec["timeline"] == manifest["timeline"]
    assert spec["cases"] == [{k: c[k] for k in ("pair_id", "source_group", "question", "options")}
                             for c in manifest["cases"]]
    assert b"PINNED_ENV_PASS" in saved["logs/environment.log"]
    assert b"DROID_STATE_CONTROL_COMPLETE" in saved["logs/droid_state_control.log"]
    assert load("completion.json") == dict(status="complete", pairs=6)

    frame_cell, = ("".join(c["source"]) for c in cells if "".join(c["source"]).startswith("INPUT_FRAMES = "))
    pngs = ast.literal_eval(ast.parse(frame_cell).body[0].value)
    receipts = load("provenance/droid_frame_receipts.json")
    cases = {case["pair_id"]: case for case in manifest["cases"]}
    assert set(pngs) == set(receipts) == {f"{p}_{b}.png" for p in cases for b in "ab"}
    assert len(cases) == len({c["source_group"] for c in cases.values()}) == len({
        c["scene_id"] for c in cases.values()
    }) == 6
    for case in cases.values():
        for side, source in case["branches"].items():
            name = f"{case['pair_id']}_{side}.png"
            raw = base64.b64decode(pngs[name], validate=True)
            r = receipts[name]
            assert r["png_sha256"] == digest(raw)
            assert r["source_episode"] == case["source_episode"] and r["source_frame"] == source["source_frame"]
            with Image.open(BytesIO(raw)) as image:
                rgb = image.convert("RGB")
                assert r["shape"] == [rgb.height, rgb.width, 3] == [360, 640, 3]
                assert r["rgb_sha256"] == digest(rgb.tobytes())

    targets = load("provenance/droid_targets.json")
    assert targets == {pair: {b: trace["gold"] for b, trace in case["branches"].items()}
                       for pair, case in cases.items()}
    assert all(t["a"] != t["b"] for t in targets.values())
    blocks = load("exports/pair_scores.json")
    assert [b["pair_id"] for b in blocks] == list(cases)
    correct, robust, flips, hidden, checked = Counter(), {m: [] for m in METHODS}, Counter(), Counter(), 0
    for block in blocks:
        pair = block["pair_id"]
        assert block["source_group"] == cases[pair]["source_group"]
        rows = block["scores"]
        assert set(rows) == {"a", "b"}
        hits = Counter()
        for side, orders in rows.items():
            assert set(orders) == set(ORDERS)
            for order, methods in orders.items():
                assert set(methods) == set(METHODS)
                assert len({row["prompt_sha256"] for row in methods.values()}) == 1
                assert len({methods[m]["visual_tokens"] for m in METHODS[:3]}) == 1
                assert methods["cue_only"]["visual_tokens"] * 10 == methods["off_grid_annotated"]["visual_tokens"]
                assert methods["aligned_uniform"]["logits"] == methods["off_grid_annotated"]["logits"]
                assert methods["off_grid_uniform"]["logits"] == rows["a"][order]["off_grid_uniform"]["logits"]
                hidden[methods["off_grid_uniform"]["prediction"]] += 1
                for method, result in methods.items():
                    logits = result["logits"]
                    assert len(logits) == len(cases[pair]["options"]) and all(math.isfinite(v) for v in logits)
                    assert result["prediction"] == LETTERS[max(range(len(logits)), key=logits.__getitem__)]
                    hits[method] += result["prediction"] == targets[pair][side]
                    checked += 1
            for method in METHODS:
                flips[method] += len({orders[o][method]["prediction"] for o in ORDERS}) > 1
        for method in METHODS:
            correct[method] += hits[method]
            if hits[method] == 4:
                robust[method].append(pair)
    assert len(blocks) == 6 and checked == 96 and not robust["off_grid_uniform"]
    summary = dict(pairs=6, scores=checked, methods={m: dict(robust_pairs=robust[m], correct_branch_orders=correct[m])
                                                    for m in METHODS},
                   controlled_capability_criterion_met=len(robust["off_grid_annotated"]) >= 4)
    assert summary == load("exports/summary.json")
    return dict(archive_sha256=digest(archive.read_bytes()), notebook_sha256=digest(notebook.read_bytes()),
                verified_payloads=len(saved), verified_input_frames=len(pngs), option_order_flip_branches=dict(flips),
                hidden_cue_predictions=dict(hidden), **summary)


if __name__ == "__main__":
    archive, notebook = map(Path, sys.argv[1:]) if len(sys.argv) == 3 else (ARCHIVE, NOTEBOOK)
    result = audit(archive, notebook)
    report = json.dumps(result, indent=2) + "\n"
    if len(sys.argv) == 1:
        (ROOT / "evidence/audits/droid_state_control_2026-11-27_audit.json").write_text(
            report, encoding="utf-8", newline="\n")
    print(report, end="")
