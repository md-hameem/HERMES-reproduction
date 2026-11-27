"""Verify the DROID natural-history run and recompute matched outcomes."""
import ast
from collections import Counter
from io import BytesIO
import json
import math
from pathlib import Path
import sys
from zipfile import ZipFile

from PIL import Image

from validate_twinpast_diagnostic_evidence import LETTERS, ORDERS, ROOT, digest, verify_inventory

ARCHIVE = ROOT / "evidence/archives/HERMES_droid_natural_history_evidence.zip"
NOTEBOOK = ROOT / "notebooks/executed/2026-11-27_hermes-droid-natural-history.ipynb"
TEMPLATE = ROOT / "notebooks/autodl/hermes-droid-natural-history.ipynb"
MANIFEST = ROOT / "data/twinpast/droid_natural_history_v1.json"
FRAMES = ROOT / "evidence/local/droid-natural-history/DROID_natural_history_frames.zip"
METHODS = ("uniform", "annotated")


def verify_frame_pack(frame_pack, manifest):
    receipts = manifest["frame_receipts"]
    assert digest(frame_pack.read_bytes()) == manifest["frame_zip_sha256"]
    with ZipFile(frame_pack) as file:
        assert len(file.namelist()) == len(set(file.namelist())) and set(file.namelist()) == set(receipts)
        for name, receipt in receipts.items():
            raw = file.read(name)
            assert digest(raw) == receipt["png_sha256"]
            with Image.open(BytesIO(raw)) as image:
                assert image.size == (640, 360) and image.mode == "RGB"
                assert digest(image.tobytes()) == receipt["rgb_sha256"]
    return len(receipts)


def audit(archive, notebook, frame_pack):
    saved = verify_inventory(archive)

    def load(name):
        return json.loads(saved[name])

    cells = json.loads(notebook.read_bytes())["cells"]
    expected = json.loads(TEMPLATE.read_bytes())["cells"]
    assert [(c["cell_type"], c["source"]) for c in cells] == [(c["cell_type"], c["source"]) for c in expected]
    for i, cell in enumerate((c for c in cells if c["cell_type"] == "code"), 1):
        assert cell["execution_count"] == i and all(o["output_type"] != "error" for o in cell["outputs"])
    source, = ("".join(c["source"]) for c in cells if "".join(c["source"]).startswith("SOURCES = "))
    embedded = ast.literal_eval(ast.parse(source).body[0].value)
    assert all(raw.encode() == saved[name] for name, raw in embedded.items())
    launch, config = load("launch.json"), load("exports/run_config.json")
    assert launch["embedded_sha256"] == {name: digest(saved[name]) for name in embedded}
    spec = load("provenance/droid_natural_cases.json")
    manifest = json.loads(MANIFEST.read_bytes())
    assert saved["provenance/droid_natural_manifest.json"] == MANIFEST.read_bytes()
    assert config["manifest_sha256"] == spec["manifest_sha256"] == digest(MANIFEST.read_bytes())
    assert config["model_revision"] == spec["model_revision"] == launch["model_revision"]
    assert config["frame_zip_sha256"] == spec["frame_zip_sha256"] == manifest["frame_zip_sha256"]
    assert spec["frame_receipts"] == manifest["frame_receipts"]
    verified_frames = verify_frame_pack(frame_pack, manifest)
    assert b"PINNED_ENV_PASS" in saved["logs/environment.log"]
    assert b"DROID_NATURAL_INPUTS_VERIFIED 114" in saved["logs/droid_natural_history.log"]
    assert b"DROID_NATURAL_HISTORY_COMPLETE" in saved["logs/droid_natural_history.log"]
    assert load("completion.json") == dict(status="complete", pairs=7)

    cases = {c["pair_id"]: c for c in manifest["cases"]}
    assert len(cases) == len({(c["source_site"], c["scene_id"]) for c in cases.values()}) == 7
    assert len({c["source_site"] for c in cases.values()}) == 4
    assert len({b["source_episode"] for c in cases.values() for b in c["branches"].values()}) == 14
    targets = load("provenance/droid_natural_targets.json")
    assert targets == {p: {s: row["gold"] for s, row in c["branches"].items()} for p, c in cases.items()}
    assert all(set(t.values()) == {"A", "B"} for t in targets.values())
    recent = manifest["cases"][0]["recent_frames"]
    assert len(recent) == len(set(recent)) == 4
    used = set(recent)
    for row in spec["cases"]:
        case = cases[row["pair_id"]]
        assert {k: row[k] for k in ("source_site", "scene_id", "question", "options")} == {
            k: case[k] for k in ("source_site", "scene_id", "question", "options")
        }
        assert row["recent_frames"] == case["recent_frames"] == recent
        for side, branch in case["branches"].items():
            terminal = branch["terminal_interval"]
            assert row["branches"][side]["frames"] == branch["frames"]
            for method in METHODS:
                paths, ticks = branch["frames"][method], branch["source_indices"][method]
                assert len(paths) == len(set(paths)) == len(ticks) == 6
                assert ticks == sorted(set(ticks)) and ticks[0] == 0 and ticks[-1] == terminal[1]
                assert set(paths).isdisjoint(recent)
                assert all(name.endswith(f"_{i:05}.png") for name, i in zip(paths, ticks, strict=True))
                covered = sum(terminal[0] <= i <= terminal[1] for i in ticks)
                assert covered in ((1, 2) if method == "uniform" else (3,))
                used.update(paths)
    assert used == set(manifest["frame_receipts"])

    rows = load("exports/pair_scores.json")
    assert [row["pair_id"] for row in rows] == list(cases) == [row["pair_id"] for row in spec["cases"]]
    correct, labels, flips = Counter(), {m: Counter() for m in METHODS}, Counter()
    robust = {m: [] for m in METHODS}
    change = Counter()
    scored, replays = 0, 0
    for row in rows:
        pair = row["pair_id"]
        assert row["scene_id"] == cases[pair]["scene_id"] and row["source_site"] == cases[pair]["source_site"]
        assert set(row["branches"]) == {"a", "b"}
        pair_correct = Counter()
        for side, trace in row["branches"].items():
            assert trace["repeat_max_abs"] == dict.fromkeys(ORDERS, 0.0) and set(trace["answers"]) == set(ORDERS)
            replays += len(trace["repeat_max_abs"])
            for order, results in trace["answers"].items():
                assert set(results) == set(METHODS)
                for k in ("prompt_sha256", "visual_tokens", "lm_tokens"):
                    assert results["uniform"][k] == results["annotated"][k]
                gold = targets[pair][side]
                before, after = (results[m]["prediction"] == gold for m in METHODS)
                change["wrong_to_correct"] += after and not before
                change["correct_to_wrong"] += before and not after
                for method, choice in results.items():
                    logits = choice["logits"]
                    assert len(logits) == len(cases[pair]["options"]) and all(math.isfinite(v) for v in logits)
                    assert choice["prediction"] == LETTERS[max(range(len(logits)), key=logits.__getitem__)]
                    pair_correct[method] += choice["prediction"] == gold
                    labels[method][choice["prediction"]] += 1
                    scored += 1
            for method in METHODS:
                flips[method] += len({trace["answers"][o][method]["prediction"] for o in ORDERS}) > 1
        for method in METHODS:
            correct[method] += pair_correct[method]
            if pair_correct[method] == 4:
                robust[method].append(pair)
    assert scored == 56 and replays == 28
    recovered = sorted(set(robust["annotated"]) - set(robust["uniform"]))
    lost = sorted(set(robust["uniform"]) - set(robust["annotated"]))
    summary = dict(pairs=7, scores=scored, uniform_replays=replays,
                   methods={m: dict(robust_pairs=robust[m], correct_branch_orders=correct[m]) for m in METHODS},
                   recoveries=recovered, regressions=lost, exploratory_criterion_met=len(recovered) >= 2 and not lost)
    assert summary == load("exports/summary.json")
    return dict(archive_sha256=digest(archive.read_bytes()), notebook_sha256=digest(notebook.read_bytes()),
                verified_payloads=len(saved), verified_input_frames=verified_frames,
                branch_order_changes=dict(change), canonical_prediction_counts={m: dict(c) for m, c in labels.items()},
                option_order_flip_branches=dict(flips), **summary)


if __name__ == "__main__":
    archive, notebook = map(Path, sys.argv[1:]) if len(sys.argv) == 3 else (ARCHIVE, NOTEBOOK)
    report = json.dumps(audit(archive, notebook, FRAMES), indent=2) + "\n"
    if len(sys.argv) == 1:
        (ROOT / "evidence/audits/droid_natural_history_2026-11-27_audit.json").write_text(
            report, encoding="utf-8", newline="\n")
    print(report, end="")
