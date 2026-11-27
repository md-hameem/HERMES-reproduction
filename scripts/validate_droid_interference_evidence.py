"""Audit the fixed-frame DROID earlier-state masking experiment."""
import ast
from collections import Counter
from io import BytesIO
import json
import math
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image

from droid_state_masks import MASK_RGB, apply_mask
from make_droid_interference_notebook import DEST as TEMPLATE, PLAN
from validate_droid_natural_history_evidence import ARCHIVE as PRIOR, FRAMES, MANIFEST, verify_frame_pack
from validate_twinpast_diagnostic_evidence import LETTERS, ORDERS, ROOT, digest, verify_inventory

ARCHIVE = ROOT / "evidence/archives/HERMES_droid_state_interference_evidence.zip"
NOTEBOOK = ROOT / "notebooks/executed/2026-11-27_hermes-droid-state-interference.ipynb"
METHODS = ("original", "target_mask", "non_target_mask")
MASKS = {"target_mask": "target", "non_target_mask": "control"}


def audit(archive, notebook):
    saved, prior = verify_inventory(archive), verify_inventory(PRIOR)

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
    assert load("launch.json")["embedded_sha256"] == {name: digest(saved[name]) for name in embedded}
    assert saved["provenance/droid_natural_manifest.json"] == MANIFEST.read_bytes()
    assert saved["provenance/droid_interference_mask_plan.json"] == PLAN.read_bytes()
    config = load("exports/run_config.json")
    spec = load("provenance/droid_interference_cases.json")
    plan = json.loads(PLAN.read_bytes())
    manifest = json.loads(MANIFEST.read_bytes())
    assert config["mask_plan_sha256"] == spec["mask_plan_sha256"] == digest(PLAN.read_bytes())
    assert config["natural_history_archive_sha256"] == spec["natural_history_archive_sha256"] == digest(PRIOR.read_bytes())
    assert plan["natural_history_archive_sha256"] == config["natural_history_archive_sha256"]
    assert plan["natural_history_manifest_sha256"] == spec["manifest_sha256"] == digest(MANIFEST.read_bytes())
    assert config["model_revision"] == spec["model_revision"] == load("launch.json")["model_revision"]
    assert config["frame_zip_sha256"] == spec["frame_zip_sha256"] == manifest["frame_zip_sha256"]
    assert spec["mask_fill_rgb"] == plan["fill_rgb"] == list(MASK_RGB)
    assert config["replay_atol"] == 0.05
    assert b"PINNED_ENV_PASS" in saved["logs/environment.log"]
    assert b"DROID_STATE_INTERFERENCE_COMPLETE" in saved["logs/droid_state_interference.log"]
    assert load("completion.json") == dict(status="complete", pairs=7)
    frames_checked = verify_frame_pack(FRAMES, manifest)
    targets = load("provenance/droid_natural_targets.json")
    assert saved["provenance/droid_natural_targets.json"] == prior["provenance/droid_natural_targets.json"]
    prior_scores = {r["pair_id"]: r for r in json.loads(prior["exports/pair_scores.json"])}
    cases = {c["pair_id"]: c for c in spec["cases"]}
    masks = {c["pair_id"]: c for c in plan["cases"]}
    originals = {c["pair_id"]: c for c in manifest["cases"]}
    blocks = load("exports/pair_scores.json")
    assert [r["pair_id"] for r in blocks] == list(cases) == list(originals)
    assert set(cases) == set(masks) == set(prior_scores) == set(targets)

    robust = {m: [] for m in METHODS}
    totals = Counter()
    changes = {m: Counter() for m in METHODS[1:]}
    scored, replays, edited, max_diff = 0, 0, 0, 0.0
    with ZipFile(FRAMES) as pack:
        for block in blocks:
            pair = block["pair_id"]
            case, mask, natural = cases[pair], masks[pair], originals[pair]
            assert case["mask"] == mask and mask["edited_slots"] == [0, 1]
            for key in ("question", "options", "recent_frames", "source_site", "scene_id"):
                assert case[key] == natural[key]
            for key in ("source_site", "scene_id"):
                assert block[key] == natural[key]
            xa, ya, xb, yb = mask["target"]
            ca, da, cb, db = mask["control"]
            assert (xb-xa, yb-ya) == (cb-ca, db-da)
            assert xb <= ca or cb <= xa or yb <= da or db <= ya
            counts = Counter()
            assert set(block["branches"]) == {"a", "b"}
            for side, result in block["branches"].items():
                branch, reference = case["branches"][side], natural["branches"][side]
                assert branch["frames"] == reference["frames"] and not result["replay_failures"]
                slots = mask["edited_slots"]
                history = branch["frames"]["annotated"]
                selected = [history[i] for i in slots]
                assert len(history) == 6 and len(case["recent_frames"]) == 4
                assert set(selected).isdisjoint(case["recent_frames"])
                initial = reference["initial_interval"]
                assert all(initial[0] <= reference["source_indices"]["annotated"][i] <= initial[1] for i in slots)
                for method, kind in MASKS.items():
                    values = {}
                    for name in selected:
                        with Image.open(BytesIO(pack.read(name))) as image:
                            rgb = np.array(image)
                        values[name] = digest(apply_mask(rgb, mask[kind]).tobytes())
                        assert values[name] == mask["edited_rgb_sha256"][side][name][kind]
                        edited += 1
                    assert result["mask_receipts"][method] == values
                assert set(result["answers"]) == set(ORDERS)
                for order, answers in result["answers"].items():
                    assert set(answers) == set(METHODS)
                    score = answers["original"]
                    earlier = prior_scores[pair]["branches"][side]["answers"][order]["annotated"]
                    record = branch["archived_annotated"][order]
                    assert all(earlier[k] == v for k, v in record.items())
                    for key in ("prediction", "prompt_sha256", "visual_tokens", "lm_tokens"):
                        assert score[key] == record[key]
                    for key in ("prompt_sha256", "visual_tokens", "lm_tokens"):
                        assert len({row[key] for row in answers.values()}) == 1
                    delta = max(abs(x-y) for x, y in zip(score["logits"], record["logits"], strict=True))
                    assert delta == score["replay_max_abs"] <= config["replay_atol"]
                    max_diff = max(max_diff, delta)
                    replays += 1
                    gold = targets[pair][side]
                    for method, row in answers.items():
                        values = row["logits"]
                        assert len(values) == len(case["options"]) and all(math.isfinite(n) for n in values)
                        assert row["prediction"] == LETTERS[max(range(len(values)), key=values.__getitem__)]
                        counts[method] += row["prediction"] == gold
                        scored += 1
                        if method in changes:
                            before, after = score["prediction"] == gold, row["prediction"] == gold
                            changes[method]["wrong_to_correct"] += after and not before
                            changes[method]["correct_to_wrong"] += before and not after
            for method in METHODS:
                totals[method] += counts[method]
                if counts[method] == 4:
                    robust[method].append(pair)
    assert scored == 84 and replays == 28 and edited == 56
    gate = dict(scores=replays, max_abs=max_diff, failures=[])
    assert gate == load("exports/replay_gate.json")
    original, target, control = (set(robust[m]) for m in METHODS)
    recovered, lost_original, lost_control = sorted(target-original-control), sorted(original-target), sorted(control-target)
    summary = dict(pairs=7, scores=scored, target_only_recoveries=recovered, lost_vs_original=lost_original,
                   lost_vs_non_target=lost_control,
                   exploratory_criterion_met=len(recovered) >= 2 and not lost_original and not lost_control,
                   methods={m: dict(robust_pairs=robust[m], correct_branch_orders=totals[m]) for m in METHODS})
    assert summary == load("exports/summary.json")
    return dict(archive_sha256=digest(archive.read_bytes()), notebook_sha256=digest(notebook.read_bytes()),
                verified_payloads=len(saved), verified_input_frames=frames_checked, verified_masked_frames=edited,
                gpu=config["gpu"], prior_gpu=json.loads(prior["exports/run_config.json"])["gpu"], replay_gate=gate,
                branch_order_changes_against_original={m: dict(c) for m, c in changes.items()}, **summary)


if __name__ == "__main__":
    archive, notebook = map(Path, sys.argv[1:]) if len(sys.argv) == 3 else (ARCHIVE, NOTEBOOK)
    report = json.dumps(audit(archive, notebook), indent=2) + "\n"
    if len(sys.argv) == 1:
        (ROOT / "evidence/audits/droid_state_interference_2026-11-27_audit.json").write_text(
            report, encoding="utf-8", newline="\n")
    print(report, end="")
