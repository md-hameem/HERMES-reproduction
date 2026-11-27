"""Audit the DROID terminal-attention run against the frozen notebook and inputs."""
import ast
from collections import Counter
import json
from math import isfinite
from pathlib import Path
from statistics import median
import sys

from make_droid_terminal_attention_notebook import DEST as TEMPLATE
from validate_droid_state_swap_evidence import ARCHIVE as PRIOR, FRAMES, MANIFEST, ROOT, digest
from validate_droid_state_swap_evidence import LETTERS, ORDERS, verify_frame_pack, verify_inventory

ARCHIVE = ROOT / "evidence/archives/HERMES_droid_terminal_attention_evidence.zip"
NOTEBOOK = ROOT / "notebooks/executed/2026-11-27_hermes-droid-terminal-attention.ipynb"
VISIBLE = {"causal_control": list(range(10)), "terminal_access": [3, 4, 5], "earlier_access": [0, 1, 2]}
METHODS = ("native", *VISIBLE)
IDENTITY = ("prediction", "prompt_sha256", "lm_tokens", "visual_tokens")


def audit(archive, notebook):
    saved = verify_inventory(archive)
    prior = verify_inventory(PRIOR)

    def load(name):
        return json.loads(saved[name])

    cells = json.loads(notebook.read_bytes())["cells"]
    assert [(c["cell_type"], c["source"]) for c in cells] == [
        (c["cell_type"], c["source"]) for c in json.loads(TEMPLATE.read_bytes())["cells"]
    ]
    for i, cell in enumerate((c for c in cells if c["cell_type"] == "code"), 1):
        assert cell["execution_count"] == i and all(o["output_type"] != "error" for o in cell["outputs"])
    embedded = ast.literal_eval(ast.parse("".join(cells[4]["source"])).body[0].value)
    assert all(source.encode() == saved[name] for name, source in embedded.items())
    launch, config = load("launch.json"), load("exports/run_config.json")
    assert launch["embedded_sha256"] == {name: digest(saved[name]) for name in embedded}
    assert config["attention_backend"] in ("sdpa", "eager") and config["visible_visual_slots"] == VISIBLE
    spec = load("provenance/droid_attention_cases.json")
    assert spec["prior_archive_sha256"] == config["prior_archive_sha256"] == digest(PRIOR.read_bytes())
    assert launch["model_revision"] == config["model_revision"] == spec["model_revision"]
    manifest = json.loads(MANIFEST.read_bytes())
    assert saved["provenance/droid_natural_manifest.json"] == MANIFEST.read_bytes()
    assert spec["manifest_sha256"] == digest(MANIFEST.read_bytes())
    assert spec["frame_receipts"] == manifest["frame_receipts"]
    assert config["frame_zip_sha256"] == spec["frame_zip_sha256"] == manifest["frame_zip_sha256"]
    verified_frames = verify_frame_pack(FRAMES, manifest)
    assert saved["provenance/droid_natural_targets.json"] == prior["provenance/droid_natural_targets.json"]
    targets = load("provenance/droid_natural_targets.json")
    assert b"PINNED_ENV_PASS" in saved["logs/environment.log"]
    assert b"DROID_TERMINAL_ATTENTION_COMPLETE" in saved["logs/droid_terminal_attention.log"]
    assert load("completion.json") == dict(status="complete", pairs=7)

    blocks = load("exports/pair_scores.json")
    previous = {row["pair_id"]: row for row in json.loads(prior["exports/pair_scores.json"])}
    assert len(blocks) == len(spec["cases"]) == len(manifest["cases"]) == len(previous) == 7
    methods = {m: dict(robust_pairs=[], correct_branch_orders=0) for m in METHODS}
    deltas, shifts = [], Counter()
    maxima = dict(archived_replay=0.0, explicit_control=0.0)
    score_count = checks = 0
    per_pair = {}
    for block, case, frozen in zip(blocks, spec["cases"], manifest["cases"], strict=True):
        pair = block["pair_id"]
        assert all(block[k] == case[k] == frozen[k] for k in ("pair_id", "source_site", "scene_id"))
        assert all(case[k] == frozen[k] for k in ("question", "options", "recent_frames"))
        assert set(block["branches"]) == set(case["branches"]) == set(frozen["branches"]) == {"a", "b"}
        correct = Counter()
        for side, branch in block["branches"].items():
            source = frozen["branches"][side]
            assert case["branches"][side]["frames"] == source["frames"]
            lo, hi = source["terminal_interval"]
            indices = source["source_indices"]["annotated"]
            assert len(indices) == 6 and all(i < lo for i in indices[:3]) and all(lo <= i <= hi for i in indices[3:])
            gold = targets[pair][side]
            assert gold == source["gold"]
            index = LETTERS.index(gold)
            assert set(branch["answers"]) == set(ORDERS)
            expected_checks = []
            for order, scores in branch["answers"].items():
                assert set(scores) == set(METHODS)
                native, causal, terminal, _ = (scores[m] for m in METHODS)
                archived = previous[pair]["branches"][side]["answers"][order]["original"]
                baked = case["branches"][side]["archived_native"][order]
                assert baked == {k: archived[k] for k in (*IDENTITY, "logits")}
                for method, row in scores.items():
                    logits = row["logits"]
                    assert len(logits) == 2 and all(isfinite(v) for v in logits)
                    assert row["prediction"] == LETTERS[max(range(2), key=logits.__getitem__)]
                    assert all(row[k] == native[k] for k in IDENTITY[1:])
                    assert row.get("visible_frames") == VISIBLE.get(method)
                    assert row["visual_tokens"] == 1960 and row["lm_tokens"] > 1960
                    correct[method] += row["prediction"] == gold
                    score_count += 1
                for name, answer, reference in (("archived_replay", native, baked), ("explicit_control", causal, native)):
                    assert all(answer[k] == reference[k] for k in IDENTITY)
                    error = max(abs(a-b) for a, b in zip(answer["logits"], reference["logits"], strict=True))
                    assert error <= config["replay_atol"] == 0.05
                    maxima[name] = max(maxima[name], error)
                    expected_checks.append(dict(gate=name, order=order, max_abs=error, passed=True))
                    checks += 1
                before, after = (r["prediction"] == gold for r in (causal, terminal))
                shifts["wrong_to_correct"] += after and not before
                shifts["correct_to_wrong"] += before and not after
                x, y = (r["logits"][index] - r["logits"][1-index] for r in (causal, terminal))
                deltas.append(y-x)
            assert branch["checks"] == expected_checks
        per_pair[pair] = {m: correct[m] for m in METHODS}
        for method in METHODS:
            methods[method]["correct_branch_orders"] += correct[method]
            if correct[method] == 4:
                methods[method]["robust_pairs"].append(pair)

    assert score_count == 112 and checks == 56 and len(deltas) == 28
    gate = dict(comparisons=checks, max_abs=maxima, verified_attention_layers=dict.fromkeys(VISIBLE, 2), failures=[])
    assert gate == load("exports/replay_gate.json")
    causal, terminal, earlier = (set(methods[m]["robust_pairs"]) for m in VISIBLE)
    gains, losses = sorted(terminal - causal - earlier), sorted(causal - terminal)
    summary = dict(pairs=len(blocks), scores=score_count, methods=methods, terminal_only_recoveries=gains,
                   lost_vs_causal_control=losses, median_terminal_gold_margin_delta=median(deltas),
                   feasibility_criterion_met=len(gains) >= 2 and not losses)
    assert summary == load("exports/summary.json")
    return dict(archive_sha256=digest(archive.read_bytes()), notebook_sha256=digest(notebook.read_bytes()),
                verified_payloads=len(saved), verified_input_frames=verified_frames, gpu=config["gpu"],
                replay_gate=gate, branch_order_changes=dict(shifts), per_pair_correct_branch_orders=per_pair, **summary)


if __name__ == "__main__":
    archive, notebook = map(Path, sys.argv[1:]) if len(sys.argv) == 3 else (ARCHIVE, NOTEBOOK)
    report = json.dumps(audit(archive, notebook), indent=2) + "\n"
    if len(sys.argv) == 1:
        (ROOT / "evidence/audits/droid_terminal_attention_2026-11-27_audit.json").write_text(
            report, encoding="utf-8", newline="\n")
    print(report, end="")
