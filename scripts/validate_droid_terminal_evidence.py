"""Verify the DROID terminal-state scores and context transitions."""
import ast
from collections import Counter
import json
import math
from pathlib import Path
import sys

from make_droid_terminal_notebook import DEST as TEMPLATE
from validate_droid_interference_evidence import ARCHIVE as PRIOR
from validate_droid_natural_history_evidence import FRAMES, MANIFEST, verify_frame_pack
from validate_twinpast_diagnostic_evidence import LETTERS, ORDERS, ROOT, digest, verify_inventory

ARCHIVE = ROOT / "evidence/archives/HERMES_droid_terminal_context_evidence.zip"
NOTEBOOK = ROOT / "notebooks/executed/2026-11-27_hermes-droid-terminal-context.ipynb"
FRAME_COUNTS = {"terminal_only": 3, "terminal_recent": 7, "full_annotated": 10}
STEPS = (("terminal_only", "terminal_recent", "added_recent"),
         ("terminal_recent", "full_annotated", "added_earlier"))


def audit(archive, notebook):
    saved, prior = verify_inventory(archive), verify_inventory(PRIOR)

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
    assert launch["embedded_sha256"] == {name: digest(saved[name]) for name in embedded}
    spec = load("provenance/droid_terminal_cases.json")
    manifest = json.loads(MANIFEST.read_bytes())
    assert saved["provenance/droid_natural_manifest.json"] == MANIFEST.read_bytes()
    assert spec["manifest_sha256"] == digest(MANIFEST.read_bytes())
    assert config["model_revision"] == spec["model_revision"] == launch["model_revision"]
    assert config["archived_full_zip_sha256"] == spec["archived_full_zip_sha256"] == digest(PRIOR.read_bytes())
    assert config["archived_full_gpu"] == spec["archived_full_gpu"] == json.loads(prior["exports/run_config.json"])["gpu"]
    assert config["frame_zip_sha256"] == spec["frame_zip_sha256"] == manifest["frame_zip_sha256"]
    assert config["replay_atol"] == 0.05 and config["frame_counts"] == FRAME_COUNTS
    assert spec["frame_receipts"] == manifest["frame_receipts"]
    assert b"PINNED_ENV_PASS" in saved["logs/environment.log"]
    assert b"DROID_TERMINAL_CONTEXT_COMPLETE" in saved["logs/droid_terminal_context.log"]
    assert load("completion.json") == dict(status="complete", pairs=7)
    verified_frames = verify_frame_pack(FRAMES, manifest)

    targets = load("provenance/droid_natural_targets.json")
    assert saved["provenance/droid_natural_targets.json"] == prior["provenance/droid_natural_targets.json"]
    cases = {c["pair_id"]: c for c in spec["cases"]}
    originals = {c["pair_id"]: c for c in manifest["cases"]}
    prior_rows = {b["pair_id"]: b for b in json.loads(prior["exports/pair_scores.json"])}
    rows = load("exports/pair_scores.json")
    assert [r["pair_id"] for r in rows] == list(cases) == list(originals)
    assert len(cases) == len(targets) == 7 and set(cases) == set(prior_rows) == set(targets)

    methods = {m: dict(frames=n, robust_pairs=[], correct_branch_orders=0, option_order_flips=0)
               for m, n in FRAME_COUNTS.items()}
    transitions = {stage: Counter() for _, _, stage in STEPS}
    scores, repeats, max_delta = 0, 0, 0.0
    for row in rows:
        pair = row["pair_id"]
        case, original = cases[pair], originals[pair]
        for key in ("question", "options", "recent_frames", "scene_id", "source_site"):
            assert case[key] == original[key]
        assert row["scene_id"] == case["scene_id"] and row["source_site"] == case["source_site"]
        assert len(case["recent_frames"]) == 4 and set(row["branches"]) == {"a", "b"}
        correct = Counter()
        for side, trace in row["branches"].items():
            branch, raw = case["branches"][side], original["branches"][side]
            assert branch["frames"] == raw["frames"]
            indices, terminal = raw["source_indices"]["annotated"], raw["terminal_interval"]
            assert len(indices) == 6 and all(i < terminal[0] for i in indices[:3])
            assert all(terminal[0] <= i <= terminal[1] for i in indices[3:])
            assert trace["terminal_frames"] == raw["frames"]["annotated"][-3:]
            assert not trace["replay_failures"] and set(trace["answers"]) == set(ORDERS)
            gold = targets[pair][side]
            for order, answers in trace["answers"].items():
                assert set(answers) == set(FRAME_COUNTS) and len({r["prompt_sha256"] for r in answers.values()}) == 1
                full, expected = answers["full_annotated"], branch["archived_full"][order]
                earlier = prior_rows[pair]["branches"][side]["answers"][order]["original"]
                assert all(earlier[k] == v for k, v in expected.items())
                assert all(full[k] == expected[k] for k in ("prediction", "prompt_sha256", "visual_tokens", "lm_tokens"))
                delta = max(abs(a-b) for a, b in zip(full["logits"], expected["logits"], strict=True))
                assert delta == full["replay_max_abs"] <= config["replay_atol"]
                repeats += 1
                max_delta = max(max_delta, delta)
                for method, answer in answers.items():
                    assert answer["visual_tokens"] * 10 == full["visual_tokens"] * FRAME_COUNTS[method]
                    assert answer["lm_tokens"]-answer["visual_tokens"] == full["lm_tokens"]-full["visual_tokens"]
                    values = answer["logits"]
                    assert len(values) == len(case["options"]) and all(math.isfinite(x) for x in values)
                    assert answer["prediction"] == LETTERS[max(range(len(values)), key=values.__getitem__)]
                    correct[method] += answer["prediction"] == gold
                    scores += 1
                for left, right, stage in STEPS:
                    before, after = (answers[m]["prediction"] == gold for m in (left, right))
                    transitions[stage]["wrong_to_correct"] += after and not before
                    transitions[stage]["correct_to_wrong"] += before and not after
            for method in methods:
                methods[method]["option_order_flips"] += len({
                    trace["answers"][order][method]["prediction"] for order in ORDERS
                }) > 1
        for method, totals in methods.items():
            totals["correct_branch_orders"] += correct[method]
            if correct[method] == 4:
                totals["robust_pairs"].append(pair)
    contexts = {}
    for left, right, stage in STEPS:
        before, after = (set(methods[m]["robust_pairs"]) for m in (left, right))
        contexts[stage] = dict(gains=sorted(after-before), losses=sorted(before-after))
    assert scores == 84 and repeats == 28
    gate = dict(scores=repeats, max_abs=max_delta, failures=[])
    assert gate == load("exports/replay_gate.json")
    summary = dict(pairs=7, scores=scores, methods=methods, context_transitions=contexts)
    assert summary == load("exports/summary.json")
    return dict(archive_sha256=digest(archive.read_bytes()), notebook_sha256=digest(notebook.read_bytes()),
                verified_payloads=len(saved), verified_input_frames=verified_frames, replay_gate=gate,
                gpu=config["gpu"], archived_full_gpu=config["archived_full_gpu"],
                branch_order_transitions={k: dict(v) for k, v in transitions.items()}, **summary)


if __name__ == "__main__":
    archive, notebook = map(Path, sys.argv[1:]) if len(sys.argv) == 3 else (ARCHIVE, NOTEBOOK)
    result = audit(archive, notebook)
    report = json.dumps(result, indent=2) + "\n"
    if len(sys.argv) == 1:
        (ROOT / "evidence/audits/droid_terminal_context_2026-11-27_audit.json").write_text(
            report, encoding="utf-8", newline="\n")
    print(report, end="")
