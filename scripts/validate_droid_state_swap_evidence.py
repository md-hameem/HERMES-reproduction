"""Verify fixed-length DROID frame substitutions and paired score changes."""
import ast
from collections import Counter
import json
import math
from pathlib import Path
from statistics import median
import sys

from make_droid_state_swap_notebook import DEST as TEMPLATE, PLAN
from validate_droid_natural_history_evidence import FRAMES, MANIFEST, verify_frame_pack
from validate_droid_terminal_evidence import ARCHIVE as PRIOR
from validate_twinpast_diagnostic_evidence import LETTERS, ORDERS, ROOT, digest, verify_inventory

ARCHIVE = ROOT / "evidence/archives/HERMES_droid_state_swap_evidence.zip"
NOTEBOOK = ROOT / "notebooks/executed/2026-11-27_hermes-droid-state-swap.ipynb"
METHODS = ("original", "swapped")


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
    assert all(text.encode() == saved[name] for name, text in embedded.items())
    launch, config = load("launch.json"), load("exports/run_config.json")
    assert launch["embedded_sha256"] == {name: digest(saved[name]) for name in embedded}
    spec, plan = load("provenance/droid_swap_cases.json"), load("provenance/droid_swap_plan.json")
    manifest = json.loads(MANIFEST.read_bytes())
    assert saved["provenance/droid_natural_manifest.json"] == MANIFEST.read_bytes()
    assert saved["provenance/droid_swap_plan.json"] == PLAN.read_bytes()
    assert spec["manifest_sha256"] == plan["natural_history_manifest_sha256"] == digest(MANIFEST.read_bytes())
    assert config["swap_plan_sha256"] == spec["swap_plan_sha256"] == digest(PLAN.read_bytes())
    assert (config["prior_archive_sha256"] == spec["prior_archive_sha256"]
            == plan["terminal_context_archive_sha256"] == digest(PRIOR.read_bytes()))
    assert config["model_revision"] == spec["model_revision"] == launch["model_revision"]
    assert config["prior_gpu"] == spec["prior_gpu"] == json.loads(prior["exports/run_config.json"])["gpu"]
    assert config["frame_zip_sha256"] == spec["frame_zip_sha256"] == plan["frame_zip_sha256"] == manifest["frame_zip_sha256"]
    assert spec["frame_receipts"] == manifest["frame_receipts"] and config["replay_atol"] == 0.05
    verified_frames = verify_frame_pack(FRAMES, manifest)
    assert b"PINNED_ENV_PASS" in saved["logs/environment.log"]
    assert b"DROID_STATE_SWAP_COMPLETE" in saved["logs/droid_state_swap.log"]
    assert load("completion.json") == dict(status="complete", pairs=7)

    cases = {r["pair_id"]: r for r in spec["cases"]}
    originals = {r["pair_id"]: r for r in manifest["cases"]}
    frozen = {r["pair_id"]: r for r in plan["cases"]}
    previous = {r["pair_id"]: r for r in json.loads(prior["exports/pair_scores.json"])}
    targets = load("provenance/droid_natural_targets.json")
    assert saved["provenance/droid_natural_targets.json"] == prior["provenance/droid_natural_targets.json"]
    blocks = load("exports/pair_scores.json")
    assert [b["pair_id"] for b in blocks] == list(cases) == list(originals)
    assert set(cases) == set(frozen) == set(previous) == set(targets) and len(cases) == 7
    robust, total = {m: [] for m in METHODS}, Counter()
    paired_answers = Counter()
    margins = []
    score_count, repeats, delta_max = 0, 0, 0.0
    for block in blocks:
        pair = block["pair_id"]
        case, raw, swap = cases[pair], originals[pair], frozen[pair]
        assert case["swap"] == swap and swap["replaced_history_slots"] == [0, 1]
        assert case["recent_frames"] == raw["recent_frames"] == swap["shared_recent_frames"]
        assert len(case["recent_frames"]) == 4
        for key in ("source_site", "scene_id", "question", "options"):
            assert case[key] == raw[key]
        assert set(block["branches"]) == {"a", "b"}
        counts = Counter()
        for side, donor in (("a", "b"), ("b", "a")):
            branch, fixed = case["branches"][side], swap["branches"][side]
            trace = block["branches"][side]
            assert branch["frames"] == raw["branches"][side]["frames"]
            annotated = branch["frames"]["annotated"]
            other = raw["branches"][donor]["frames"]["annotated"]
            assert len(annotated) == len(set(annotated)) == 6
            assert fixed["donor_branch"] == donor and fixed["original_history"] == annotated
            assert fixed["swapped_history"] == other[:2]+annotated[2:]
            assert all(a != b for a, b in zip(annotated[:2], other[:2], strict=True))
            for label in (side, donor):
                src = raw["branches"][label]
                lo, hi = src["initial_interval"]
                assert all(lo <= i <= hi for i in src["source_indices"]["annotated"][:2])
            assert not trace["replay_failures"] and set(trace["answers"]) == set(ORDERS)
            gold = targets[pair][side]
            assert gold != targets[pair][donor]
            g = LETTERS.index(gold)
            for order, by_method in trace["answers"].items():
                assert set(by_method) == set(METHODS)
                left, right = (by_method[m] for m in METHODS)
                expected = branch["archived_full"][order]
                prior_full = previous[pair]["branches"][side]["answers"][order]["full_annotated"]
                assert all(prior_full[k] == v for k, v in expected.items())
                for key in ("prediction", "prompt_sha256", "lm_tokens", "visual_tokens"):
                    assert left[key] == expected[key]
                assert all(left[k] == right[k] for k in ("prompt_sha256", "visual_tokens", "lm_tokens"))
                delta = max(abs(a-b) for a, b in zip(left["logits"], expected["logits"], strict=True))
                assert delta == left["replay_max_abs"] <= config["replay_atol"]
                delta_max = max(delta_max, delta)
                repeats += 1
                for m, answer in by_method.items():
                    values = answer["logits"]
                    assert len(values) == 2 and all(math.isfinite(x) for x in values)
                    assert answer["prediction"] == LETTERS[max(range(2), key=values.__getitem__)]
                    counts[m] += answer["prediction"] == gold
                    score_count += 1
                before, after = (x["prediction"] == gold for x in (left, right))
                paired_answers["wrong_to_correct"] += after and not before
                paired_answers["correct_to_wrong"] += before and not after
                a, b = (row["logits"][g]-row["logits"][1-g] for row in (left, right))
                margins.append(b-a)
        for method, n in counts.items():
            total[method] += n
            if n == 4:
                robust[method].append(pair)
    assert score_count == 56 and repeats == len(margins) == 28
    gate = dict(scores=repeats, max_abs=delta_max, failures=[])
    assert gate == load("exports/replay_gate.json")
    a, b = (set(robust[m]) for m in METHODS)
    summary = dict(pairs=7, scores=score_count,
                   methods={m: dict(robust_pairs=robust[m], correct_branch_orders=total[m]) for m in METHODS},
                   recoveries=sorted(b-a), regressions=sorted(a-b), median_gold_margin_delta=median(margins),
                   positive_gold_margin_deltas=sum(d > 0 for d in margins))
    assert summary == load("exports/summary.json")
    return dict(archive_sha256=digest(archive.read_bytes()), notebook_sha256=digest(notebook.read_bytes()),
                verified_payloads=len(saved), verified_input_frames=verified_frames, replay_gate=gate,
                gpu=config["gpu"], prior_gpu=config["prior_gpu"], branch_order_changes=dict(paired_answers), **summary)


if __name__ == "__main__":
    archive, notebook = map(Path, sys.argv[1:]) if len(sys.argv) == 3 else (ARCHIVE, NOTEBOOK)
    report = json.dumps(audit(archive, notebook), indent=2) + "\n"
    if len(sys.argv) == 1:
        (ROOT / "evidence/audits/droid_state_swap_2026-11-27_audit.json").write_text(
            report, encoding="utf-8", newline="\n")
    print(report, end="")
