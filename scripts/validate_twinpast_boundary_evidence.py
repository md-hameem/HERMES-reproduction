"""Audit the GPU scene-boundary comparison against the frozen selection and replay."""
import ast
from collections import Counter
import json
import math
import sys
from pathlib import Path
from statistics import median

from make_twinpast_boundary_notebook import CPU_AUDIT, DEST as TEMPLATE, freeze
from make_twinpast_readout_notebook import archive_cases
from validate_twinpast_diagnostic_evidence import LETTERS, ORDERS, ROOT, digest, verify_inventory
from validate_twinpast_refinement_evidence import ARCHIVE as REFINEMENT

ARCHIVE = ROOT / "evidence/archives/HERMES_twinpast_boundary_evidence.zip"
NOTEBOOK = ROOT / "notebooks/executed/2026-11-27_hermes-twinpast-boundary.ipynb"
METHODS = ("coarse_uniform", "local_uniform", "boundary")


def audit(archive, notebook):
    saved, prior = verify_inventory(archive), verify_inventory(REFINEMENT)

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
    launch = load("launch.json")
    assert launch["embedded_sha256"] == {name: digest(saved[name]) for name in embedded}
    config = load("exports/run_config.json")
    cpu = json.loads(CPU_AUDIT.read_bytes())
    spec = load("provenance/boundary_cases.json")
    assert spec == freeze(archive_cases(prior), prior, cpu)
    assert config["model_revision"] == spec["model_revision"] == launch["model_revision"]
    assert config["cpu_audit_sha256"] == spec["cpu_audit_sha256"]
    assert config["refinement_archive_sha256"] == spec["source_archive_sha256"]
    assert config["replay_atol"] == 0.05
    assert b"PINNED_ENV_PASS" in saved["logs/environment.log"]
    assert b"TWINPAST_BOUNDARY_COMPLETE" in saved["logs/twinpast_boundary.log"]
    assert load("completion.json") == dict(status="complete", pairs=8)

    cases = {case["pair_id"]: case for case in spec["cases"]}
    prior_targets = json.loads(prior["provenance/evaluation_targets.json"])
    targets = load("provenance/evaluation_targets.json")
    assert targets == {p: prior_targets[p] for p in cases}
    assert set(cases) == {Path(name).stem for name in saved if name.startswith("checkpoints/")}
    robust = {m: [] for m in METHODS}
    durations = {m: [] for m in METHODS}
    totals, changes = Counter(), {m: Counter() for m in METHODS[:2]}
    scores, replays, scanned, answer_max, retrieval_max = 0, 0, 0, 0.0, 0.0
    for pair, case in cases.items():
        block = load(f"checkpoints/{pair}.json")
        assert block["pair_id"] == pair and set(block["branches"]) == {"a", "b"}
        correct = Counter()
        for side, trace in block["branches"].items():
            branch = case["branches"][side]
            boundary = trace["boundary"]
            assert all(boundary[k] == value for k, value in branch["boundary"].items())
            assert len(boundary["hashes"]) == len(boundary["timestamps_s"]) == len(boundary["indices"]) == 6
            assert boundary["indices"] == sorted(set(boundary["indices"]))
            assert boundary["indices"][-1] < branch["coarse"]["indices"][-4]
            assert not trace["replay_failures"]
            scanned += boundary["scanned_indices"]
            retrieval_max = max(retrieval_max, trace["retrieval_max_abs"])
            assert set(trace["answers"]) == set(ORDERS)
            stages = trace["stages_s"]
            assert all(math.isfinite(v) and v >= 0 for v in stages.values())
            timing = Counter()
            for order, group in trace["answers"].items():
                assert set(group) == set(METHODS)
                for key in ("prompt_sha256", "visual_tokens", "lm_tokens"):
                    assert len({row[key] for row in group.values()}) == 1
                gold = targets[pair][f"answer_{side}"]
                for method, row in group.items():
                    values = row["logits"]
                    assert len(values) == len(case["options"]) and all(math.isfinite(v) for v in values)
                    assert row["prediction"] == LETTERS[max(range(len(values)), key=values.__getitem__)]
                    correct[method] += row["prediction"] == gold
                    assert math.isfinite(row["synchronized_answer_s"]) and row["synchronized_answer_s"] >= 0
                    timing[method] += row["synchronized_answer_s"]
                    scores += 1
                    if method != "boundary":
                        expected = branch["direct"][method][order]
                        assert all(row[k] == v for k, v in expected.items())
                        assert row["replay_max_abs"] == 0
                        answer_max = max(answer_max, row["replay_max_abs"])
                        replays += 1
                        before, after = row["prediction"] == gold, group["boundary"]["prediction"] == gold
                        changes[method]["wrong_to_correct"] += after and not before
                        changes[method]["correct_to_wrong"] += before and not after
            for method in METHODS:
                total = stages["coarse_decode_encode"] + stages[f"{method}_history"]
                if method != "coarse_uniform":
                    total += stages["question_relevance"]
                if method == "boundary":
                    total += stages["boundary_scan_and_select"]
                durations[method].append(total + timing[method]/len(ORDERS))
        for method in METHODS:
            totals[method] += correct[method]
            if correct[method] == 4:
                robust[method].append(pair)
    assert scores == 96 and replays == 64 and retrieval_max == answer_max == 0
    assert scanned == cpu["summary"]["decoded_probe_frames"]
    gate = dict(scores=replays, max_abs=answer_max, retrieval_max_abs=retrieval_max, failures=[])
    assert gate == load("exports/replay_gate.json")
    paired = {m: dict(recoveries=sorted(set(robust["boundary"])-set(robust[m])),
                      regressions=sorted(set(robust[m])-set(robust["boundary"]))) for m in METHODS[:2]}
    primary = paired["coarse_uniform"]
    summary = dict(pairs=8, scores=scores, boundary_against=paired,
                   development_criterion_met=bool(primary["recoveries"]) and not primary["regressions"],
                   methods={m: dict(robust_pairs=robust[m], correct_branch_orders=totals[m],
                                    median_stage_sum_per_branch_s=median(durations[m])) for m in METHODS})
    assert summary == load("exports/summary.json")
    return dict(archive_sha256=digest(archive.read_bytes()), notebook_sha256=digest(notebook.read_bytes()),
                verified_payloads=len(saved), decoded_probe_frames=scanned, replay_gate=gate,
                branch_order_changes_against={m: dict(counts) for m, counts in changes.items()}, **summary)


if __name__ == "__main__":
    archive, notebook = map(Path, sys.argv[1:]) if len(sys.argv) == 3 else (ARCHIVE, NOTEBOOK)
    result = audit(archive, notebook)
    text = json.dumps(result, indent=2) + "\n"
    if len(sys.argv) == 1:
        (ROOT / "evidence/audits/twinpast_boundary_gpu_2026-11-27.json").write_text(text, encoding="utf-8", newline="\n")
    print(text, end="")
