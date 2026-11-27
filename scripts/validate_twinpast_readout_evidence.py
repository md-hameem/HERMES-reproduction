"""Verify the fixed-frame readout archive independently of the GPU runner."""
import ast
from collections import Counter
import json
import math
import sys
from pathlib import Path

from make_twinpast_readout_notebook import archive_cases
from validate_twinpast_diagnostic_evidence import LETTERS, ORDERS, ROOT, digest, verify_inventory
from validate_twinpast_refinement_evidence import ARCHIVE as REFINEMENT

ARCHIVE = ROOT / "evidence/archives/HERMES_twinpast_readout_evidence.zip"
NOTEBOOK = ROOT / "notebooks/executed/2026-11-27_hermes-twinpast-fixed-frame-readout.ipynb"
TEMPLATE = ROOT / "notebooks/autodl/hermes-twinpast-fixed-frame-readout.ipynb"
FRAME_SETS = ("coarse_uniform", "state_change")
METHODS = ("direct", "observation")


def audit(archive, notebook):
    saved, prior = verify_inventory(archive), verify_inventory(REFINEMENT)
    load = lambda name: json.loads(saved[name])
    cells = json.loads(notebook.read_bytes())["cells"]
    expected = json.loads(TEMPLATE.read_bytes())["cells"]
    assert [c["source"] for c in cells] == [c["source"] for c in expected]
    for execution, cell in enumerate((c for c in cells if c["cell_type"] == "code"), start=1):
        assert cell["execution_count"] == execution
        assert not any(o["output_type"] == "error" for o in cell["outputs"])
    source, = ("".join(c["source"]) for c in cells if "".join(c["source"]).startswith("SOURCES = "))
    embedded = ast.literal_eval(ast.parse(source).body[0].value)
    assert all(raw.encode() == saved[name] for name, raw in embedded.items())
    launch, config = load("launch.json"), load("exports/run_config.json")
    assert launch["embedded_sha256"] == {name: digest(saved[name]) for name in embedded}
    spec = load("provenance/readout_cases.json")
    assert spec == archive_cases(prior)
    assert config["source_archive_sha256"] == spec["source_archive_sha256"]
    assert config["model_revision"] == spec["model_revision"] == launch["model_revision"]
    assert b"PINNED_ENV_PASS" in saved["logs/environment.log"]
    run_log = saved["logs/twinpast_readout.log"]
    assert run_log.index(b"REPLAY_PASS") < run_log.index(b"READOUT_PAIR") < run_log.index(b"TWINPAST_READOUT_COMPLETE")
    assert load("completion.json") == dict(status="complete", pairs=8)
    assert load("exports/replay_gate.json") == dict(scores=64, max_abs=0.0, failures=[])
    assert config["replay_atol"] == 0.05 and config["observation_max_new_tokens"] == 24

    cases = {case["pair_id"]: case for case in spec["cases"]}
    prior_targets = json.loads(prior["provenance/evaluation_targets.json"])
    targets = load("provenance/evaluation_targets.json")
    assert targets == {p: prior_targets[p] for p in cases}
    assert set(cases) == {Path(n).stem for n in saved if n.startswith("checkpoints/")}
    summaries, diagnostics, scored = {}, {}, 0
    for frame_set in FRAME_SETS:
        counters = {m: Counter() for m in METHODS}
        robust = {m: [] for m in METHODS}
        diagnostics[frame_set] = Counter()
        for pair, case in cases.items():
            checkpoint = load(f"checkpoints/{pair}.json")
            assert checkpoint["pair_id"] == pair
            assert len(checkpoint["rows"]) == 4
            assert {(r["frame_set"], r["branch"]) for r in checkpoint["rows"]} == {
                (f, b) for f in FRAME_SETS for b in "ab"
            }
            correct, notes = Counter(), []
            for row in (r for r in checkpoint["rows"] if r["frame_set"] == frame_set):
                side, note = row["branch"], row["observation"]
                gold = targets[pair][f"answer_{side}"]
                notes.append(note["text"])
                count = len(note["generated_token_ids"])
                assert count <= config["observation_max_new_tokens"]
                assert note["forward_calls"] == min(count+1, config["observation_max_new_tokens"])
                assert note["input_tokens"] > 0 and math.isfinite(note["forward_s"]) and note["forward_s"] >= 0
                for field in ("forward_calls", "input_tokens", "forward_s"):
                    counters["observation"][field] += note[field]
                assert set(row["orders"]) == set(ORDERS)
                for order, result in row["orders"].items():
                    assert set(result) == set(METHODS)
                    direct = result["direct"]
                    archived = case["branches"][side]["direct"][frame_set][order]
                    assert direct["replay_max_abs"] == 0 and all(direct[k] == v for k, v in archived.items())
                    assert result["observation"]["visual_tokens"] == direct["visual_tokens"]
                    before, after = (result[m]["prediction"] == gold for m in METHODS)
                    diagnostics[frame_set]["wrong_to_correct"] += after and not before
                    diagnostics[frame_set]["correct_to_wrong"] += before and not after
                    for method, choice in result.items():
                        scores = choice["logits"]
                        assert len(scores) == len(case["options"]) and all(math.isfinite(x) for x in scores)
                        assert choice["prediction"] == LETTERS[max(range(len(scores)), key=scores.__getitem__)]
                        assert choice["lm_tokens"] > 0 and math.isfinite(choice["forward_s"]) and choice["forward_s"] >= 0
                        correct[method] += choice["prediction"] == gold
                        counters[method].update(forward_calls=1, input_tokens=choice["lm_tokens"], forward_s=choice["forward_s"])
                        scored += 1
            diagnostics[frame_set]["identical_notes_on_both_branches"] += notes[0] == notes[1]
            for method in METHODS:
                counters[method]["correct_branch_orders"] += correct[method]
                if correct[method] == 4:
                    robust[method].append(pair)
        summaries[frame_set] = dict(methods={m: dict(robust_pairs=robust[m], **counters[m]) for m in METHODS},
                                    recoveries=sorted(set(robust["observation"])-set(robust["direct"])),
                                    regressions=sorted(set(robust["direct"])-set(robust["observation"])),
                                    max_replay_abs=0.0)
    assert scored == 128
    primary = summaries["coarse_uniform"]
    summary = dict(pairs=8, choice_scores=scored, frame_sets=summaries,
                   development_criterion_met=bool(primary["recoveries"]) and not primary["regressions"])
    assert summary == load("exports/summary.json")
    return dict(archive_sha256=digest(archive.read_bytes()), notebook_sha256=digest(notebook.read_bytes()),
                verified_payloads=len(saved), replay_gate=load("exports/replay_gate.json"),
                diagnostics={f: dict(c) for f, c in diagnostics.items()}, **summary)


if __name__ == "__main__":
    archive, notebook = map(Path, sys.argv[1:]) if len(sys.argv) == 3 else (ARCHIVE, NOTEBOOK)
    report = audit(archive, notebook)
    text = json.dumps(report, indent=2) + "\n"
    if len(sys.argv) == 1:
        out = ROOT / "evidence/audits/twinpast_readout_2026-11-27_audit.json"
        out.write_text(text, encoding="utf-8", newline="\n")
    print(text, end="")
