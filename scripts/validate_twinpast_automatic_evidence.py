"""Recompute the automatic retrieval screen from its archived frame and score traces."""
import ast
import json
import math
import sys
from collections import Counter
from pathlib import Path

from validate_twinpast_diagnostic_evidence import LETTERS, ORDERS, ROOT, digest, verify_inventory
from validate_twinpast_fixed_recent_evidence import ARCHIVE as PRIOR

ARCHIVE = ROOT / "evidence/archives/HERMES_twinpast_automatic_evidence.zip"
NOTEBOOK = ROOT / "notebooks/executed/2026-11-23_hermes-twinpast-automatic.ipynb"
TEMPLATE = ROOT / "notebooks/autodl/hermes-twinpast-automatic.ipynb"
METHODS = ("recent", "latest", "uniform", "relevance", "margin_gate", "history_control_gate")


def spread(count, size):
    return [i * (count - 1) // (size - 1) for i in range(size)]


def audit(archive, notebook):
    payload = verify_inventory(archive)
    prior_hash = digest(PRIOR.read_bytes())
    previous = verify_inventory(PRIOR)
    old_spec = json.loads(previous["provenance/fixed_recent_cases.json"])
    old_cases = {r["annotation"]["pair_id"]: r for r in old_spec["cases"]}
    bundle = json.loads(previous["provenance/bundle.json"])
    old_media = {r["pair_id"]: r["decoded"] for r in bundle["media"] if "decoded" in r}
    spec = json.loads(payload["provenance/automatic_cases.json"])
    cases = {r["pair_id"]: r for r in spec["cases"]}
    targets = json.loads(payload["provenance/evaluation_targets.json"])
    assert len(spec["cases"]) == len(cases) == 7 and cases.keys() == targets.keys() == old_cases.keys()
    assert spec["fixed_recent_archive_sha256"] == prior_hash
    for pair_id, case in cases.items():
        old = old_cases[pair_id]
        annotation = old["annotation"]
        assert set(case) == {"pair_id", "query_s", "question", "options", "recent_hashes"}
        assert case["query_s"] == annotation["t_query"] and all(case[k] == annotation[k] for k in ("question", "options"))
        assert case["recent_hashes"] == old_media[pair_id]["a"]["decoded_rgb_sha256"][-4:]
        assert targets[pair_id] == {"cohort": old["cohort"], **{k: annotation[k] for k in (
            "answer_a", "answer_b", "evidence_intervals")}}

    nb = json.loads(notebook.read_bytes())
    template = json.loads(TEMPLATE.read_bytes())
    assert [(c["cell_type"], "".join(c["source"]).strip()) for c in nb["cells"]] == [
        (c["cell_type"], "".join(c["source"]).strip()) for c in template["cells"]]
    for i, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
        assert cell["execution_count"] == i + 1 and all(o["output_type"] != "error" for o in cell["outputs"])
    source, = [c for c in nb["cells"] if "".join(c["source"]).startswith("SOURCES = ")]
    embedded = ast.literal_eval(ast.parse("".join(source["source"])).body[0].value)
    launch = json.loads(payload["launch.json"])
    assert embedded.keys() == launch["embedded_sha256"].keys()
    assert all(text.encode("utf-8") == payload[name] and launch["embedded_sha256"][name] == digest(payload[name])
               for name, text in embedded.items())
    assert "TWINPAST_AUTOMATIC_COMPLETE" in payload["logs/twinpast_automatic.log"].decode()
    assert json.loads(payload["completion.json"]) == dict(status="complete", pairs=7)
    config = json.loads(payload["exports/run_config.json"])
    assert config["model_revision"] == launch["model_revision"] == old_spec["prior_config"]["model_revision"]
    assert config["source_archive_sha256"] == prior_hash
    assert (config["max_candidates"], config["max_history"], config["min_lift"]) == (13, 6, 0.5)
    assert all(digest(payload[f"provenance/{n}"]) == value for n, value in config["hashes"].items())
    identity = digest(json.dumps(config, sort_keys=True).encode())
    counts = {m: dict(robust_pairs=[], recoveries_vs_recent=0, regressions_vs_recent=0) for m in METHODS}
    correct_counts, overrides, calls, times = Counter(), Counter(), Counter(), Counter()
    vs_direct, coverage = Counter(), []
    curated_overlap = []
    for pair_id, case in cases.items():
        checkpoint = json.loads(payload[f"checkpoints/{pair_id}.json"])
        assert checkpoint["identity"] == identity and checkpoint["block"]["pair_id"] == pair_id
        branches = checkpoint["block"]["branches"]
        assert set(branches) == {"a", "b"}
        outcomes = {m: [] for m in METHODS}
        gold = targets[pair_id]
        previous_block = json.loads(previous[f"checkpoints/{pair_id}.json"])["block"]
        for branch, trace in branches.items():
            sample = trace["sample"]
            grid = list(range(0, math.ceil(case["query_s"]), 2))
            older = grid[:-4]
            pool = [older[i] for i in spread(len(older), min(13, len(older)))]
            assert sample["grid_s"] == pool + grid[-4:]
            assert len(sample["timestamps_s"]) == len(sample["indices"]) == len(sample["hashes"]) == len(pool) + 4
            assert sample["hashes"][-4:] == case["recent_hashes"]
            assert sample["indices"] == sorted(set(sample["indices"]))
            assert all(0 <= t <= s < case["query_s"] for t, s in zip(sample["timestamps_s"], sample["grid_s"], strict=True))
            n = len(pool)
            size = min(n, 6)
            assert trace["history_budget"] == size and len(trace["candidates"]) == n - 1
            relevance = [float("-inf")] * n
            for i, probe in enumerate(trace["candidates"]):
                assert probe["slots"] == [i, i + 1] and set(probe["yes_no"]) == set(ORDERS)
                margin = sum(x[0] - x[1] for x in probe["yes_no"].values()) / 2
                assert probe["relevance"] == margin and math.isfinite(margin)
                relevance[i] = max(relevance[i], margin)
                relevance[i + 1] = max(relevance[i + 1], margin)
            ranked = sorted(sorted(range(n), key=lambda i: (-relevance[i], i))[:size])
            uniform = spread(n, size)
            recent = list(range(n, n + 4))
            selection = dict(recent=recent, latest=list(range(n - size, n)) + recent,
                             uniform=uniform + recent, relevance=ranked + recent)
            assert trace["selection"] == selection
            for call in trace["calls"]:
                phase = call["phase"]
                assert phase in {"retrieval", *selection} and 0 <= call["forward_s"]
                calls[phase] += 1
                times[phase] += call["forward_s"]
                assert call["slots"] == selection[phase] if phase != "retrieval" else len(call["slots"]) == 2
            assert sum(c["phase"] == "retrieval" for c in trace["calls"]) == 2 * (n - 1)
            answer = gold[f"answer_{branch}"]
            for order in ORDERS:
                item = trace["answers"][order]
                scores = item["logits"]
                assert set(scores) == selection.keys()
                assert all(len(v) == len(case["options"]) and all(math.isfinite(x) for x in v) for v in scores.values())
                pred = {k: max(range(len(v)), key=v.__getitem__) for k, v in scores.items()}
                initial, alternative = pred["recent"], pred["relevance"]
                delta = {k: v[alternative] - v[initial] for k, v in scores.items()}
                lift, specificity = delta["relevance"] - delta["recent"], delta["relevance"] - delta["uniform"]
                assert item["margins"] == dict(history_margin=delta["relevance"], lift_from_recent=lift,
                                               lift_from_uniform=specificity)
                pred["margin_gate"] = alternative if alternative != initial and lift >= 0.5 else initial
                pred["history_control_gate"] = alternative if (
                    pred["margin_gate"] == alternative and ranked != uniform and specificity >= 0.5) else initial
                assert item["predictions"] == {k: LETTERS[v] for k, v in pred.items()}
                for method in METHODS:
                    correct = LETTERS[pred[method]] == answer
                    baseline = LETTERS[initial] == answer
                    outcomes[method].append(correct)
                    correct_counts[method] += correct
                    counts[method]["recoveries_vs_recent"] += correct and not baseline
                    counts[method]["regressions_vs_recent"] += baseline and not correct
                for method in ("margin_gate", "history_control_gate"):
                    overrides[method] += pred[method] != initial
                controlled = LETTERS[pred["history_control_gate"]] == answer
                direct = LETTERS[alternative] == answer
                vs_direct["recoveries"] += controlled and not direct
                vs_direct["regressions"] += direct and not controlled
            intervals = gold["evidence_intervals"][branch.upper()]
            inside = [i for i, t in enumerate(sample["timestamps_s"][:-4]) if any(lo <= t <= hi for lo, hi in intervals)]
            coverage.append(dict(pair_id=pair_id, branch=branch, candidate_slots_in_evidence=inside,
                                 selected_slots_in_evidence={k: sorted(set(inside) & set(selection[k]))
                                                             for k in ("latest", "uniform", "relevance")}))
            if pair_id in ("twinpast_0094", "twinpast_0259"):
                old = next(r for r in previous_block["rows"]["history_only"] if r["branch"] == branch)
                hashes = set(old["history_frame_hashes"])
                curated_overlap.append(dict(pair_id=pair_id, branch=branch, curated_slots=len(hashes),
                    in_candidates=len(hashes & set(sample["hashes"][:-4])),
                    in_relevance=len(hashes & {sample["hashes"][i] for i in ranked})))
        for method, correct in outcomes.items():
            assert len(correct) == 4
            if all(correct):
                counts[method]["robust_pairs"].append(pair_id)
    assert coverage == json.loads(payload["exports/postrun_evidence_overlap.json"])
    summary = dict(pairs=7, branch_order_answers=28, methods=counts, overrides=dict(overrides),
                   controlled_gate_vs_same_frames_direct=dict(vs_direct), total_forward_calls=sum(calls.values()))
    assert summary == json.loads(payload["exports/summary.json"])
    return dict(archive_sha256=digest(archive.read_bytes()), notebook_sha256=digest(notebook.read_bytes()),
                prior_archive_sha256=prior_hash, verified_payloads=len(payload), branch_order_correct=correct_counts,
                forward_calls_by_phase=calls, forward_seconds_by_phase=times,
                recovered_state_curated_frame_overlap=curated_overlap, **summary)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        report = audit(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        report = audit(ARCHIVE, NOTEBOOK)
        output = ROOT / "evidence/audits/twinpast_automatic_2026-11-23_audit.json"
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Verified {report['verified_payloads']} payloads and {report['total_forward_calls']} forward-call records")
