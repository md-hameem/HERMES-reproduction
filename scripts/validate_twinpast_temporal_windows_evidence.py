"""Audit the fresh-pair temporal-window screen and its sampling coverage."""
import ast
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path
import sys

from validate_twinpast_automatic_evidence import ARCHIVE as PRIOR
from validate_twinpast_diagnostic_evidence import LETTERS, ORDERS, ROOT, digest, verify_inventory

ARCHIVE = ROOT / "evidence/archives/HERMES_twinpast_temporal_windows_evidence.zip"
NOTEBOOK = ROOT / "notebooks/executed/2026-11-23_hermes-twinpast-temporal-windows.ipynb"
TEMPLATE = ROOT / "notebooks/autodl/hermes-twinpast-temporal-windows.ipynb"
MANIFEST = ROOT / "data/twinpast/temporal_window_development.json"
METHODS = ("recent", "uniform", "individual", "endpoints")


def spread(n, k):
    return list(range(n)) if n <= k else [i * (n - 1) // (k - 1) for i in range(k)]


def audit(archive, notebook):
    saved = verify_inventory(archive)
    spec = json.loads(saved["provenance/temporal_cases.json"])
    targets = json.loads(saved["provenance/evaluation_targets.json"])
    manifest = json.loads(MANIFEST.read_bytes())
    assert spec["manifest_sha256"] == digest(MANIFEST.read_bytes())
    assert spec["automatic_archive_sha256"] == digest(PRIOR.read_bytes())
    assert spec["excluded"] == manifest["excluded_development_ids"] and len(spec["excluded"]) == 12
    assert spec["media_plan"] == manifest["media_plan"]
    assert spec["media_source"]["manifest"] == {k: manifest[k] for k in ("dataset", "revision")}
    cases = {r["pair_id"]: r for r in spec["cases"]}
    annotations = {r["pair_id"]: r for r in manifest["candidates"]}
    assert len(cases) == 12 and cases.keys() == targets.keys() == annotations.keys()
    assert cases.keys().isdisjoint(spec["excluded"])
    for pair, a in annotations.items():
        assert cases[pair] == dict(pair_id=pair, query_s=a["t_query"], question=a["question"], options=a["options"])
        assert targets[pair] == {k: a[k] for k in ("answer_a", "answer_b")}

    nb, template = [json.loads(p.read_bytes()) for p in (notebook, TEMPLATE)]
    def signature(cell):
        return cell["cell_type"], "".join(cell["source"]).strip()

    added = ("code", "!source /etc/network_turbo")
    assert sum(signature(c) == added for c in nb["cells"]) == 1
    cells = [c for c in nb["cells"] if signature(c) != added]
    assert [signature(c) for c in cells] == [signature(c) for c in template["cells"]]
    assert all(c["execution_count"] and all(o["output_type"] != "error" for o in c["outputs"])
               for c in nb["cells"] if c["cell_type"] == "code")
    source, = [c for c in cells if "".join(c["source"]).startswith("SOURCES = ")]
    embedded = ast.literal_eval(ast.parse("".join(source["source"])).body[0].value)
    launch = json.loads(saved["launch.json"])
    assert embedded.keys() == launch["embedded_sha256"].keys()
    assert all(s.encode() == saved[n] and launch["embedded_sha256"][n] == digest(saved[n]) for n, s in embedded.items())
    assert "TWINPAST_TEMPORAL_WINDOWS_COMPLETE" in saved["logs/twinpast_temporal_windows.log"].decode()
    config = json.loads(saved["exports/run_config.json"])
    assert config["model_revision"] == launch["model_revision"] == spec["model_revision"]
    assert (config["window_span"], config["window_cap"]) == (3, 12)
    assert all(digest(saved[f"provenance/{n}"]) == value for n, value in config["hashes"].items())
    identity = digest(json.dumps(config, sort_keys=True).encode())
    screen = json.loads(saved["exports/source_screen.json"])
    accepted = {r["pair_id"]: r["recent_hashes"] for r in screen["decisions"] if r["disposition"] == "accepted"}
    assert len(accepted) == 8 and list(accepted) == config["accepted"] and accepted.keys() <= cases.keys()
    assert json.loads(saved["completion.json"]) == dict(status="complete", pairs=len(accepted))
    assert {Path(n).stem for n in saved if n.startswith("checkpoints/")} == accepted.keys()

    stats = {m: dict(robust_pairs=[], correct_branch_orders=0) for m in METHODS}
    by_order = {m: {o: Counter() for o in ORDERS} for m in METHODS}
    calls, times, coverage = Counter(), Counter(), Counter()
    matched_selections = 0
    for pair, recent_hashes in accepted.items():
        block = json.loads(saved[f"checkpoints/{pair}.json"])
        assert block["identity"] == identity and block["block"]["pair_id"] == pair
        branches = block["block"]["branches"]
        assert set(branches) == {"a", "b"}
        for side, trace in branches.items():
            sample = trace["sample"]
            grid = list(range(math.ceil(cases[pair]["query_s"])))
            n = len(grid) - 4
            assert sample["grid_s"] == grid and sample["hashes"][-4:] == recent_hashes
            assert len(sample["hashes"]) == len(sample["indices"]) == len(sample["timestamps_s"]) == len(grid)
            assert sample["indices"] == sorted(set(sample["indices"]))
            assert all(0 <= t <= g < cases[pair]["query_s"] for t, g in zip(sample["timestamps_s"], grid, strict=True))
            starts = spread(n-3, 12)
            assert len(trace["windows"]) == len(starts)
            individual = {}
            for start, window in zip(starts, trace["windows"], strict=True):
                assert window["slots"] == [start, start+3] and window["yes_no"].keys() == set(ORDERS)
                value = sum(a-b for a, b in window["yes_no"].values()) / 2
                assert math.isfinite(value) and window["relevance"] == value
                for i in window["slots"]:
                    individual[i] = max(value, individual.get(i, -math.inf))
            allowed = [w for w in combinations(trace["windows"], 3)
                       if all(a["slots"][-1] < b["slots"][0] for a,b in zip(w,w[1:]))]
            endpoints = [i for w in max(allowed, key=lambda x: sum(r["relevance"] for r in x)) for i in w["slots"]]
            recent = list(range(n, n+4))
            assert trace["selection"] == dict(recent=recent, uniform=spread(n, 6)+recent,
                individual=sorted(sorted(individual, key=lambda i: (-individual[i], i))[:6])+recent,
                endpoints=endpoints+recent)
            matched_selections += trace["selection"]["individual"] == trace["selection"]["endpoints"]
            for order in ORDERS:
                answer = trace["answers"][order]
                assert answer["logits"].keys() == set(METHODS)
                assert all(len(v) == len(cases[pair]["options"]) and all(math.isfinite(s) for s in v) for v in answer["logits"].values())
                assert answer["predictions"] == {m: LETTERS[max(range(len(v)), key=v.__getitem__)]
                                                 for m,v in answer["logits"].items()}
            for call in trace["calls"]:
                phase = call["phase"]
                assert phase in {"retrieval", *METHODS} and math.isfinite(call["forward_s"]) and call["forward_s"] >= 0
                calls[phase] += 1
                times[phase] += call["forward_s"]
            intervals = annotations[pair]["evidence_intervals"]
            if intervals and intervals.get(side.upper()):
                coverage["annotated_branches"] += 1
                within = {i for i,t in enumerate(sample["timestamps_s"][:-4])
                          if any(lo-1e-6 <= t <= hi+1e-6 for lo,hi in intervals[side.upper()])}
                coverage["candidate_pool_hit"] += bool(within)
                for method in METHODS[1:]:
                    coverage[f"{method}_hit"] += bool(within & set(trace["selection"][method][:-4]))
        gold = tuple(targets[pair][f"answer_{side}"] for side in "ab")
        for method in METHODS:
            robust = True
            for order in ORDERS:
                pred = tuple(branches[b]["answers"][order]["predictions"][method] for b in "ab")
                if method == "recent":
                    assert pred[0] == pred[1]
                robust &= pred == gold
                stats[method]["correct_branch_orders"] += sum(p == g for p,g in zip(pred,gold,strict=True))
                by_order[method][order]["both_correct"] += pred == gold
                by_order[method][order]["unchanged_answer"] += pred[0] == pred[1]
            if robust:
                stats[method]["robust_pairs"].append(pair)
    chosen = set(stats["endpoints"]["robust_pairs"])
    changes = {m: dict(recoveries=sorted(chosen-set(stats[m]["robust_pairs"])),
                       regressions=sorted(set(stats[m]["robust_pairs"])-chosen)) for m in ("uniform","individual")}
    summary = dict(pairs=len(accepted), methods=stats, matched_pair_changes=changes,
                   endpoints_more_robust_than_both_dense_baselines=len(chosen) >
                   max(len(stats[m]["robust_pairs"]) for m in changes), forward_calls=sum(calls.values()), forward_calls_by_phase=dict(calls))
    assert summary == json.loads(saved["exports/summary.json"])
    return dict(archive_sha256=digest(archive.read_bytes()), notebook_sha256=digest(notebook.read_bytes()),
                verified_payloads=len(saved), source_status=screen["source_status"], per_order=by_order,
                forward_seconds_by_phase=times, identical_endpoint_and_individual_branches=matched_selections,
                annotated_frame_coverage=coverage, **summary)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        report = audit(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        report = audit(ARCHIVE, NOTEBOOK)
        destination = ROOT / "evidence/audits/twinpast_temporal_windows_2026-11-23_audit.json"
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Verified {report['verified_payloads']} archive payloads and {report['forward_calls']} forward-call records")
