"""Audit TwinPast refinement provenance, frame selections and paired outcomes."""
import ast
from collections import Counter
from itertools import combinations
import json
import math
import sys
from pathlib import Path

from validate_twinpast_diagnostic_evidence import LETTERS, ORDERS, ROOT, digest, verify_inventory

ARCHIVE = ROOT / "evidence/archives/HERMES_twinpast_refinement_evidence.zip"
NOTEBOOK = ROOT / "notebooks/executed/2026-11-27_hermes-twinpast-refinement.ipynb"
MANIFEST = ROOT / "data/twinpast/refinement_development.json"
METHODS = ("coarse_uniform", "local_uniform", "state_change")


def audit(archive, notebook):
    saved = verify_inventory(archive)

    def load(name):
        return json.loads(saved[name])

    spec = load("provenance/refinement_cases.json")
    manifest = json.loads(MANIFEST.read_bytes())
    assert spec["manifest_sha256"] == digest(MANIFEST.read_bytes())
    annotations = {row["pair_id"]: row for row in manifest["candidates"]}
    assert set(annotations).isdisjoint(manifest["excluded_ids"])
    targets = load("provenance/evaluation_targets.json")
    assert targets == {pair: {f"answer_{b}": row[f"answer_{b}"] for b in "ab"}
                       for pair, row in annotations.items()}

    launch, config = load("launch.json"), load("exports/run_config.json")
    assert launch["model_revision"] == config["model_revision"] == spec["model_revision"]
    cells = json.loads(notebook.read_bytes())["cells"]
    cell, = (c for c in cells if "".join(c["source"]).startswith("SOURCES = "))
    embedded = ast.literal_eval(ast.parse("".join(cell["source"])).body[0].value)
    assert launch["embedded_sha256"] == {name: digest(src.encode()) for name, src in embedded.items()}
    assert all(saved[name] == src.encode() for name, src in embedded.items())
    assert config["hashes"] == {name.split("/", 1)[1]: digest(raw) for name, raw in saved.items()
                                if name.startswith("provenance/")}
    assert not any(o["output_type"] == "error" for c in cells for o in c.get("outputs", []))
    assert b"PINNED_ENV_PASS" in saved["logs/environment.log"]
    assert b"TWINPAST_REFINEMENT_COMPLETE" in saved["logs/twinpast_refinement.log"]
    identity = digest(json.dumps(config, sort_keys=True).encode())
    pairs = config["accepted"]
    assert len(pairs) == len(set(pairs)) == 8
    assert set(pairs) <= annotations.keys()
    screen = load("exports/source_screen.json")
    screened = {r["pair_id"]: r["recent_hashes"] for r in screen["decisions"] if r["disposition"] == "accepted"}
    assert list(screened) == pairs
    assert {Path(n).stem for n in saved if n.startswith("checkpoints/")} == set(pairs)
    assert load("completion.json") == dict(status="complete", pairs=8)

    methods = {m: dict(robust_pairs=[], correct_branch_orders=0) for m in METHODS}
    calls, coverage, covered_correct = Counter(), Counter(), Counter()
    for pair in pairs:
        checkpoint = load(f"checkpoints/{pair}.json")
        block = checkpoint["block"]
        assert checkpoint["identity"] == identity and block["pair_id"] == pair
        assert set(block["branches"]) == {"a", "b"}
        correct = Counter()
        for side, branch in block["branches"].items():
            row = annotations[pair]
            coarse, fine = branch["coarse"], branch["refined"]
            assert coarse["hashes"][-4:] == screened[pair]
            windows = branch["selected_windows"]
            candidates = sorted(branch["candidates"], key=lambda candidate: candidate["slots"])
            assert len(candidates) == 12
            for candidate in candidates:
                assert candidate["relevance"] == sum(v[0]-v[1] for v in candidate["yes_no"].values()) / 2
            compatible = (group for group in combinations(candidates, 3)
                          if all(left["slots"][-1] < right["slots"][0] for left, right in zip(group, group[1:])))
            ranked = max(compatible, key=lambda group: sum(candidate["relevance"] for candidate in group))
            assert windows == [candidate["slots"] for candidate in ranked]
            assert fine["grid_s"] == [start + i/4 for start, end in windows for i in range(4*(end-start)+1)]
            assert len(fine["grid_s"]) == branch["inspected_refinement_frames"] == 39
            assert max(fine["timestamps_s"]) < coarse["timestamps_s"][-4] < row["t_query"]
            uniform, change = [], []
            for i, trace in enumerate(branch["changes"]):
                assert trace["window_s"] == windows[i]
                values = trace["adjacent_rgb_mae"]
                assert len(values) == 12 and all(math.isfinite(v) and v >= 0 for v in values)
                peak = max(range(12), key=values.__getitem__)
                uniform.extend([13*i+3, 13*i+9])
                change.extend([13*i+peak, 13*i+peak+1])
            assert branch["refined_picks"] == dict(local_uniform=uniform, state_change=change)
            slots = branch["coarse_uniform_slots"]
            assert all(len(set(s)) == 6 for s in (slots, uniform, change))
            assert all(0 <= i < len(coarse["timestamps_s"])-4 for i in slots)
            intervals = (row["evidence_intervals"] or {}).get(side.upper())
            if intervals:
                coverage["annotated_branches"] += 1
                samples = dict(coarse_uniform=[coarse["timestamps_s"][i] for i in slots],
                               local_uniform=[fine["timestamps_s"][i] for i in uniform],
                               state_change=[fine["timestamps_s"][i] for i in change],
                               fine_pool=fine["timestamps_s"], coarse_pool=coarse["timestamps_s"][:-4])
                for method, ticks in samples.items():
                    hit = any(lo-1e-6 <= t <= hi+1e-6 for t in ticks for lo, hi in intervals)
                    coverage[method] += hit
                    if method in METHODS:
                        covered_correct[method] += hit and all(
                            branch["answers"][order][method]["prediction"] == targets[pair][f"answer_{side}"]
                            for order in ORDERS
                        )
            assert set(branch["answers"]) == set(ORDERS)
            assert Counter(c["phase"] for c in branch["calls"]) == dict(shared_retrieval=24, **dict.fromkeys(METHODS, 2))
            calls.update(c["phase"] for c in branch["calls"])
            for order in ORDERS:
                answer_calls = [c for c in branch["calls"] if c["phase"] in METHODS and c["order"] == order]
                assert {c["phase"] for c in answer_calls} == set(METHODS)
                for field in ("lm_tokens", "visual_tokens", "prompt_sha256"):
                    assert len({c[field] for c in answer_calls}) == 1
                for method in METHODS:
                    answer = branch["answers"][order][method]
                    logits = answer["logits"]
                    assert len(logits) == len(row["options"]) and all(math.isfinite(x) for x in logits)
                    assert answer["prediction"] == LETTERS[max(range(len(logits)), key=logits.__getitem__)]
                    correct[method] += answer["prediction"] == targets[pair][f"answer_{side}"]
        for method in METHODS:
            methods[method]["correct_branch_orders"] += correct[method]
            if correct[method] == 4:
                methods[method]["robust_pairs"].append(pair)
    a, b = (set(methods[m]["robust_pairs"]) for m in ("state_change", "local_uniform"))
    summary = dict(pairs=8, methods=methods, state_change_only=sorted(a-b), local_uniform_only=sorted(b-a),
                   state_change_net_robust_gain=len(a)-len(b), forward_calls_by_phase=dict(calls))
    assert summary == load("exports/summary.json")
    return dict(archive_sha256=digest(archive.read_bytes()), notebook_sha256=digest(notebook.read_bytes()),
                verified_payloads=len(saved), scores=sum(calls[m] for m in METHODS), gpu=config["gpu"],
                timestamp_overlap=dict(coverage), covered_branches_correct_both_orders=dict(covered_correct),
                model_download_log_archived="logs/model_download.log" in saved,
                **summary)


if __name__ == "__main__":
    archive, notebook = map(Path, sys.argv[1:]) if len(sys.argv) == 3 else (ARCHIVE, NOTEBOOK)
    result = audit(archive, notebook)
    print(json.dumps(result, indent=2))
    if len(sys.argv) == 1:
        out = ROOT / "evidence/audits/twinpast_refinement_2026-11-27_audit.json"
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
