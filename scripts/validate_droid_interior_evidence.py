"""Audit the completed DROID interior GPU screen against frozen CPU receipts."""
import ast
import json
from math import isfinite
from pathlib import Path
import sys

from check_droid_interior_coverage import INTAKE, OUTPUT as COVERAGE, REVIEW, check
from validate_droid_natural_history_evidence import verify_frame_pack
from validate_twinpast_diagnostic_evidence import LETTERS, ORDERS, ROOT, digest, verify_inventory

ARCHIVE = ROOT / "evidence/archives/HERMES_droid_interior_evidence.zip"
NOTEBOOK = ROOT / "notebooks/executed/2026-11-28_hermes-droid-interior.ipynb"
TEMPLATE = ROOT / "notebooks/autodl/hermes-droid-interior.ipynb"
FRAMES = ROOT / "evidence/local/droid-interior/DROID_interior_frames.zip"
OUTPUT = ROOT / "evidence/audits/droid_interior_gpu_2026-11-28_audit.json"
METHODS = ("native", "fixed", "recency", "annotated")


def audit(archive, notebook):
    saved = verify_inventory(archive)

    def load(name):
        return json.loads(saved[name])

    cells = json.loads(notebook.read_bytes())["cells"]
    expected = json.loads(TEMPLATE.read_bytes())["cells"]
    assert [(c["cell_type"], c["source"]) for c in cells] == [(c["cell_type"], c["source"]) for c in expected]
    for i, cell in enumerate((c for c in cells if c["cell_type"] == "code"), 1):
        assert cell["execution_count"] == i and all(o["output_type"] != "error" for o in cell["outputs"])
    sources = ast.literal_eval(ast.parse("".join(cells[4]["source"])).body[0].value)
    assert all(s.encode() == saved[name] for name, s in sources.items())
    launch, config = load("launch.json"), load("exports/run_config.json")
    assert launch["embedded_sha256"] == {n: digest(saved[n]) for n in sources}
    spec = load("provenance/droid_interior_cases.json")
    assert config["model_revision"] == spec["model_revision"] == launch["model_revision"]
    assert config["attention_backend"] in {"sdpa", "eager"}
    assert saved["provenance/droid_interior_runner.py"].decode() == (
        ROOT / "scripts/autodl_droid_interior.py").read_text(encoding="utf-8")
    log = saved["logs/droid_interior.log"]
    assert log.count(b"DROID_INTERIOR_CLIP ") == 7 and b"DROID_INTERIOR_COMPLETE" in log
    assert b"PINNED_ENV_PASS" in saved["logs/environment.log"]
    assert load("completion.json") == {"status": "complete", "clips": 7}
    cpu = json.loads(COVERAGE.read_bytes())
    assert cpu == json.loads(json.dumps(check(INTAKE, REVIEW))) and cpu["gpu_feasibility_criterion_met"]
    assert spec["coverage_audit_sha256"] == digest(COVERAGE.read_bytes()) and spec["review_sha256"] == cpu["review_sha256"]
    assert spec["static_window"] == max(cpu["all_static_windows"],
                                      key=lambda r: (r["covered"], -abs(r["slots"][1]-5)))["slots"]
    frames = verify_frame_pack(FRAMES, spec)
    results = load("exports/scores.json")
    assert frames == 70 and len(results) == len(cpu["cases"]) == len(spec["cases"]) == 7
    methods = {m: {"correct_orders": 0, "correct_both_orders": []} for m in METHODS}
    per_clip, errors, names = [], [], []
    for case, frozen, result in zip(spec["cases"], cpu["cases"], results, strict=True):
        assert all(case[k] == v for k, v in frozen.items()) and case["id"] == result["case_id"]
        assert len(case["frames"]) == 10 and all(name.endswith(f"_{frame:05d}.png")
                                                for name, frame in zip(case["frames"], case["frame_grid"], strict=True))
        names.extend(case["frames"])
        assert all(a < b for a, b in zip(case["timestamps_s"], case["timestamps_s"][1:]))
        gold = "A" if case["state"] == "open" else "B"
        assert result["gold"] == gold and set(result["orders"]) == set(ORDERS)
        windows = dict(fixed=spec["static_window"], recency=[7, 8, 9], annotated=case["annotated_window"],
                       causal=list(range(10)))
        counts = dict.fromkeys(METHODS, 0)
        for order in result["orders"].values():
            for method in (*METHODS, "causal"):
                row = order[method]
                logits = row["logits"]
                assert len(logits) == 2 and all(isfinite(v) for v in logits)
                assert row["prediction"] == LETTERS[max(range(2), key=logits.__getitem__)]
                assert all(row[k] == order["native"][k] for k in ("visual_tokens", "lm_tokens", "prompt_sha256"))
                assert row["visual_tokens"] == 1960 and row["lm_tokens"] > 1960
                if method in windows:
                    assert row["slots"] == windows[method]
                if method in counts:
                    counts[method] += row["prediction"] == gold
            for left in windows:
                for right in windows:
                    if windows[left] == windows[right]:
                        assert order[left] == order[right]
            error = max(abs(a-b) for a, b in zip(order["causal"]["logits"], order["native"]["logits"], strict=True))
            assert order["parity_passed"] and error == order["parity_error"] <= 0.05
            assert order["causal"]["prediction"] == order["native"]["prediction"]
            errors.append(error)
        per_clip.append(dict(case=case["id"], fixed_support=case["support_counts"]["center"] > 0,
                             both_endpoints_opposite={0, 9}.issubset(case["opposite_slots"]), correct_orders=counts))
        for method, count in counts.items():
            methods[method]["correct_orders"] += count
            if count == 2:
                methods[method]["correct_both_orders"].append(case["id"])
    assert len(set(names)) == len(names) == frames and set(names) == set(spec["frame_receipts"])
    annotated = set(methods["annotated"]["correct_both_orders"])
    comparisons = {}
    for method in ("fixed", "recency", "native"):
        baseline = set(methods[method]["correct_both_orders"])
        comparisons[method] = dict(recoveries=sorted(annotated - baseline), losses=sorted(baseline - annotated),
                                   net=len(annotated)-len(baseline))
    summary = dict(clips=7, choice_scores=56, by_method=methods, annotated_contrasts=comparisons,
                   verified_attention_layers={"three_frame": 2, "causal": 2})
    assert len(errors) == 14 and summary == load("exports/summary.json")
    return dict(archive_sha256=digest(archive.read_bytes()), notebook_sha256=digest(notebook.read_bytes()),
                gpu=config["gpu"], verified_payloads=len(saved), verified_frames=frames,
                parity_checks=len(errors), max_parity_error=max(errors), **summary,
                per_clip=per_clip, improved_over_best_fixed=comparisons["fixed"]["net"] > 0)


if __name__ == "__main__":
    archive, notebook = map(Path, sys.argv[1:]) if len(sys.argv) == 3 else (ARCHIVE, NOTEBOOK)
    report = json.dumps(audit(archive, notebook), indent=2) + "\n"
    if len(sys.argv) == 1:
        OUTPUT.write_text(report, encoding="utf-8", newline="\n")
    print(report, end="")
