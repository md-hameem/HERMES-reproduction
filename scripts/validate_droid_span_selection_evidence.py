"""Verify the DROID span-selection run and recompute its decision metrics."""
import ast
from collections import Counter
import json
from math import isfinite
from pathlib import Path
from statistics import median
import sys

import numpy as np

from droid_span_selector import RECENCY, WINDOWS, endpoint_window
from validate_droid_natural_history_evidence import verify_frame_pack
from validate_twinpast_diagnostic_evidence import LETTERS, ORDERS, ROOT, digest, verify_inventory

ARCHIVE = ROOT / "evidence/archives/HERMES_droid_span_selection_evidence.zip"
NOTEBOOK = ROOT / "notebooks/executed/2026-11-28_hermes-droid-span-selection.ipynb"
TEMPLATE = ROOT / "notebooks/autodl/hermes-droid-span-selection.ipynb"
MANIFEST = ROOT / "data/twinpast/droid_span_cohort_v1.json"
REVIEW = ROOT / "data/twinpast/droid_span_review_v1.json"
FRAMES = ROOT / "evidence/local/droid-span-cohort/DROID_span_cohort_frames.zip"
METHODS = ("native", "recency", "endpoint", "selector", "annotated")
TASKS = ("first", "last")


def audit(archive, notebook):
    saved = verify_inventory(archive)

    def load(name):
        return json.loads(saved[name])

    cells = json.loads(notebook.read_bytes())["cells"]
    expected = json.loads(TEMPLATE.read_bytes())["cells"]
    assert [(c["cell_type"], c["source"]) for c in cells] == [(c["cell_type"], c["source"]) for c in expected]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    for n, cell in enumerate(code_cells, 1):
        assert cell["execution_count"] == n and all(o["output_type"] != "error" for o in cell["outputs"])
    sources = ast.literal_eval(ast.parse("".join(cells[4]["source"])).body[0].value)
    assert all(s.encode() == saved[name] for name, s in sources.items())
    assert load("launch.json")["embedded_sha256"] == {n: digest(saved[n]) for n in sources}
    for name, path in (("droid_span_selector.py", "droid_span_selector.py"),
                       ("droid_span_runner.py", "autodl_droid_span_selection.py")):
        assert saved[f"provenance/{name}"].decode() == (ROOT / "scripts" / path).read_text(encoding="utf-8")
    log = saved["logs/droid_span_selection.log"]
    assert b"DROID_SPAN_SELECTION_COMPLETE" in log and log.count(b"DROID_SPAN_SCENE ") == 6
    assert b"PINNED_ENV_PASS" in saved["logs/environment.log"]
    assert load("completion.json") == {"status": "complete", "scenes": 6}
    config = load("exports/run_config.json")
    spec = load("provenance/droid_span_cohort.json")
    frozen = json.loads(MANIFEST.read_bytes())
    assert all(spec[k] == v for k, v in frozen.items()) and spec["review_sha256"] == digest(REVIEW.read_bytes())
    assert config["model_revision"] == spec["model_revision"] == load("launch.json")["model_revision"]
    assert config["attention_backend"] in ("sdpa", "eager")
    frames = verify_frame_pack(FRAMES, spec)
    cases, blocks = spec["cases"], load("exports/pair_scores.json")
    scenes = {(c["source_site"], c["scene_id"]) for c in cases}
    episodes = {b["source_episode"] for c in cases for b in c["branches"].values()}
    assert len(scenes) == len(cases) == len(blocks) == 6 and len(episodes) == 12 and frames == 120
    assert scenes.isdisjoint(map(tuple, spec["excluded_prior_scenes"]))
    assert episodes.isdisjoint(spec["excluded_prior_episodes"])

    totals = {t: {m: Counter(robust=0, correct=0, support=0) for m in METHODS} for t in TASKS}
    pairs, changed = {}, []
    errors, shifts, pooling_ms, selector_ms = [], Counter(), [], []
    for block, case in zip(blocks, cases, strict=True):
        pair_id = case["pair_id"]
        assert block["pair_id"] == pair_id and set(block["branches"]) == {"a", "b"}
        pairs[pair_id] = {}
        pooling_ms.extend(b["selector_pooling_s"] * 1000 for b in block["branches"].values())
        for task in TASKS:
            correct = Counter()
            for side, branch in block["branches"].items():
                trace = branch["tasks"][task]
                label = case["branches"][side]["labels"][task]
                gold, support = label["gold"], set(label["support_slots"])
                selection = {k: tuple(v) for k, v in trace["selection"].items()}
                question = case["questions"][task]
                assert selection["recency"] == RECENCY and selection["endpoint"] == endpoint_window(question)
                eligible = WINDOWS[:3] if task == "first" else WINDOWS[-3:]
                anchor = 0 if task == "first" else 9
                scored = [(tuple(row["slots"]), row["contrast"]) for row in trace["contrast"]]
                assert tuple(w for w, _ in scored) == eligible and all(isfinite(v) for _, v in scored)
                assert selection["selector"] == max(scored, key=lambda x: (round(x[1], 8), -abs(x[0][1]-anchor)))[0]
                assert tuple(label["annotation_window"]) == selection["annotated"]
                assert selection["annotated"] == max(eligible, key=lambda w: (len(support.intersection(w)),
                                                                            -abs(w[1]-anchor)))
                assert selection["causal"] == tuple(range(10))
                assert set(trace["orders"]) == set(ORDERS)
                selector_ms.append(trace["selector_s"] * 1000)
                if selection["selector"] != selection["endpoint"]:
                    changed.append(dict(pair=pair_id, branch=side, task=task, selector=selection["selector"],
                                        endpoint=selection["endpoint"]))
                for method in METHODS:
                    totals[task][method]["support"] += len(support.intersection(selection.get(method, range(10))))
                for scores in trace["orders"].values():
                    native = scores["native"]
                    assert native["visual_tokens"] == 1960 and native["lm_tokens"] > 1960
                    for method in (*METHODS, "causal"):
                        row = scores[method]
                        values = row["logits"]
                        assert len(values) == 2 and all(isfinite(v) for v in values)
                        assert row["prediction"] == LETTERS[max(range(2), key=values.__getitem__)]
                        assert all(row[k] == native[k] for k in ("prompt_sha256", "lm_tokens", "visual_tokens"))
                        if method != "native":
                            assert tuple(row["slots"]) == selection[method]
                        if method in METHODS:
                            correct[method] += row["prediction"] == gold
                    for left in selection:
                        for right in selection:
                            if selection[left] == selection[right]:
                                assert scores[left] == scores[right]
                    error = max(abs(a-b) for a, b in zip(native["logits"], scores["causal"]["logits"], strict=True))
                    assert scores["passed"] and error == scores["causal_max_abs"] <= 0.05
                    assert scores["causal"]["prediction"] == native["prediction"]
                    errors.append(error)
                    for method in ("recency", "endpoint", "native"):
                        before, after = (scores[m]["prediction"] == gold for m in (method, "selector"))
                        shifts[(method, "wrong_to_correct")] += after and not before
                        shifts[(method, "correct_to_wrong")] += before and not after
            pairs[pair_id][task] = {m: correct[m] for m in METHODS}
            for method in METHODS:
                totals[task][method]["correct"] += correct[method]
                totals[task][method]["robust"] += correct[method] == 4

    assert len(errors) == 48
    sampled = np.random.default_rng(20261127).integers(len(cases), size=(10_000, len(cases)))
    contrasts = {}
    for method in ("recency", "endpoint"):
        diffs = np.array([sum(int(row[t]["selector"] == 4) - int(row[t][method] == 4) for t in TASKS)/2
                          for row in pairs.values()])
        contrasts[method] = dict(absolute=float(diffs.mean()),
                                 scene_bootstrap_95=np.quantile(diffs[sampled].mean(axis=1), [.025, .975]).tolist())
    summary = dict(scenes=len(scenes), tasks=totals, contrasts=contrasts, total_reported_choice_scores=240,
                   verified_attention_layers={"three_frame": 2, "causal": 2})
    assert summary == load("exports/summary.json")
    return dict(archive_sha256=digest(archive.read_bytes()), notebook_sha256=digest(notebook.read_bytes()),
                verified_payloads=len(saved), verified_frames=frames, gpu=config["gpu"],
                parity_checks=len(errors), max_causal_logit_error=max(errors), **summary,
                distinct_selector_decisions=changed,
                branch_order_changes={m: {k: shifts[(m, k)] for k in ("wrong_to_correct", "correct_to_wrong")}
                                      for m in ("recency", "endpoint", "native")},
                median_pooling_ms=median(pooling_ms), median_selection_ms=median(selector_ms),
                per_scene_correct_branch_orders=pairs,
                advancement_criterion_met=all(r["absolute"] > 0 for r in contrasts.values()))


if __name__ == "__main__":
    archive, notebook = map(Path, sys.argv[1:]) if len(sys.argv) == 3 else (ARCHIVE, NOTEBOOK)
    output = json.dumps(audit(archive, notebook), indent=2) + "\n"
    if len(sys.argv) == 1:
        (ROOT / "evidence/audits/droid_span_selection_2026-11-28_audit.json").write_text(
            output, encoding="utf-8", newline="\n")
    print(output, end="")
