"""Audit the fixed-recent run and recompute its development decision."""
import ast
import json
import math
import sys
from pathlib import Path

from validate_twinpast_backbone_evidence import ARCHIVE as PRIOR, REVISION
from validate_twinpast_diagnostic_evidence import BUNDLE, LETTERS, ORDERS, ROOT, digest, verify_inventory

ARCHIVE = ROOT / "evidence/archives/HERMES_twinpast_fixed_recent_evidence.zip"
NOTEBOOK = ROOT / "notebooks/executed/2026-11-23_hermes-twinpast-fixed-recent.ipynb"
TEMPLATE = ROOT / "notebooks/autodl/hermes-twinpast-fixed-recent.ipynb"


def score_pair(rows, gold):
    a, b = (LETTERS.index(x) for x in gold)
    outcomes = {}
    for i, order in enumerate(ORDERS):
        left, right = rows[2 * i:2 * i + 2]
        assert (left["order"], left["branch"], right["order"], right["branch"]) == (order, "a", order, "b")
        assert [left["gold"], right["gold"]] == gold
        x, y = left["scores"], right["scores"]
        outcomes[order] = dict(
            branches_correct=(left["prediction"] == gold[0]) + (right["prediction"] == gold[1]),
            directional_change=(x[a] - x[b]) - (y[a] - y[b]))
    return dict(orders=outcomes, robust=all(row["branches_correct"] == 2 for row in outcomes.values()))


def audit(archive, notebook):
    payload = verify_inventory(archive)
    previous = verify_inventory(PRIOR)
    prior_sha = digest(PRIOR.read_bytes())
    spec = json.loads(payload["provenance/fixed_recent_cases.json"])
    assert spec["backbone_archive_sha256"] == prior_sha
    assert spec["prior_config"] == json.loads(previous["exports/run_config.json"])
    assert payload[BUNDLE] == previous[BUNDLE]
    media = {m["pair_id"]: m["decoded"] for m in json.loads(payload[BUNDLE])["media"] if "decoded" in m}
    old_cases = {x["annotation"]["pair_id"]: x for x in json.loads(previous["provenance/backbone_cases.json"])["cases"]}
    assert len(old_cases) == len(spec["cases"]) == 7
    assert old_cases.keys() == {x["annotation"]["pair_id"] for x in spec["cases"]}

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
    trace = "\n".join("".join(o.get("text", "")) for c in nb["cells"] for o in c.get("outputs", []))
    assert trace.count("MODEL_SHARD_VERIFIED ") == 4 and "TWINPAST_FIXED_RECENT_COMPLETE" in trace
    assert "TWINPAST_FIXED_RECENT_COMPLETE" in payload["logs/twinpast_fixed_recent.log"].decode()
    assert json.loads(payload["completion.json"]) == dict(status="complete", pairs=7, scores=70)
    config = json.loads(payload["exports/run_config.json"])
    assert config["model_revision"] == launch["model_revision"] == REVISION
    assert config["source_archive_sha256"] == prior_sha and config["replay_atol"] == 0.05
    assert all(config[k] == spec["prior_config"][k] for k in ("gpu", "torch", "cuda", "dtype"))
    assert all(digest(payload[f"provenance/{n}"]) == value for n, value in config["provenance_hashes"].items())
    identity = digest(json.dumps(config, sort_keys=True).encode())
    preflight = json.loads(payload["exports/preflight.json"])
    assert preflight["status"] == "pass" and preflight["pair_id"] == "twinpast_0208"

    recovered, retained, controls, pairs, failures, recent_counts = set(), set(), 0, [], [], dict.fromkeys(ORDERS, 0)
    scores = 0
    for case in spec["cases"]:
        annotation = case["annotation"]
        pair_id = annotation["pair_id"]
        saved = json.loads(payload[f"checkpoints/{pair_id}.json"])
        assert saved["identity"] == identity
        block = saved["block"]
        old = json.loads(previous[f"checkpoints/{pair_id}.json"])["block"]
        assert case["backbone_rows"] == old["rows"]
        assert all(case[k] == old_cases[pair_id][k] for k in ("annotation", "cohort", "original_prompt_rows"))
        assert block["pair_id"] == pair_id and block["cohort"] == case["cohort"]
        if old["current"]["robust"] and case["cohort"] == "historical_state":
            recovered.add(pair_id)
        fixed = media[pair_id]["a"]["decoded_rgb_sha256"][-4:]
        assert fixed == media[pair_id]["b"]["decoded_rgb_sha256"][-4:]
        original = {(r["order"], r["arm"][-1]): r for r in case["original_prompt_rows"]}
        gold = [annotation[f"answer_{b}"] for b in "ab"]
        groups = block["rows"]
        assert set(groups) == {"history_only", "history_with_recent", "recent_only"}
        assert {k: len(v) for k, v in groups.items()} == dict(history_only=4, history_with_recent=4, recent_only=2)
        scores += sum(map(len, groups.values()))
        new_checks = []
        for i, order in enumerate(ORDERS):
            recent = groups["recent_only"][i]
            assert recent["order"] == order and recent["branch"] is None
            assert recent["history_frame_hashes"] == [] and recent["recent_frame_hashes"] == fixed
            recent_counts[order] += recent["prediction"] in gold
            for j, branch in enumerate("ab"):
                k = 2 * i + j
                a, b = groups["history_only"][k], groups["history_with_recent"][k]
                baseline = old["rows"][k]
                assert a["history_frame_hashes"] == b["history_frame_hashes"] == baseline["frame_hashes"]
                assert a["recent_frame_hashes"] == [] and b["recent_frame_hashes"] == fixed
                assert a["prompt_sha256"] == b["prompt_sha256"] == recent["prompt_sha256"] == baseline["prompt_sha256"]
                assert a["lm_tokens"] == baseline["lm_tokens"]
                assert b["lm_tokens"] == a["lm_tokens"] + recent["visual_tokens"]
                assert b["visual_tokens"] == a["visual_tokens"] + recent["visual_tokens"]
                assert all(r["option_token_ids"] == original[order, branch]["label_token_ids"] for r in (a, b, recent))
                assert all(r["branch"] == branch and r["order"] == order for r in (a, b))
                delta = max(abs(x - y) for x, y in zip(a["scores"], baseline["scores"], strict=True))
                check = dict(order=order, branch=branch, archived_prediction=baseline["prediction"],
                             max_abs_score_difference=delta,
                             pass_replay=a["prediction"] == baseline["prediction"] and delta <= 0.05)
                new_checks.append(check)
                if not check["pass_replay"]:
                    failures.append(dict(pair_id=pair_id, **check))
        for rows in groups.values():
            for row in rows:
                v = row["scores"]
                assert len(v) == len(annotation["options"]) and all(math.isfinite(x) for x in v)
                assert row["prediction"] == LETTERS[max(range(len(v)), key=v.__getitem__)]
        assert block["replay_checks"] == new_checks
        for name in ("history_only", "history_with_recent"):
            assert block[name] == score_pair(groups[name], gold)
        if block["history_with_recent"]["robust"]:
            if block["cohort"] == "historical_state":
                retained.add(pair_id)
            else:
                controls += 1
        if pair_id == preflight["pair_id"]:
            assert block["smoke_checks"] == preflight["checks"] and len(preflight["checks"]) == 4
            assert block["peak_gpu_allocated_bytes"] == preflight["peak_gpu_allocated_bytes"]
        pairs.append({k: v for k, v in block.items() if k not in {"rows", "smoke_checks"}})
    assert recovered == set(spec["recovered_states"]) and scores == 70
    totals = dict(pairs=7, scores=scores, replay_failures=failures, replay_atol=0.05,
                  previously_recovered_states=sorted(recovered), retained_recovered_states=sorted(retained & recovered),
                  newly_recovered_states=sorted(retained - recovered), positive_controls_retained=controls,
                  development_criterion_met=not failures and recovered <= retained and controls >= 2, per_pair=pairs)
    assert totals == json.loads(payload["exports/summary.json"])
    return dict(archive_sha256=digest(archive.read_bytes()), notebook_sha256=digest(notebook.read_bytes()),
                prior_archive_sha256=prior_sha, verified_payloads=len(payload), preflight=preflight,
                recent_only_branch_correct=recent_counts, **totals)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        report = audit(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        report = audit(ARCHIVE, NOTEBOOK)
        destination = ROOT / "evidence/audits/twinpast_fixed_recent_2026-11-23_audit.json"
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Verified {report['verified_payloads']} ZIP payloads and {report['scores']} scores")
