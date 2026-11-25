"""Audit archived sub-second selections, uniform replay and the decision rule."""
import ast
import json
import math
from pathlib import Path
import sys

from validate_twinpast_diagnostic_evidence import LETTERS, ORDERS, ROOT, digest, verify_inventory
from validate_twinpast_fixed_recent_evidence import score_pair
from validate_twinpast_temporal_windows_evidence import ARCHIVE as PRIOR, MANIFEST

ARCHIVE = ROOT / "evidence/archives/HERMES_twinpast_subsecond_evidence.zip"
NOTEBOOK = ROOT / "notebooks/executed/2026-11-23_hermes-twinpast-subsecond.ipynb"
TEMPLATE = ROOT / "notebooks/autodl/hermes-twinpast-subsecond.ipynb"
ARMS = ("uniform", "annotated_native")


def audit(archive, notebook):
    saved, prior = verify_inventory(archive), verify_inventory(PRIOR)
    spec = json.loads(saved["provenance/subsecond_cases.json"])
    assert spec["source_archive_sha256"] == digest(PRIOR.read_bytes())
    accepted = json.loads(prior["exports/run_config.json"])["accepted"]
    annotations = {x["pair_id"]: x for x in json.loads(MANIFEST.read_bytes())["candidates"]}
    eligible = [p for p in accepted if all((annotations[p]["evidence_intervals"] or {}).get(b) for b in "AB")]
    assert [x["annotation"]["pair_id"] for x in spec["cases"]] == eligible and len(eligible) == 7
    assert {Path(n).stem for n in saved if n.startswith("checkpoints/")} == set(eligible)
    template, executed = (json.loads(p.read_bytes()) for p in (TEMPLATE, notebook))
    assert [(c["cell_type"], "".join(c["source"]).strip()) for c in executed["cells"]] == [
        (c["cell_type"], "".join(c["source"]).strip()) for c in template["cells"]]
    for i, cell in enumerate(c for c in executed["cells"] if c["cell_type"] == "code"):
        assert cell["execution_count"] == i + 1 and all(o["output_type"] != "error" for o in cell["outputs"])
    cell, = [c for c in executed["cells"] if "".join(c["source"]).startswith("SOURCES = ")]
    embedded = ast.literal_eval(ast.parse("".join(cell["source"])).body[0].value)
    launch = json.loads(saved["launch.json"])
    assert embedded.keys() == launch["embedded_sha256"].keys()
    assert all(s.encode() == saved[n] and launch["embedded_sha256"][n] == digest(saved[n]) for n, s in embedded.items())
    assert "TWINPAST_SUBSECOND_COMPLETE" in saved["logs/twinpast_subsecond.log"].decode()
    assert json.loads(saved["completion.json"]) == dict(status="complete", pairs=7, scores=56)
    config = json.loads(saved["exports/run_config.json"])
    assert config["model_revision"] == launch["model_revision"] == spec["model_revision"]
    assert config["replay_atol"] == 0.05
    assert all(digest(saved[f"provenance/{n}"]) == value for n, value in config["hashes"].items())
    identity = digest(json.dumps(config, sort_keys=True).encode())

    robust, correct = {m: [] for m in ARMS}, {m: 0 for m in ARMS}
    missed_both, failures, total, max_replay = [], [], 0, 0.0
    for case in spec["cases"]:
        annotation = case["annotation"]
        pair = annotation["pair_id"]
        assert annotation == annotations[pair]
        old = json.loads(prior[f"checkpoints/{pair}.json"])["block"]["branches"]
        checkpoint = json.loads(saved[f"checkpoints/{pair}.json"])
        block = checkpoint["block"]
        assert checkpoint["identity"] == identity and block["pair_id"] == pair and set(block["rows"]) == set(ARMS)
        covered = []
        for branch, trace in block["sampling"].items():
            baseline = case["baseline"][branch]
            original = old[branch]
            assert baseline == dict(sample=original["sample"], selection=original["selection"]["uniform"],
                                   scores={o: a["logits"]["uniform"] for o,a in original["answers"].items()})
            intervals = annotation["evidence_intervals"][branch.upper()]
            samples = baseline["sample"]["timestamps_s"]
            covered.append(any(lo-1e-6 <= t <= hi+1e-6 for t in samples[:-4] for lo,hi in intervals))
            indices, times = trace["indices"], trace["seconds"]
            assert len(indices) == len(times) == len(trace["hashes"]) == 6
            assert indices == sorted(set(indices)) and times == sorted(set(times))
            assert trace["shared_recent_hashes"] == original["sample"]["hashes"][-4:]
            assert all(0 <= t < samples[-4] for t in times)
            in_interval = [i for i,t in zip(indices,times,strict=True) if any(lo-1e-6 <= t <= hi+1e-6 for lo,hi in intervals)]
            assert len(in_interval) == trace["matching_selected"] == min(trace["eligible_native_frames"], 6)
            if len(in_interval) < 6:
                uniform = [baseline["sample"]["indices"][i] for i in baseline["selection"][:-4]]
                assert [i for i in indices if i not in in_interval] == [i for i in uniform if i not in in_interval][:6-len(in_interval)]
        if not any(covered):
            missed_both.append(pair)
        gold = [annotation[f"answer_{b}"] for b in "ab"]
        checks = []
        for arm in ARMS:
            assert saved[f"exports/review/{pair}_{arm}.jpg"].startswith(b"\xff\xd8")
            rows = block["rows"][arm]
            assert len(rows) == 4
            result = score_pair(rows, gold)
            assert result == block["outcomes"][arm]
            correct[arm] += sum(x["branches_correct"] for x in result["orders"].values())
            if result["robust"]:
                robust[arm].append(pair)
            for row, match in zip(rows, block["rows"]["uniform"], strict=True):
                assert all(row[k] == match[k] for k in ("order","branch","prompt_sha256","lm_tokens"))
                values = row["scores"]
                assert len(values) == len(annotation["options"]) and all(math.isfinite(v) for v in values)
                assert row["prediction"] == LETTERS[max(range(len(values)),key=values.__getitem__)]
                if arm == "uniform":
                    previous = case["baseline"][row["branch"]]["scores"][row["order"]]
                    delta = max(abs(x-y) for x,y in zip(values,previous,strict=True))
                    check = dict(order=row["order"], branch=row["branch"], max_abs=delta, passed=delta <= 0.05
                                 and row["prediction"] == LETTERS[max(range(len(previous)),key=previous.__getitem__)])
                    checks.append(check)
                    max_replay = max(max_replay,delta)
                    if not check["passed"]:
                        failures.append(dict(pair_id=pair,**check))
            total += len(rows)
        assert checks == block["replay_checks"]
    assert total == 56
    recovered = sorted(set(robust["annotated_native"]) - set(robust["uniform"]))
    summary = dict(pairs=7, scores=total, robust_pairs=robust, recovered_pairs=recovered,
                   replay_failures=failures, development_criterion_met=len(recovered)>=2 and not failures)
    assert summary == json.loads(saved["exports/summary.json"])
    return dict(archive_sha256=digest(archive.read_bytes()), notebook_sha256=digest(notebook.read_bytes()),
                verified_payloads=len(saved), correct_branch_orders=correct, max_abs_replay_difference=max_replay,
                prior_1fps_missed_both_branches=missed_both, recovered_from_missed_both=sorted(set(recovered)&set(missed_both)),
                **summary)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        report = audit(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        report = audit(ARCHIVE, NOTEBOOK)
        out = ROOT / "evidence/audits/twinpast_subsecond_2026-11-23_audit.json"
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Verified {report['verified_payloads']} payloads and {report['scores']} scores")
