"""Independently verify the candidate-pool experiment and its advancement rule."""
import ast
import json
import math
import sys
from pathlib import Path

from validate_twinpast_automatic_evidence import ARCHIVE as AUTO
from validate_twinpast_diagnostic_evidence import BUNDLE, LETTERS, ORDERS, ROOT, digest, verify_inventory
from validate_twinpast_fixed_recent_evidence import ARCHIVE as FIXED, score_pair

ARCHIVE = ROOT / "evidence/archives/HERMES_twinpast_candidate_pool_evidence.zip"
NOTEBOOK = ROOT / "notebooks/executed/2026-11-23_hermes-twinpast-candidate-pool.ipynb"
TEMPLATE = ROOT / "notebooks/autodl/hermes-twinpast-candidate-pool.ipynb"
ARMS = ("curated_replay", "capped_pool", "dense_pool")


def audit(archive, notebook):
    saved = verify_inventory(archive)
    auto, fixed = verify_inventory(AUTO), verify_inventory(FIXED)
    prior = json.loads(fixed["provenance/fixed_recent_cases.json"])
    cases = {c["annotation"]["pair_id"]: c for c in prior["cases"] if c["cohort"] == "historical_state"}
    media = {r["pair_id"]: r for r in json.loads(fixed[BUNDLE])["media"]}
    spec = json.loads(saved["provenance/candidate_pool_cases.json"])
    assert spec["source_archives"] == {"automatic": digest(AUTO.read_bytes()), "fixed_recent": digest(FIXED.read_bytes())}
    assert len(cases) == len(spec["cases"]) == 4
    assert {c["annotation"]["pair_id"] for c in spec["cases"]} == cases.keys()
    nb, template = (json.loads(p.read_bytes()) for p in (notebook, TEMPLATE))
    assert [(c["cell_type"], "".join(c["source"]).strip()) for c in nb["cells"]] == [
        (c["cell_type"], "".join(c["source"]).strip()) for c in template["cells"]]
    for i, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
        assert cell["execution_count"] == i + 1 and all(o["output_type"] != "error" for o in cell["outputs"])
    source, = [c for c in nb["cells"] if "".join(c["source"]).startswith("SOURCES = ")]
    embedded = ast.literal_eval(ast.parse("".join(source["source"])).body[0].value)
    launch = json.loads(saved["launch.json"])
    assert embedded.keys() == launch["embedded_sha256"].keys()
    assert all(s.encode() == saved[n] and launch["embedded_sha256"][n] == digest(saved[n]) for n, s in embedded.items())
    assert "TWINPAST_CANDIDATE_POOL_PREFLIGHT_PASS" in saved["logs/twinpast_candidate_pool.log"].decode()
    assert "TWINPAST_CANDIDATE_POOL_COMPLETE" in saved["logs/twinpast_candidate_pool.log"].decode()
    assert json.loads(saved["completion.json"]) == dict(status="complete", pairs=4, scores=48)
    config = json.loads(saved["exports/run_config.json"])
    assert config["model_revision"] == launch["model_revision"] == spec["model_revision"] == prior["prior_config"]["model_revision"]
    assert config["source_archives"] == spec["source_archives"] and config["replay_atol"] == 0.05
    assert all(digest(saved[f"provenance/{n}"]) == value for n, value in config["hashes"].items())
    identity = digest(json.dumps(config, sort_keys=True).encode())
    robust = {name: [] for name in ARMS}
    failures, sizing, maximum, scores = [], [], 0.0, 0
    for case in spec["cases"]:
        annotation = case["annotation"]
        pair = annotation["pair_id"]
        assert annotation == cases[pair]["annotation"]
        old = json.loads(fixed[f"checkpoints/{pair}.json"])["block"]["rows"]["history_with_recent"]
        refs = {(r["order"], r["branch"]): r for r in cases[pair]["backbone_rows"]}
        assert case["curated_rows"] == [{**r, **{k: refs[r["order"], r["branch"]][k]
                for k in ("frame_indices", "frame_seconds", "frame_hashes")}} for r in old]
        auto_branch = json.loads(auto[f"checkpoints/{pair}.json"])["block"]["branches"]
        metadata = media[pair]
        assert case["recent_grid_s"] == metadata["grid_s"][-4:]
        for branch in "ab":
            checked = metadata["decoded"][branch]
            assert case["recent"][branch] == dict(source_frames=checked["source_frames"][-4:],
                decoded_rgb_sha256=checked["decoded_rgb_sha256"][-4:], start_s=checked["frame_start_s"][-4])
            assert case["capped_grid_s"] == auto_branch[branch]["sample"]["grid_s"][:-4]
        cutoff = min(case["recent_grid_s"][0], *(r["start_s"] for r in case["recent"].values()))
        assert case["dense_grid_s"] == list(range(math.ceil(cutoff))) and set(case["capped_grid_s"]) < set(case["dense_grid_s"])
        checkpoint = json.loads(saved[f"checkpoints/{pair}.json"])
        block = checkpoint["block"]
        assert checkpoint["identity"] == identity and block["pair_id"] == pair and set(block["rows"]) == set(ARMS)
        row_replays = []
        for name in ARMS:
            rows = block["rows"][name]
            assert len(rows) == 4
            scores += len(rows)
            gold = [annotation[f"answer_{b}"] for b in "ab"]
            result = score_pair(rows, gold)
            assert result == block["outcomes"][name]
            if result["robust"]:
                robust[name].append(pair)
            for row, baseline in zip(rows, old, strict=True):
                assert all(row[k] == baseline[k] for k in ("order", "branch", "gold", "lm_tokens", "prompt_sha256"))
                values = row["scores"]
                assert len(values) == len(annotation["options"]) and all(math.isfinite(x) for x in values)
                assert row["prediction"] == LETTERS[max(range(len(values)), key=values.__getitem__)]
                if name == "curated_replay":
                    delta = max(abs(x - y) for x, y in zip(values, baseline["scores"], strict=True))
                    maximum = max(maximum, delta)
                    check = dict(order=row["order"], branch=row["branch"], max_abs=delta,
                                 passed=delta <= 0.05 and row["prediction"] == baseline["prediction"])
                    row_replays.append(check)
                    if not check["passed"]:
                        failures.append(dict(pair_id=pair, **check))
        assert row_replays == block["curated_replay_checks"]
        for branch, samplings in block["sampling"].items():
            for arm, trace in samplings.items():
                grid = case[f"{arm.split('_')[0]}_grid_s"]
                times = trace["candidate_timestamps_s"]
                assert trace["grid_s"] == grid and len(times) == len(grid)
                assert all(0 <= t <= s < cutoff for t, s in zip(times, grid, strict=True))
                eligible = [i for i, t in enumerate(times) if any(
                    lo - 1e-6 <= t <= hi + 1e-6 for lo, hi in annotation["evidence_intervals"][branch.upper()])]
                selected = ([eligible[i * (len(eligible) - 1) // 5] for i in range(6)] if len(eligible) >= 6
                            else sorted(eligible + [i for i in range(len(times)) if i not in eligible][:6-len(eligible)]))
                assert trace["chosen_timestamps_s"] == [times[i] for i in selected]
                assert trace["chosen_in_intervals"] == len(set(selected) & set(eligible))
                assert len(trace["chosen_hashes"]) == len(set(trace["chosen_indices"])) == 6
                sizing.append(dict(pair_id=pair, branch=branch, arm=arm, pool_size=len(grid),
                                   evidence_frames=trace["chosen_in_intervals"]))
    assert len(sizing) == 16 and scores == 48
    gains = sorted(set(robust["dense_pool"]) - set(robust["capped_pool"]))
    losses = sorted(set(robust["capped_pool"]) - set(robust["dense_pool"]))
    summary = dict(pairs=4, scores=scores, robust_pairs=robust, dense_only=gains, capped_only=losses,
                   replay_failures=failures, pool_expansion_supported=bool(gains) and not losses and not failures)
    assert summary == json.loads(saved["exports/summary.json"])
    return dict(archive_sha256=digest(archive.read_bytes()), notebook_sha256=digest(notebook.read_bytes()),
                verified_payloads=len(saved), max_abs_replay_difference=maximum, sampling=sizing, **summary)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        report = audit(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        report = audit(ARCHIVE, NOTEBOOK)
        out = ROOT / "evidence/audits/twinpast_candidate_pool_2026-11-23_audit.json"
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Verified {report['verified_payloads']} archive payloads and {report['scores']} scores")
