"""Verify the exact-frame 7B replay against the archived 0.5B inputs and scores."""
import ast
import json
import sys
from pathlib import Path

from validate_twinpast_diagnostic_evidence import BUNDLE, ORDERS, ROOT, digest, verify_inventory
from validate_twinpast_history_only_evidence import ARCHIVE as PRIOR_ARCHIVE, analyze

ARCHIVE = ROOT / "evidence/archives/HERMES_twinpast_backbone_evidence.zip"
NOTEBOOK = ROOT / "notebooks/executed/2026-11-23_hermes-twinpast-backbone.ipynb"
TEMPLATE = ROOT / "notebooks/autodl/hermes-twinpast-backbone.ipynb"
REVISION = "0d50680527681998e456c7b78950205bedd8a068"


def audit(archive, notebook):
    payload = verify_inventory(archive)
    previous = verify_inventory(PRIOR_ARCHIVE)
    prior = json.loads(previous["provenance/prior.json"])
    annotations = {a["pair_id"]: a for a in prior["annotations"]}
    bundle = json.loads(payload[BUNDLE])
    assert payload[BUNDLE] == previous[BUNDLE]
    media = {record["pair_id"]: record["decoded"] for record in bundle["media"] if "decoded" in record}
    spec = json.loads(payload["provenance/backbone_cases.json"])
    previous_sha = digest(PRIOR_ARCHIVE.read_bytes())
    assert spec["model_revision"] == REVISION and spec["history_only_archive_sha256"] == previous_sha
    recorded = {case["annotation"]["pair_id"]: case for case in spec["cases"]}
    expected = {}
    for pair_id, annotation in annotations.items():
        checkpoint = json.loads(previous[f"checkpoints/{pair_id}.json"])
        if annotation["task"] == "HSR":
            expected[pair_id] = ("historical_state", "history_dense", 6)
        elif checkpoint["summary"]["two_history_only"]["robust"]:
            expected[pair_id] = ("positive_control", "history_two", 2)
    assert len(spec["cases"]) == len(expected) == 7 and recorded.keys() == expected.keys()
    assert {Path(n).stem for n in payload if n.startswith("checkpoints/")} == expected.keys()

    nb = json.loads(notebook.read_bytes())
    template = json.loads(TEMPLATE.read_bytes())
    assert [(c["cell_type"], "".join(c["source"]).strip()) for c in nb["cells"]] == [
        (c["cell_type"], "".join(c["source"]).strip()) for c in template["cells"]]
    sources, = [c for c in nb["cells"] if "".join(c["source"]).startswith("SOURCES = ")]
    embedded = ast.literal_eval(ast.parse("".join(sources["source"])).body[0].value)
    launch = json.loads(payload["launch.json"])
    assert set(embedded) == set(launch["embedded_sha256"])
    assert all(text.encode("utf-8") == payload[name] and digest(payload[name]) == launch["embedded_sha256"][name]
               for name, text in embedded.items())
    for i, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
        assert cell["execution_count"] == i + 1 and all(o["output_type"] != "error" for o in cell["outputs"])
    trace = "\n".join("".join(o.get("text", "")) for c in nb["cells"] for o in c.get("outputs", []))
    assert trace.count("MODEL_SHARD_VERIFIED ") == 4 and "MODEL_REVISION_PASS" in trace
    assert "TWINPAST_BACKBONE_COMPLETE" in trace
    assert "TWINPAST_BACKBONE_COMPLETE" in payload["logs/twinpast_backbone.log"].decode()
    assert json.loads(payload["completion.json"]) == dict(status="complete", pairs=7, scores=28)
    config = json.loads(payload["exports/run_config.json"])
    assert config["model_revision"] == launch["model_revision"] == REVISION and config["dtype"] == "torch.float16"
    assert config["source_history_only_archive"] == previous_sha
    assert all(digest(payload[f"provenance/{n}"]) == value for n, value in config["runtime_source_hashes"].items())
    identity = digest(json.dumps(config, sort_keys=True).encode())
    preflight = json.loads(payload["exports/preflight.json"])
    assert preflight["status"] == "pass" and preflight["pair_id"] == "twinpast_0208"
    assert [check["order"] for check in preflight["checks"]] == list(ORDERS)
    pairs = []
    for pair_id, (cohort, condition, count) in expected.items():
        case = recorded[pair_id]
        annotation = annotations[pair_id]
        assert case["annotation"] == annotation and case["cohort"] == cohort
        prior_checkpoint = json.loads(previous[f"checkpoints/{pair_id}.json"])
        old = case["reference_rows"]
        assert old == prior_checkpoint[condition] and case["original_prompt_rows"] == prior["pairs"][pair_id]
        prompt_rows = {(r["order"], r["arm"][-1]): r for r in case["original_prompt_rows"]}
        old_result = analyze(old, count, annotation, media[pair_id], prompt_rows)
        reference = {(r["order"], r["branch"]): {**r, f"gold_{r['branch']}": r["gold"]} for r in old}
        saved = json.loads(payload[f"checkpoints/{pair_id}.json"])
        assert saved["identity"] == identity
        block = saved["block"]
        for row in block["rows"]:
            baseline = reference[row["order"], row["branch"]]
            assert all(row[key] == baseline[key] for key in ("frame_indices", "frame_seconds", "frame_hashes",
                                                             "prompt_sha256", "lm_tokens"))
        current = analyze(block["rows"], count, annotation, media[pair_id], reference)
        result = dict(pair_id=pair_id, cohort=cohort, prior=old_result, current=current,
                      peak_gpu_allocated_bytes=block["peak_gpu_allocated_bytes"])
        assert result == {k: v for k, v in block.items() if k not in {"rows", "preflight"}}
        if pair_id == preflight["pair_id"]:
            assert block["preflight"] == preflight["checks"]
            assert block["peak_gpu_allocated_bytes"] == preflight["peak_gpu_allocated_bytes"]
        pairs.append(result)
    counts = {k: sum(p["current"]["robust"] for p in pairs if p["cohort"] == k)
              for k in ("historical_state", "positive_control")}
    totals = dict(pairs=7, scores=28, historical_state_robust=counts["historical_state"],
                  historical_state_total=4, positive_control_robust=counts["positive_control"],
                  positive_control_total=3, development_criterion_met=min(counts.values()) >= 2, per_pair=pairs)
    assert totals == json.loads(payload["exports/summary.json"])
    return dict(archive_sha256=digest(archive.read_bytes()), notebook_sha256=digest(notebook.read_bytes()),
                verified_payloads=len(payload), previous_archive_sha256=previous_sha, preflight=preflight,
                peak_gpu_allocated_bytes=max(r["peak_gpu_allocated_bytes"] for r in pairs), **totals)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        report = audit(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        report = audit(ARCHIVE, NOTEBOOK)
        out = ROOT / "evidence/audits/twinpast_backbone_2026-11-23_audit.json"
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Verified {report['verified_payloads']} ZIP payloads and {report['scores']} scores")
