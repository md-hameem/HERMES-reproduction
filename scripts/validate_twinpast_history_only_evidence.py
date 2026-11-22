"""Audit the history-only follow-up and independently reconstruct its outcomes."""
import ast
import json
import math
import sys
from collections import Counter
from pathlib import Path

from validate_twinpast_diagnostic_evidence import (
    ARCHIVE as PRIOR_ARCHIVE,
    BUNDLE,
    LETTERS,
    ORDERS,
    ROOT,
    digest,
    verify_inventory,
)

ARCHIVE = ROOT / "evidence/archives/HERMES_twinpast_history_only_evidence.zip"
NOTEBOOK = ROOT / "notebooks/executed/2026-11-23_hermes-twinpast-history-only.ipynb"
TEMPLATE = ROOT / "notebooks/autodl/hermes-twinpast-history-only.ipynb"


def analyze(rows, count, annotation, media, prior):
    keyed = {(r["order"], r["branch"]): r for r in rows}
    assert len(rows) == len(keyed) == 4 and set(keyed) == {(o, b) for o in ORDERS for b in "ab"}
    gold = [annotation[f"answer_{b}"] for b in "ab"]
    ai, bi = (LETTERS.index(label) for label in gold)
    results = {}
    cutoff = min(annotation["t_query"], *(media[b]["frame_start_s"][2] for b in "ab"))
    for order in ORDERS:
        a, b = (keyed[order, branch] for branch in "ab")
        assert a["lm_tokens"] == b["lm_tokens"]
        for branch, rec, answer in zip("ab", (a, b), gold, strict=True):
            earlier = media[branch]["source_frames"][:2]
            hashes = media[branch]["decoded_rgb_sha256"][:2]
            indices, times = rec["frame_indices"], rec["frame_seconds"]
            assert len(indices) == len(rec["frame_hashes"]) == len(times) == count
            assert indices == sorted(set(indices)) and times == sorted(set(times))
            assert [rec["frame_hashes"][indices.index(i)] for i in earlier] == hashes
            assert [times[indices.index(i)] for i in earlier] == media[branch]["frame_start_s"][:2]
            assert all(t < cutoff and any(lo - 1e-6 <= t <= hi + 1e-6
                       for lo, hi in annotation["evidence_intervals"][branch.upper()]) for t in times)
            original = prior[order, branch]
            assert rec["gold"] == answer == original[f"gold_{branch}"]
            assert rec["prompt_sha256"] == original["prompt_sha256"]
            assert rec["prior_prediction"] == original["prediction"] and rec["prior_scores"] == original["scores"]
            assert len(rec["scores"]) == len(annotation["options"]) and all(math.isfinite(v) for v in rec["scores"])
            assert rec["prediction"] == LETTERS[max(range(len(rec["scores"])), key=rec["scores"].__getitem__)]
            assert math.isfinite(rec["forward_s"]) and rec["forward_s"] >= 0
        results[order] = dict(branches_correct=(a["prediction"] == gold[0]) + (b["prediction"] == gold[1]),
                              directional_change=(a["scores"][ai] - a["scores"][bi]) - (b["scores"][ai] - b["scores"][bi]))
    return dict(orders=results, robust=all(value["branches_correct"] == 2 for value in results.values()))


def audit(archive, notebook):
    contents = verify_inventory(archive)
    nb = json.loads(notebook.read_bytes())
    template = json.loads(TEMPLATE.read_bytes())
    assert [(c["cell_type"], "".join(c["source"]).strip()) for c in nb["cells"]] == [
        (c["cell_type"], "".join(c["source"]).strip()) for c in template["cells"]]
    for index, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
        assert cell["execution_count"] == index + 1 and all(o["output_type"] != "error" for o in cell["outputs"])
    source, = [c for c in nb["cells"] if "".join(c["source"]).startswith("SOURCES = ")]
    embedded = ast.literal_eval(ast.parse("".join(source["source"])).body[0].value)
    launch = json.loads(contents["launch.json"])
    assert set(launch["embedded_sha256"]) == set(embedded)
    assert all(text.encode("utf-8") == contents[name] and digest(contents[name]) == launch["embedded_sha256"][name]
               for name, text in embedded.items())
    assert "TWINPAST_HISTORY_ONLY_COMPLETE" in contents["logs/twinpast_history_only.log"].decode()
    config = json.loads(contents["exports/run_config.json"])
    identity = digest(json.dumps(config, sort_keys=True).encode())
    assert launch["model_revision"] == config["model_revision"]
    assert all(digest(contents[f"provenance/{n}"]) == value for n, value in config["provenance_hashes"].items())
    preflight = json.loads(contents["exports/preflight.json"])
    assert preflight["status"] == "pass" and [c["order"] for c in preflight["checks"]] == list(ORDERS)
    original = verify_inventory(PRIOR_ARCHIVE)
    prior = json.loads(contents["provenance/prior.json"])
    assert prior["archive_sha256"] == digest(PRIOR_ARCHIVE.read_bytes())
    assert prior["annotations"] == json.loads(original["provenance/selected_annotations.json"])
    assert contents[BUNDLE] == original[BUNDLE]
    media = {r["pair_id"]: r for r in json.loads(contents[BUNDLE])["media"]}
    expected = json.loads(contents["exports/summary.json"])
    reports, counts = [], Counter()
    for annotation in prior["annotations"]:
        pair = annotation["pair_id"]
        name = f"checkpoints/{pair}.json"
        original_rows = [r for r in json.loads(original[name])["records"] if r["arm"].startswith("history_")]
        assert prior["pairs"][pair] == original_rows
        reference = {(r["order"], r["arm"][-1]): r for r in original_rows}
        block = json.loads(contents[name])
        assert block["identity"] == identity and block["pair_id"] == pair
        two = analyze(block["history_two"], 2, annotation, media[pair]["decoded"], reference)
        report = dict(pair_id=pair, task=annotation["task"], two_history_only=two,
                      prior_history_with_recent_robust=all(r["prediction"] == r[f"gold_{b}"] for (_, b), r in reference.items()))
        counts["two_scores"] += len(block["history_two"])
        if annotation["task"] == "HSR":
            assert contents[f"exports/review/{pair}_two.jpg"].startswith(b"\xff\xd8")
            if not two["robust"]:
                dense = analyze(block["history_dense"], 6, annotation, media[pair]["decoded"], reference)
                assert contents[f"exports/review/{pair}_dense.jpg"].startswith(b"\xff\xd8")
                report["dense_history_only"] = dense
                counts["dense_scores"] += len(block["history_dense"])
        assert report == block["summary"]
        reports.append(report)
    summary = dict(pairs=len(reports), two_history_only_robust=sum(r["two_history_only"]["robust"] for r in reports),
                   restored_from_original=sum(r["two_history_only"]["robust"] and not r["prior_history_with_recent_robust"] for r in reports),
                   lost_from_original=sum(not r["two_history_only"]["robust"] and r["prior_history_with_recent_robust"] for r in reports),
                   dense_pairs=sum("dense_history_only" in r for r in reports),
                   dense_robust=sum(r.get("dense_history_only", {}).get("robust", False) for r in reports), per_pair=reports)
    assert summary == expected
    assert json.loads(contents["completion.json"]) == dict(status="complete", pairs=9, dense_pairs=4)
    assert counts == {"two_scores": 36, "dense_scores": 16}
    return dict(archive_sha256=digest(archive.read_bytes()), notebook_sha256=digest(notebook.read_bytes()),
                previous_archive_sha256=prior["archive_sha256"], verified_payloads=len(contents), preflight=preflight,
                score_counts=dict(counts), **summary)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        summary = audit(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        summary = audit(ARCHIVE, NOTEBOOK)
        out = ROOT / "evidence/audits/twinpast_history_only_2026-11-23_audit.json"
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Verified {summary['verified_payloads']} ZIP payloads and {sum(summary['score_counts'].values())} scores")
