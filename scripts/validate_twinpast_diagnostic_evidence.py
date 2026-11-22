"""Audit the saved TwinPast run and recompute its paired development outcomes."""
import ast
import csv
import hashlib
import io
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "evidence/archives/HERMES_twinpast_diagnostic_evidence.zip"
NOTEBOOK = ROOT / "notebooks/executed/2026-11-23_hermes-twinpast-diagnostic.ipynb"
TEMPLATE = ROOT / "notebooks/autodl/hermes-twinpast-diagnostic.ipynb"
ORDERS = ("canonical", "reversed")
ARMS = ("question_only", "recent_four", "history_a", "history_b")
LETTERS = "ABCDEFGH"
RUNNER = "provenance/twinpast_runner.py"
BUNDLE = "provenance/bundle.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_inventory(path):
    with ZipFile(path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        manifest = json.loads(archive.read("evidence_manifest.json"))
        listed = {entry["path"]: entry for entry in manifest}
        assert len(listed) == len(manifest) and set(names) == listed.keys() | {"evidence_manifest.json"}
        contents = {}
        for name, entry in listed.items():
            raw = archive.read(name)
            assert len(raw) == entry["bytes"] and digest(raw) == entry["sha256"], name
            contents[name] = raw
        return contents


def verify_notebook(path, contents):
    notebook = json.loads(path.read_bytes())
    expected = json.loads(TEMPLATE.read_bytes())
    cells = notebook["cells"]
    extra = [cell for cell in cells if "".join(cell["source"]).strip() == "!source /etc/network_turbo"]
    assert len(extra) == 1
    original = [cell for cell in cells if cell not in extra]
    assert [("".join(c["source"]).strip(), c["cell_type"]) for c in original] == [
        ("".join(c["source"]).strip(), c["cell_type"]) for c in expected["cells"]]
    assert all(c.get("execution_count") for c in original if c["cell_type"] == "code")
    assert not any(o["output_type"] == "error" for c in cells for o in c.get("outputs", []))
    source_cell, = [c for c in original if "".join(c["source"]).startswith("SOURCES = ")]
    assignment = ast.parse("".join(source_cell["source"])).body[0]
    sources = ast.literal_eval(assignment.value)
    assert all(text.encode("utf-8") == contents[name] for name, text in sources.items())
    output = "\n".join("".join(o.get("text", "")) for c in cells for o in c.get("outputs", []))
    assert "TWINPAST_PREFLIGHT_PASS" in output and "TWINPAST_DIAGNOSTIC_COMPLETE pairs=9 scores=72" in output


def collect_records(contents, bundle, config, checkpoints):
    source = ROOT / "results/twinpast_development_2026-11-23"
    assert bundle["manifest"] == json.loads((ROOT / "data/twinpast/historical_correction_development.json").read_bytes())
    assert bundle["media"] == json.loads((source / "media_check.json").read_bytes())
    assert bundle["visual"] == list(csv.DictReader((source / "pair_visual_screen.csv").read_text(
        encoding="utf-8").splitlines()))
    clear = {row["pair_id"]: row for row in bundle["visual"] if row["status"] == "clear_contrast"}
    media = {row["pair_id"]: row for row in bundle["media"]}
    annotated = json.loads(contents["provenance/selected_annotations.json"])
    assert len(clear) == len(annotated) == len(checkpoints) == 9
    assert set(clear) == {row["pair_id"] for row in annotated} == checkpoints.keys()
    identity = digest(json.dumps(config, sort_keys=True).encode())
    records = []
    for annotation in annotated:
        pair_id = annotation["pair_id"]
        bank = media[pair_id]["decoded"]
        recent = bank["a"]["decoded_rgb_sha256"][-4:]
        assert recent == bank["b"]["decoded_rgb_sha256"][-4:]
        for branch in "ab":
            assert annotation[f"answer_{branch}"] == clear[pair_id][f"observed_option_{branch}"]
            starts = bank[branch]["frame_start_s"]
            assert all(t < annotation["t_query"] for t in starts)
            assert all(any(lo - 1e-6 <= t <= hi + 1e-6 for lo, hi in annotation["evidence_intervals"][branch.upper()])
                       for t in starts[:2])
        saved = checkpoints[pair_id]
        assert saved["identity"] == identity and saved["pair_id"] == pair_id
        records.extend(saved["records"])
        rows = {(row["order"], row["arm"]): row for row in saved["records"]}
        assert len(saved["records"]) == len(rows) == 8 and set(rows) == {(o, a) for o in ORDERS for a in ARMS}
        for order in ORDERS:
            history = [rows[order, f"history_{branch}"] for branch in "ab"]
            recent_only = rows[order, "recent_four"]
            assert len({row["prompt_sha256"] for row in [*history, recent_only]}) == 1
            assert history[0]["lm_tokens"] == history[1]["lm_tokens"] > recent_only["lm_tokens"]
            for arm in ARMS:
                row = rows[order, arm]
                assert row["pair_id"] == pair_id and row["task"] == annotation["task"]
                assert row["gold_a"] == annotation["answer_a"] and row["gold_b"] == annotation["answer_b"]
                assert row["recent_frame_sha256"] == (recent if arm != "question_only" else [])
                assert row["earlier_frame_sha256"] == (
                    bank[arm[-1]]["decoded_rgb_sha256"][:2] if arm.startswith("history_") else [])
                count = len(annotation["options"])
                indices = list(range(count))
                assert row["permutation"] == (indices if order == "canonical" else indices[::-1])
                assert len(set(row["label_token_ids"])) == count
                scores = row["scores"]
                assert len(scores) == count and all(math.isfinite(s) for s in scores)
                assert row["prediction"] == LETTERS[max(indices, key=scores.__getitem__)]
                assert math.isfinite(row["forward_s"]) and row["forward_s"] >= 0
    table = list(csv.DictReader(io.StringIO(contents["exports/scores.csv"].decode("utf-8"))))
    assert len(table) == len(records) == 72
    for checkpoint, tabular in zip(records, table, strict=True):
        assert tabular == {k: json.dumps(v) if isinstance(v, list) else str(v) for k, v in checkpoint.items()}
    return records


def pair_results(records):
    groups = {}
    for row in records:
        groups.setdefault(row["pair_id"], {})[row["order"], row["arm"]] = row
    totals = {order: Counter() for order in ORDERS}
    margins = {order: [] for order in ORDERS}
    pairs = []
    for pair_id, rows in groups.items():
        pair = {"pair_id": pair_id, "task": rows["canonical", "history_a"]["task"]}
        for order in ORDERS:
            a, b, recent, question = (rows[order, arm] for arm in ("history_a", "history_b", "recent_four", "question_only"))
            gold = a["gold_a"], b["gold_b"]
            predictions = a["prediction"], b["prediction"]
            wins = sum(p == g for p, g in zip(predictions, gold, strict=True))
            totals[order]["correct_branches"] += wins
            totals[order]["both_correct"] += wins == 2
            totals[order]["unchanged"] += predictions[0] == predictions[1]
            totals[order]["wrong_reversals"] += predictions[0] != predictions[1] and wins != 2
            for label, baseline in (("recent_four", recent), ("question_only", question)):
                totals[order][f"{label}_correct_branches"] += baseline["prediction"] in gold
            for prediction, answer in zip(predictions, gold, strict=True):
                totals[order]["recoveries_vs_recent"] += prediction == answer and recent["prediction"] != answer
                totals[order]["regressions_vs_recent"] += prediction != answer and recent["prediction"] == answer
            i, j = (LETTERS.index(answer) for answer in gold)
            change = (a["scores"][i] - a["scores"][j]) - (b["scores"][i] - b["scores"][j])
            margins[order].append(change)
            pair.update({f"{order}_correct_branches": wins, f"{order}_directional_change": change})
        pair["both_orders_correct"] = all(pair[f"{order}_correct_branches"] == 2 for order in ORDERS)
        pairs.append(pair)
    for order, counter in totals.items():
        counter["median_directional_change"] = statistics.median(margins[order])
        counter["positive_direction_pairs"] = sum(value > 0 for value in margins[order])
    return pairs, totals


def audit(path, notebook):
    contents = verify_inventory(path)
    verify_notebook(notebook, contents)
    launch = json.loads(contents["launch.json"])
    config = json.loads(contents["exports/run_config.json"])
    assert all(digest(contents[name]) == value for name, value in launch["embedded_sha256"].items())
    assert digest(contents[BUNDLE]) == config["bundle_sha256"]
    assert digest(contents[RUNNER]) == config["runner_sha256"]
    assert config["model_name"] == launch["model_revision"]
    assert "TWINPAST_DIAGNOSTIC_COMPLETE pairs=9 scores=72" in contents["logs/twinpast.log"].decode("utf-8")
    assert json.loads(contents["completion.json"]) == {"status": "complete", "pairs": 9, "scores": 72}
    preflight = json.loads(contents["exports/preflight.json"])
    assert preflight["status"] == "pass" and preflight["pair_id"] == "twinpast_0208"
    assert [check["order"] for check in preflight["checks"]] == list(ORDERS)
    checkpoints = {Path(name).stem: json.loads(value) for name, value in contents.items() if name.startswith("checkpoints/")}
    records = collect_records(contents, json.loads(contents[BUNDLE]), config, checkpoints)
    pairs, totals = pair_results(records)
    original = json.loads(contents["exports/summary.json"])
    robust = sum(row["both_orders_correct"] for row in pairs)
    numeric_gate = robust >= 5 and all(row["median_directional_change"] > 0 for row in totals.values())
    assert original["pairs"] == len(pairs) == 9 and original["robust_correct_pairs"] == robust
    assert original["numeric_development_gate"] == numeric_gate
    assert all(totals[order][measure] == value
               for order, summary in original["order_summary"].items() for measure, value in summary.items())
    return dict(archive_sha256=digest(path.read_bytes()), notebook_sha256=digest(notebook.read_bytes()),
                verified_zip_payloads=len(contents), runtime=config, preflight=preflight, pairs=len(pairs),
                scores=len(records), robust_correct_pairs=robust, order_summary=totals,
                numeric_development_gate=numeric_gate, source_groups_verified=False, per_pair=pairs)


if __name__ == "__main__":
    result = audit(ARCHIVE, NOTEBOOK)
    dest = ROOT / "evidence/audits/twinpast_diagnostic_2026-11-23_audit.json"
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Verified {result['verified_zip_payloads']} ZIP payloads and {result['scores']} scores")
