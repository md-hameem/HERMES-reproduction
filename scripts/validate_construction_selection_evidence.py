"""Verify the construction selector evidence without executing uploaded source."""
import ast
import json
import math
import random
import statistics
from zipfile import ZipFile

from make_construction_selection_notebook import DEST, ROOT, reference_data
from validate_causal_pilot_evidence import check_manifest, check_prediction, sha, table, write_csv
from validate_context_diagnostic_evidence import expected_groups
from validate_input_diagnostic_evidence import compare_csv
from validate_stage_attribution_evidence import fingerprint

DATE = "2026-11-21"
ARCHIVE = ROOT / "evidence/archives/HERMES_construction_selection_evidence.zip"
ARCHIVE_SHA = "ff35ca3c7431f911a904d486acd0ec80bbbd585545ab672feb2c28a5e4144cfa"
NOTEBOOK_SHA = "5c45d70d0ec8b351b55623165255303dbe3235ef78e11ab2a161ace054014b8b"
ARMS = ("selected_two", "latest_two", "attention_mass_two", "no_history", "all_history", "compact_recent4")
SOURCES = dict(zip(
    ("stage_runner.py", "provenance/stage_base.py", "provenance/matched_base.py", "provenance/short_base.py",
     "stage_context_utils.py", "context_input_utils.py", "diagnostic_pilot_utils.py"),
    ("RUNNER_SOURCE", "STAGE_BASE_SOURCE", "MATCHED_BASE_SOURCE", "SHORT_BASE_SOURCE",
     "CONTEXT_SOURCE", "INPUT_SOURCE", "UTILS_SOURCE"),
))


def audit(tables, output):
    assert sha(ARCHIVE.read_bytes()) == ARCHIVE_SHA and sha(DEST.read_bytes()) == NOTEBOOK_SHA
    with ZipFile(ARCHIVE) as archive:
        manifest, payload = check_manifest(archive)

    def read(name):
        return json.loads(payload[name])

    reference, config, completion = (read(n) for n in ("reference.json", "exports/run_config.json", "exports/completion.json"))
    assert reference == reference_data()
    assert config["short_query_archive_sha256"] == reference["short_query_archive_sha256"]
    assert manifest["status"] == completion["status"] == "complete"
    assert (completion["main_questions"], completion["main_answers"]) == (40, 240)
    assert config["arms"] == list(ARMS) and config["history_budget_frames"] == 2
    assert config["reference_sha256"] == sha(payload["reference.json"])
    assert config["sources"] == {n: sha(payload[n]) for n in SOURCES}
    assert read("launch.json")["runner_sha256"] == config["sources"]["stage_runner.py"]
    assert fingerprint({k: v for k, v in config.items() if k != "config_hash"}) == config["config_hash"] == completion["config_hash"]
    cells = json.loads(DEST.read_bytes())["cells"]
    assert all(not c.get("outputs") for c in cells)
    embedded_cell = next(c for c in cells if c["id"] == "selection-08")
    embedded = {}
    for node in ast.parse("".join(embedded_cell["source"])).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in (*SOURCES.values(), "REFERENCE_JSON"):
                assert name not in embedded
                embedded[name] = ast.literal_eval(node.value).encode()
    assert all(embedded[v] == payload[n] for n, v in SOURCES.items())
    assert embedded["REFERENCE_JSON"] == payload["reference.json"]
    log = payload["captured_logs/selection.log"].decode("utf-8")
    assert "SHORT_QUERY_PREFLIGHT_PASS" in log and "CONSTRUCTION_SELECTION_COMPLETE" in log
    assert log.count("QUESTION_BLOCK_DONE ") == 40
    assert read("exports/matched_preflight.json")["status"] == "pass"
    fp32 = read("exports/short_preflight.json")["torch.float32"]
    assert len(fp32) == 3 and all(r["argmax_equal"] and r["exceeds_active_bound"] == 0 for r in fp32)

    old = {(r["sample_id"], r["qa_index"], r["arm"]): r["choice"] for r in reference["baseline_choices"]}
    videos = {v["sample_id"]: v for v in read("exports/replay_annotations.json")}
    assert videos.keys() == {v["sample_id"] for v in reference["videos"]}
    checkpoints, receipts, replays = {}, {}, 0
    for prior in reference["videos"]:
        sid = prior["sample_id"]
        video = videos[sid]
        assert all(video[n] == prior[n] for n in ("video_sha256", "video_id", "conversations"))
        feature = read(f"provenance/features_{sid}.json")
        assert feature["feature_fingerprint"] == fingerprint(feature["metadata"])
        assert feature["timestamp_rows"] == video["timestamps"] and feature["origin"] == "verified_historical_cache"
        for qi, qa in enumerate(video["conversations"]):
            end = math.ceil(float(qa["end_time"]) * .5)
            all_frames = list(range(max(0, end - 20), end))
            history, recent = all_frames[:-4], all_frames[-4:]
            assert all(float(video["timestamps"][i]["start_s"]) <= qa["end_time"] + 1e-6 for i in all_frames)
            name = f"sample_{sid}_q{qi}.json"
            checkpoint, prep = read(f"checkpoints/{name}"), read(f"preparation/{name}")
            expected_order = list(ARMS)
            random.Random(config["seed"] + sid * 31 + qi).shuffle(expected_order)
            assert prep["answer_order"] == expected_order
            assert (prep["history_indices"], prep["recent_indices"]) == (history, recent)
            score = prep["score_trace"]
            if history:
                assert len(score["attention_mass"]) == len(history)
                assert len(score["selected_residual_mse"]) == min(2, len(history))
                assert sum(score["attention_mass"]) <= 1 + 1e-6
                assert all(math.isfinite(x) and x >= 0 for x in score["attention_mass"] + score["selected_residual_mse"])
            else:
                assert score == {} and prep["score_s"] == 0
            rows = checkpoint["rows"]
            assert [r["method"] for r in rows] == list(ARMS)
            assert checkpoint["rows_sha256"] == fingerprint(rows)
            assert checkpoint["key"] == fingerprint(dict(config=config["config_hash"], feature=feature["feature_fingerprint"],
                                                          sample=sid, qa=qi, h=history, r=recent))
            known = {"latest_two": history[-2:], "no_history": [], "all_history": history, "compact_recent4": []}
            mass = sorted(range(len(history)), key=lambda i: (score["attention_mass"][i], i), reverse=True)[:2]
            known["attention_mass_two"] = sorted(history[i] for i in mass)
            by_subset = {}
            for row in rows:
                arm = row["method"]
                check_prediction(row, video, qi)
                assert row["sample_id"] == sid and row["feature_fingerprint"] == feature["feature_fingerprint"]
                assert (row["history_indices"], row["recent_indices"]) == (history, recent)
                selected = row["selected_history"]
                assert selected == sorted(set(selected)) and all(i in history for i in selected)
                if arm == "selected_two":
                    assert len(selected) == min(2, len(history))
                else:
                    assert selected == known[arm]
                assert row["recent_groups"] == expected_groups(recent, "original", feature["metadata"])
                assert row["construction_history_slots"] == 196 * len(selected)
                assert row["score_s"] == (prep["score_s"] if arm in ("selected_two", "attention_mass_two") else 0)
                assert all(math.isfinite(row[k]) and row[k] >= 0 for k in ("construction_s", "score_s", "extraction_s", "answer_s"))
                past = 13 + 196 * len(recent)
                pos = past if arm == "compact_recent4" else 13 + 196 * len(all_frames)
                assert row["query_kv_bytes"] == 12288 * past
                assert 1 <= len(row["position_trace"]) <= 17
                for i, event in enumerate(row["position_trace"]):
                    n = event["query_tokens"]
                    assert n > 1 if i == 0 else n == 1
                    assert (event["past_tokens"], event["logical_start"]) == (past, pos)
                    past, pos = past + n, pos + n
                key = sid, qi, arm
                expected = old.get(key)
                assert row["reference_choice"] == expected
                assert row["reference_matches"] is (True if expected is not None else None)
                if expected is not None:
                    assert row["pred_choice"] == expected
                    replays += 1
                if arm != "compact_recent4":
                    assert by_subset.setdefault(tuple(selected), row["pred_choice"]) == row["pred_choice"]
                checkpoints[key] = row
            if not history:
                assert len({r["pred_choice"] for r in rows}) == 1
            receipts[sid, qi] = prep
    assert len(receipts) == 40 and len(checkpoints) == 240 and replays == 80
    assert sum(n.startswith("checkpoints/") for n in payload) == 40
    answers = table(payload["exports/answers.csv"])
    keys = [(int(r["sample_id"]), int(r["qa_index"]), r["method"]) for r in answers]
    assert len(keys) == len(set(keys)) and set(keys) == checkpoints.keys()
    compare_csv([checkpoints[k] for k in keys], answers)

    summary, paired, changes, proxies = [], [], [], []
    for arm in ARMS:
        rows = [checkpoints[sid, qi, arm] for sid, qi in receipts]
        summary.append(dict(arm=arm, answered=len(rows), correct=sum(r["correct"] for r in rows),
                            median_query_kv_bytes=statistics.median(r["query_kv_bytes"] for r in rows),
                            median_score_and_construction_s=statistics.median(r["score_s"] + r["construction_s"] for r in rows)))
        if arm == ARMS[0]:
            continue
        pairs = [(checkpoints[r["sample_id"], r["qa_index"], ARMS[0]], r) for r in rows]
        paired.append(dict(candidate=ARMS[0], baseline=arm, questions=len(rows),
                           candidate_only_correct=sum(a["correct"] > b["correct"] for a, b in pairs),
                           baseline_only_correct=sum(a["correct"] < b["correct"] for a, b in pairs),
                           choice_changes=sum(a["pred_choice"] != b["pred_choice"] for a, b in pairs),
                           median_score_and_construction_s_saved=statistics.median(
                               b["score_s"] + b["construction_s"] - a["score_s"] - a["construction_s"] for a, b in pairs)))
        changes.extend(dict(sample_id=a["sample_id"], qa_index=a["qa_index"], baseline=arm,
                            selected_choice=a["pred_choice"], baseline_choice=b["pred_choice"],
                            selected_correct=a["correct"], baseline_correct=b["correct"],
                            selected_history=a["selected_history"], baseline_history=b["selected_history"])
                       for a, b in pairs if a["pred_choice"] != b["pred_choice"])
    for (sid, qi), prep in receipts.items():
        if len(prep["history_indices"]) > 2:
            score = prep["score_trace"]
            a, b, c = (checkpoints[sid, qi, arm]["selected_history"] for arm in ARMS[:3])
            proxies.append(dict(sample_id=sid, qa_index=qi, selected_residual=score["selected_residual_mse"][-1],
                                latest_residual=score["latest_residual_mse"], mass_residual=score["mass_residual_mse"],
                                different_from_latest=a != b, different_from_mass=a != c))
    compare_csv(summary, table(payload["exports/summary.csv"]))
    compare_csv(paired, table(payload["exports/paired.csv"]))
    tables.mkdir(parents=True, exist_ok=True)
    for name, records in (("summary", summary), ("paired", paired), ("choice_changes", changes), ("proxy_diagnostics", proxies)):
        write_csv(tables / f"{name}.csv", records)
    recency_pairs = [(checkpoints[sid, qi, "latest_two"], checkpoints[sid, qi, "all_history"]) for sid, qi in receipts]
    latest_vs_all = dict(questions=len(recency_pairs),
                         latest_only_correct=sum(a["correct"] > b["correct"] for a, b in recency_pairs),
                         all_only_correct=sum(a["correct"] < b["correct"] for a, b in recency_pairs),
                         median_construction_s_saved=statistics.median(b["construction_s"] - a["construction_s"]
                                                                       for a, b in recency_pairs))
    result = dict(status="pass", archive_sha256=ARCHIVE_SHA, source_notebook_sha256=NOTEBOOK_SHA,
                  source_notebook_has_execution_outputs=False, payloads=len(payload), questions=40, answers=240,
                  exact_replays=replays, elapsed_post_model_s=completion["elapsed_post_model_s"],
                  median_selector_s=statistics.median(r["score_s"] for r in receipts.values()),
                  selection_possible_questions=len(proxies),
                  different_from_latest=sum(r["different_from_latest"] for r in proxies),
                  different_from_mass=sum(r["different_from_mass"] for r in proxies),
                  lower_proxy_error_than_mass=sum(r["mass_residual"] - r["selected_residual"] > 1e-12 for r in proxies),
                  lower_proxy_error_than_latest=sum(r["latest_residual"] - r["selected_residual"] > 1e-12 for r in proxies),
                  summary=summary, paired=paired, latest_vs_all=latest_vs_all)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(ROOT / f"results/construction_selection_{DATE}",
                           ROOT / f"evidence/audits/construction_selection_{DATE}_audit.json"), indent=2))
