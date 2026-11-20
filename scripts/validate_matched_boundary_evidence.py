"""Verify the completed matched-boundary run without executing uploaded code."""
import ast
import json
import math
from collections import Counter
from zipfile import ZipFile

from make_matched_boundary_notebook import ROOT, reference_data
from validate_causal_pilot_evidence import check_manifest, check_prediction, sha, table, write_csv
from validate_context_diagnostic_evidence import expected_groups
from validate_input_diagnostic_evidence import compare_csv
from validate_stage_attribution_evidence import ARMS, CONTRASTS, check_event, fingerprint

DATE = "2026-11-21"
ARCHIVE = ROOT / "evidence/archives/HERMES_matched_boundary_evidence.zip"
NOTEBOOK = ROOT / f"notebooks/executed/{DATE}_hermes-matched-boundary.ipynb"
ARCHIVE_SHA = "cef5ea2d59b6fdf212a27cf13de470e56c936054c45387e0a7d27fecccac8a59"
NOTEBOOK_SHA = "3e97795ecacb6b2a8296b99102efced0511e8f0a7ace07ece155ed413e9b6030"
SOURCE_FILES = {
    "stage_runner.py": "RUNNER_SOURCE",
    "provenance/stage_base.py": "STAGE_BASE_SOURCE",
    "stage_context_utils.py": "CONTEXT_SOURCE",
    "context_input_utils.py": "INPUT_SOURCE",
    "diagnostic_pilot_utils.py": "UTILS_SOURCE",
}
PAIRS = (("primary_C_matched_vs_D", 2, 3, False), ("A_matched_vs_B", 0, 1, False),
         ("C_matched_vs_original_C", 2, 2, True), ("A_matched_vs_original_A", 0, 0, True),
         ("C_matched_vs_compact_recent4", 2, 4, False), ("C_matched_vs_A_matched_readout", 2, 0, False))


def audit(tables, output):
    assert sha(ARCHIVE.read_bytes()) == ARCHIVE_SHA and sha(NOTEBOOK.read_bytes()) == NOTEBOOK_SHA
    with ZipFile(ARCHIVE) as zipped:
        manifest, payload = check_manifest(zipped)

    def read(name):
        return json.loads(payload[name])

    ref = read("reference.json")
    assert ref == reference_data()
    config, complete = read("exports/run_config.json"), read("exports/completion.json")
    assert config["reference_sha256"] == sha(payload["reference.json"])
    assert config["previous_stage_archive_sha256"] == ref["stage_archive_sha256"]
    assert config["sources"].keys() == SOURCE_FILES.keys()
    assert all(sha(payload[name]) == config["sources"][name] for name in SOURCE_FILES)
    assert read("launch.json")["runner_sha256"] == config["sources"]["stage_runner.py"]
    assert fingerprint({k: v for k, v in config.items() if k != "config_hash"}) == config["config_hash"] == complete["config_hash"]
    assert manifest["status"] == complete["status"] == "complete"
    assert (complete["main_questions"], complete["main_answers"], complete["exact_replays"]) == (40, 200, 120)
    assert config["arms"] == list(ARMS) and config["exact_replay_arms"] == [ARMS[i] for i in (1, 3, 4)]
    assert (config["main_window"], config["recent_window"]) == (20, 4)

    notebook = json.loads(NOTEBOOK.read_bytes())
    sources = {c["id"]: c["source"] for c in notebook["cells"]}
    embedded = {}
    for node in ast.parse("".join(sources["matched-08"])).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name in (*SOURCE_FILES.values(), "REFERENCE_JSON"):
            assert name not in embedded
            embedded[name] = ast.literal_eval(node.value).encode()
    assert all(payload[name] == embedded[variable] for name, variable in SOURCE_FILES.items())
    assert embedded["REFERENCE_JSON"] == payload["reference.json"]
    outputs = [o for c in notebook["cells"] for o in c.get("outputs", [])]
    assert not any(o["output_type"] == "error" for o in outputs)
    streams = "\n".join("".join(o["text"]) for o in outputs if o["output_type"] == "stream")
    assert f"SHA256={ARCHIVE_SHA}" in streams and "MATCHED_BOUNDARY_COMPLETE" in streams
    log = payload["captured_logs/matched.log"].decode("utf-8")
    assert log.count("STAGE_DONE ") == 200 and log.count("QUESTION_BLOCK_DONE ") == 40
    assert "MATCHED_PREFLIGHT_PASS" in log and "MATCHED_BOUNDARY_COMPLETE" in log

    preflight = read("exports/matched_preflight.json")
    assert preflight["status"] == "pass" and preflight["fp32"]["result"]["status"] == "pass"
    assert preflight["policy"] == "fp32_physical_reference_and_fp16_same_shape_invariance"
    assert preflight["fp16_masked_readout_exact"] and preflight["fp16_hidden_and_future_attention_zero"]
    assert preflight["fp32"]["result"]["all_layer_hidden_history_attention_zero"]
    assert preflight["fp32"]["result"]["all_layer_future_attention_zero"]
    assert len(preflight["fp32"]["construction_by_layer"]) == len(preflight["fp16"]["construction"]) == 48
    assert all(r["exceeds_active_bound"] == 0 for r in preflight["fp32"]["construction_by_layer"])
    assert all(r["max_abs"] == 0 for r in preflight["fp16"]["construction"])
    logits = preflight["fp16"]["logits"]
    assert len(logits) == 9 and all(r["argmax_equal"] for r in logits)

    videos = {v["sample_id"]: v for v in read("exports/replay_annotations.json")}
    assert videos.keys() == {v["sample_id"] for v in ref["videos"]}
    for v in ref["videos"]:
        assert all(videos[v["sample_id"]][k] == v[k] for k in ("conversations", "video_sha256", "video_id"))
    old = {(r["sample_id"], r["qa_index"], r["method"]): r for r in ref["stage_rows"]}
    features = {sid: read(f"provenance/features_{sid}.json") for sid in videos}
    by_key, questions, replays = {}, [], 0
    for sid, video in sorted(videos.items()):
        feature = features[sid]
        assert feature["feature_fingerprint"] == fingerprint(feature["metadata"])
        assert feature["timestamp_rows"] == video["timestamps"]
        for qi, qa in enumerate(video["conversations"]):
            end = math.ceil(qa["end_time"] * .5)
            frames = list(range(max(0, end - 20), end))
            history, recent = frames[:-4], frames[-4:]
            assert all(float(video["timestamps"][i]["start_s"]) <= qa["end_time"] + 1e-6 for i in frames)
            prep = read(f"preparation/main/sample_{sid}_q{qi}.json")
            assert (prep["history_indices"], prep["recent_indices"]) == (history, recent)
            stop = 13 + 196 * len(history)
            assert prep["history_stop"] == stop and prep["prompt_tokens"] == 13
            groups = expected_groups(recent, "original", feature["metadata"])
            assert prep["visible_recent_groups"] == prep["hidden_recent_groups"] == prep["compact_groups"] == groups
            assert prep["original_groups"] == expected_groups(frames, "original", feature["metadata"])
            visible, hidden = prep["visible_construction_mask_trace"], prep["hidden_construction_mask_trace"]
            assert len(visible) == len(hidden) == (len(groups) if history else 0)
            past = stop
            for a, b, group in zip(visible, hidden, groups):
                query = 196 * len(group)
                check_event(a, query, past, 13)
                check_event(b, query, past, stop)
                assert a == dict(b, blocked_stop=13)
                past += query
            questions.append((sid, qi))
            for i, arm in enumerate(ARMS):
                value = read(f"checkpoints/main/{arm}/sample_{sid}_q{qi}.json")
                row = value["row"]
                assert fingerprint(row) == value["row_sha256"]
                assert (row["scope"], row["sample_id"], row["qa_index"], row["method"]) == ("main", sid, qi, arm)
                assert (row["history_indices"], row["recent_indices"]) == (history, recent)
                assert row["feature_fingerprint"] == feature["feature_fingerprint"]
                identity = {k: row[k] for k in ("sample_id", "video_id", "qa_index", "method", "question", "choices",
                                                "gold_answer", "gold_choice", "query_time_s", "task")}
                assert value["identity"] == fingerprint(dict(config=config["config_hash"], scope="main", question=identity,
                    history=history, recent=recent, features=feature["feature_fingerprint"]))
                check_prediction(row, video, qi)
                assert row["status"] == "answered" and row["skip_reason"] == ""
                assert (row["independent_construction"], row["history_readout_hidden"]) == (i in (1, 3), i in (2, 3))
                assert (row["history_empty"], row["factor_cell"], row["history_physically_present"]) == (not history, i != 4, bool(history) and i != 4)
                actual = recent if i == 4 else frames
                past = 13 + len(actual) * 196
                assert row["actual_visual_indices"] == actual and row["cache_lengths"] == [past] * 24
                assert row["query_logical_start"] == past
                assert row["recent_logical_start"] == row["recent_physical_start"] == (13 if i == 4 else stop)
                events = row["mask_trace"]
                if history and i in (2, 3):
                    assert 1 < len(events) <= 17 and events[0]["query_tokens"] > 1
                    for step, event in enumerate(events):
                        query = events[0]["query_tokens"] if not step else 1
                        check_event(event, query, past, stop)
                        past += query
                else:
                    assert not events
                key = (sid, qi, arm)
                if i in (1, 3, 4):
                    assert row["reference_choice_matches"] is True
                    assert row["reference_choice"] == row["pred_choice"] == old[key]["pred_choice"]
                    replays += 1
                else:
                    assert row["reference_choice_matches"] == row["reference_choice"] == ""
                by_key[key] = row
            if not history:
                assert len({by_key[sid, qi, arm]["pred_choice"] for arm in ARMS}) == 1
    assert len(questions) == 40 and len(by_key) == len(old) == 200 and replays == 120
    assert sum(n.startswith("checkpoints/") for n in payload) == 200
    predictions = table(payload["exports/predictions.csv"])
    keys = [(int(r["sample_id"]), int(r["qa_index"]), r["method"]) for r in predictions]
    assert len(keys) == len(set(keys)) and set(keys) == by_key.keys()
    compare_csv([by_key[k] for k in keys], predictions)

    def compare(a, b, historical=False):
        other = old if historical else by_key
        return [(by_key[sid, qi, ARMS[a]], other[sid, qi, ARMS[b]]) for sid, qi in questions]

    def stats(pairs):
        return (sum(a["correct"] > b["correct"] for a, b in pairs),
                sum(a["correct"] < b["correct"] for a, b in pairs),
                sum(a["pred_choice"] != b["pred_choice"] for a, b in pairs))

    summary, paired, contrasts, details = [], [], [], []
    for arm in ARMS:
        subset = [by_key[sid, qi, arm] for sid, qi in questions]
        summary.append(dict(scope="main", arm=arm, recorded=40, answered=40,
                            correct=sum(r["correct"] for r in subset),
                            history_empty_questions=sum(r["history_empty"] for r in subset)))
    for a, b in CONTRASTS:
        pairs = compare(a, b)
        wins, losses, changes = stats(pairs)
        paired.append(dict(scope="main", arm_a=ARMS[a], arm_b=ARMS[b], questions=len(pairs),
                           a_only_correct=wins, b_only_correct=losses, choice_changes=changes,
                           history_empty_pairs=sum(x["history_empty"] for x, _ in pairs)))
    for name, a, b, historical in PAIRS:
        pairs = compare(a, b, historical)
        wins, losses, changes = stats(pairs)
        contrasts.append(dict(contrast=name, questions=len(pairs), left_only_correct=wins,
                              right_only_correct=losses, choice_changes=changes))
        details.extend(dict(contrast=name, sample_id=x["sample_id"], qa_index=x["qa_index"], history_empty=x["history_empty"],
                            left_choice=x["pred_choice"], right_choice=y["pred_choice"],
                            left_correct=x["correct"], right_correct=y["correct"]) for x, y in pairs)
    for name, rows in (("summary", summary), ("paired", paired), ("matched_pairs", contrasts), ("matched_question_pairs", details)):
        compare_csv(rows, table(payload[f"exports/{name}.csv"]))

    tables.mkdir(parents=True, exist_ok=True)
    write_csv(tables / "summary.csv", summary)
    write_csv(tables / "matched_pairs.csv", contrasts)
    write_csv(tables / "matched_question_pairs.csv", details)
    write_csv(tables / "per_video.csv", [
        dict(sample_id=sid, **{arm: sum(by_key[sid, qi, arm]["correct"] for qi in range(len(video["conversations"])))
                               for arm in ARMS}) for sid, video in sorted(videos.items())
    ])
    primary_changes = [dict(sample_id=x["sample_id"], qa_index=x["qa_index"], question=x["question"],
                            gold=x["gold_answer"], c_answer=x["pred_choice"], d_answer=y["pred_choice"])
                       for x, y in compare(2, 3) if x["pred_choice"] != y["pred_choice"]]
    result = dict(status="pass", archive_sha256=ARCHIVE_SHA, notebook_sha256=NOTEBOOK_SHA, payloads=len(payload),
                  questions=len(questions), checkpoints=len(by_key), exact_replays=replays,
                  all_arm_historical_choice_changes=sum(by_key[k]["pred_choice"] != old[k]["pred_choice"] for k in by_key),
                  elapsed_post_model_s=complete["elapsed_post_model_s"],
                  fp16_logit_top_token_checks=len(logits), fp16_logit_exceedances=sum(r["exceeds_active_bound"] for r in logits),
                  feature_origins=dict(Counter(f["origin"] for f in features.values())),
                  summary=summary, contrasts=contrasts, primary_changes=primary_changes)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(ROOT / f"results/matched_boundary_{DATE}",
                           ROOT / f"evidence/audits/matched_boundary_{DATE}_audit.json"), indent=2))
