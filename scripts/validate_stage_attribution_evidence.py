"""Audit the saved stage run with the standard library, without executing uploaded code."""
import json
import math
from collections import Counter
from zipfile import ZipFile

from make_stage_attribution_notebook import reference_data
from validate_causal_pilot_evidence import ROOT, check_manifest, check_prediction, sha, stable, table, write_csv
from validate_input_diagnostic_evidence import compare_csv

DATE = "2026-11-21"
ARCHIVE = ROOT / "evidence/archives/HERMES_stage_attribution_evidence.zip"
NOTEBOOK = ROOT / f"notebooks/executed/{DATE}_hermes-stage-attribution.ipynb"
ARCHIVE_SHA = "5719f961f5d01d07b5c45b0035f6c33bd33f4e2d91319b8cce652967c12e4acd"
NOTEBOOK_SHA = "2d6dddaf3cd9ccd25646f532de03377464c41e6b55cc04f541a4bc11c3191f2b"
ARMS = ("A_history_both", "B_independent_recent", "C_hide_history_readout",
        "D_independent_hide_readout", "compact_recent4")
CONTRASTS = ((0, 1), (0, 2), (1, 3), (2, 3), (3, 4))


def fingerprint(value):
    return sha(stable(value).encode())


def check_event(event, query, past, stop):
    assert event["blocked_start"] == 13 and event["blocked_stop"] == stop
    assert event["past_tokens"] == past and event["query_tokens"] == query
    assert event["shape"] == [1, 1, query, past + query]


def audit(archive, notebook, tables, output):
    assert sha(archive.read_bytes()) == ARCHIVE_SHA
    assert sha(notebook.read_bytes()) == NOTEBOOK_SHA
    with ZipFile(archive) as zipped:
        manifest, data = check_manifest(zipped)

    def record(name):
        return json.loads(data[name])

    reference = record("reference.json")
    assert reference == reference_data()
    config = record("exports/run_config.json")
    completion = record("exports/completion.json")
    assert config["reference_sha256"] == sha(data["reference.json"])
    assert all(sha(data[name]) == digest for name, digest in config["sources"].items())
    assert fingerprint({k: v for k, v in config.items() if k != "config_hash"}) == config["config_hash"] == completion["config_hash"]
    assert manifest["status"] == completion["status"] == "complete"
    assert (completion["main_questions"], completion["main_answers"], completion["rows"]) == (40, 200, 204)
    assert config["long_case"] == 1 and completion["long_case_enabled"] and completion["long_case_done"]
    assert config["arms"] == list(ARMS) and (config["main_window"], config["recent_window"]) == (20, 4)

    log = data["captured_logs/stage.log"].decode()
    assert log.count("STAGE_DONE ") == 204 and log.count("QUESTION_BLOCK_DONE ") == 40
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    streams = "\n".join("".join(o["text"]) for c in nb["cells"] for o in c.get("outputs", []) if o["output_type"] == "stream")
    assert f"SHA256={ARCHIVE_SHA}" in streams and "STAGE_ATTRIBUTION_COMPLETE" in streams
    assert "Status: complete | Main answers: 200" in streams

    preflight = record("stage_preflight.json")
    diagnostics = record("exports/stage_preflight_diagnostics.json")
    fp32, fp16 = diagnostics["fp32"], diagnostics["production"]
    assert preflight == fp16["result"] and preflight["fp32_structural_reference"] == fp32["result"]
    assert preflight["validation_policy"] == "fp32_physical_reference_and_fp16_same_shape_invariance"
    for result in (preflight, fp32["result"]):
        assert result["status"] == "pass" and result["layers"] == 24 and result["attention_checks_steps"] == 3
        assert result["all_layer_hidden_history_attention_zero"] and result["all_layer_future_attention_zero"]
        assert result["source_snapshots_unchanged"] and all(c["argmax_equal"] for c in result["comparisons"])
    assert fp32["dtype"] == "torch.float32" and fp16["dtype"] == "torch.float16"
    assert len(fp32["construction_by_layer"]) == len(fp16["construction_by_layer"]) == 48
    assert all(r["exceeds_active_bound"] == 0 for r in fp32["construction_by_layer"])
    assert all(r["max_abs"] == 0 for r in fp16["construction_by_layer"])
    logits = fp16["fp16_logit_comparisons"]
    assert len(logits) == 9 and all(r["argmax_equal"] for r in logits)

    prior = {(r["sample_id"], r["qa_index"], r["method"]): r["pred_choice"] for r in reference["input_rows"]}
    videos = {v["sample_id"]: v for v in record("exports/replay_annotations.json")}
    assert videos.keys() == {v["sample_id"] for v in reference["videos"]}
    for v in reference["videos"]:
        assert all(videos[v["sample_id"]][k] == v[k] for k in ("conversations", "video_sha256", "video_id"))

    blocks = [("main", sid, qi) for sid in sorted(videos) for qi in range(5)] + [("long_204_q3", 204, 3)]
    rows, casebook = {}, []
    replay_count = 0
    for scope, sid, qi in blocks:
        video = videos[sid]
        end = math.ceil(video["conversations"][qi]["end_time"] * .5)
        frames = list(range(max(0, end - 20) if scope == "main" else 0, end))
        history, recent = frames[:-4], frames[-4:]
        assert all(float(video["timestamps"][i]["start_s"]) <= video["conversations"][qi]["end_time"] + 1e-6 for i in frames)
        prep_path = f"preparation/main/sample_{sid}_q{qi}.json" if scope == "main" else "preparation/long_204_q3.json"
        prep = record(prep_path)
        assert prep["history_indices"] == history and prep["recent_indices"] == recent
        stop = 13 + 196 * len(history)
        assert prep["history_stop"] == stop and prep["prompt_tokens"] == 13
        assert sum(prep["full_groups"], []) == frames and sum(prep["compact_groups"], []) == recent
        encoding_trace = prep["independent_encoding_mask_trace"]
        if history:
            assert sum(prep["independent_recent_groups"], []) == recent
            assert len(encoding_trace) == len(prep["independent_recent_groups"])
            past = stop
            for group, event in zip(prep["independent_recent_groups"], encoding_trace):
                check_event(event, len(group) * 196, past, stop)
                past += len(group) * 196
        else:
            assert not encoding_trace
        feature = record(f"provenance/features_{sid}.json")
        assert feature["feature_fingerprint"] == fingerprint(feature["metadata"])
        assert feature["timestamp_rows"] == video["timestamps"]
        h, r = set(history), set(recent)
        case = dict(scope=scope, sample_id=sid, qa_index=qi, history_frames=len(h), recent_frames=len(r),
                    original_group_crosses_boundary=any(h.intersection(g) and r.intersection(g) for g in prep["full_groups"]))
        for i, arm in enumerate(ARMS if scope == "main" else ARMS[:4]):
            name = f"checkpoints/{scope}/{arm}/sample_{sid}_q{qi}.json"
            checkpoint = record(name)
            row = checkpoint["row"]
            assert fingerprint(row) == checkpoint["row_sha256"]
            assert (row["scope"], row["sample_id"], row["qa_index"], row["method"]) == (scope, sid, qi, arm)
            assert (row["history_indices"], row["recent_indices"]) == (history, recent)
            assert row["feature_fingerprint"] == feature["feature_fingerprint"]
            identity = {k: row[k] for k in ("sample_id", "video_id", "qa_index", "method", "question", "choices",
                                            "gold_answer", "gold_choice", "query_time_s", "task")}
            assert checkpoint["identity"] == fingerprint(dict(config=config["config_hash"], scope=scope, question=identity,
                history=history, recent=recent, features=feature["feature_fingerprint"]))
            check_prediction(row, video, qi)
            assert row["status"] == "answered" and row["skip_reason"] == ""
            assert (row["independent_construction"], row["history_readout_hidden"]) == (i in (1, 3), i in (2, 3))
            assert (row["history_empty"], row["factor_cell"], row["history_physically_present"]) == (not history, i != 4, bool(history) and i != 4)
            actual = recent if i == 4 else frames
            past = 13 + len(actual) * 196
            assert row["actual_visual_indices"] == actual and row["cache_lengths"] == [past] * 24
            assert row["query_logical_start"] == past and row["recent_logical_start"] == row["recent_physical_start"] == (13 if i == 4 else stop)
            events = row["mask_trace"]
            if history and i in (2, 3):
                assert 1 < len(events) <= 17 and events[0]["query_tokens"] > 1
                for step, event in enumerate(events):
                    query = events[0]["query_tokens"] if step == 0 else 1
                    check_event(event, query, past, stop)
                    past += query
            else:
                assert not events
            expected = prior[sid, qi, "recent_20" if i == 0 else "recent_4"] if scope == "main" and i in (0, 4) else (
                reference["long_reference_choice"] if scope != "main" and i == 0 else "")
            assert row["reference_choice"] == expected
            if expected:
                assert row["reference_choice_matches"] is True and row["pred_choice"] == expected
                replay_count += 1
            else:
                assert row["reference_choice_matches"] == ""
            rows[scope, sid, qi, arm] = row
            case.update({arm + "_choice": row["pred_choice"], arm + "_correct": row["correct"]})
        casebook.append(case)
    assert len(rows) == sum(n.startswith("checkpoints/") for n in data) and replay_count == 81

    exported = table(data["exports/predictions.csv"])
    keys = [(r["scope"], int(r["sample_id"]), int(r["qa_index"]), r["method"]) for r in exported]
    assert len(keys) == len(set(keys)) and set(keys) == rows.keys()
    compare_csv([rows[k] for k in keys], exported)
    scopes = ("main", "long_204_q3")
    summary, pairs, per_video = [], [], []
    for scope in scopes:
        for arm in ARMS if scope == "main" else ARMS[:4]:
            subset = [v for k, v in rows.items() if k[0] == scope and k[3] == arm]
            summary.append(dict(scope=scope, arm=arm, recorded=len(subset), answered=len(subset),
                                correct=sum(v["correct"] for v in subset), history_empty_questions=sum(v["history_empty"] for v in subset)))
    for a, b in CONTRASTS:
        for scope in scopes if b != 4 else ("main",):
            matched = [(rows[s, sid, qi, ARMS[a]], rows[s, sid, qi, ARMS[b]]) for s, sid, qi in blocks if s == scope]
            pairs.append(dict(scope=scope, arm_a=ARMS[a], arm_b=ARMS[b], questions=len(matched),
                              a_only_correct=sum(x["correct"] > y["correct"] for x, y in matched),
                              b_only_correct=sum(y["correct"] > x["correct"] for x, y in matched),
                              choice_changes=sum(x["pred_choice"] != y["pred_choice"] for x, y in matched),
                              history_empty_pairs=sum(x["history_empty"] for x, _ in matched)))
    compare_csv(summary, table(data["exports/summary.csv"]))
    compare_csv(pairs, table(data["exports/paired.csv"]))
    for sid in sorted(videos):
        per_video.append(dict(sample_id=sid, **{arm: sum(rows["main", sid, qi, arm]["correct"] for qi in range(5)) for arm in ARMS}))

    tables.mkdir(parents=True, exist_ok=True)
    write_csv(tables / "summary.csv", summary)
    write_csv(tables / "paired.csv", pairs)
    write_csv(tables / "question_casebook.csv", casebook[:40])
    write_csv(tables / "long_case.csv", casebook[40:])
    write_csv(tables / "per_video.csv", per_video)
    result = dict(status="pass", archive_sha256=ARCHIVE_SHA, notebook_sha256=NOTEBOOK_SHA, payloads=len(data),
                  checkpoints=len(rows), exact_replays=replay_count, main_questions=40, main_answers=200, long_answers=4,
                  elapsed_post_model_s=completion["elapsed_post_model_s"], fp16_logit_top_token_checks=len(logits),
                  fp16_logit_exceedances=sum(r["exceeds_active_bound"] for r in logits),
                  main_groups_crossing_factor_boundary=sum(c["original_group_crosses_boundary"] for c in casebook[:40]),
                  summary=summary, paired=pairs, feature_origins=dict(Counter(record(f"provenance/features_{sid}.json")["origin"] for sid in videos)))
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = audit(ARCHIVE, NOTEBOOK, ROOT / f"results/stage_attribution_{DATE}",
                   ROOT / f"evidence/audits/stage_attribution_{DATE}_audit.json")
    print(json.dumps(result, indent=2))
