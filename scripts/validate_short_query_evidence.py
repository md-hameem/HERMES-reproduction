"""Audit short query-cache evidence as data and reconstruct paired outcomes."""
import ast
import json
import math
import random
import statistics
from zipfile import ZipFile

from make_short_query_notebook import ROOT, reference_data
from validate_causal_pilot_evidence import check_manifest, check_prediction, sha, table, write_csv
from validate_context_diagnostic_evidence import expected_groups
from validate_input_diagnostic_evidence import compare_csv
from validate_stage_attribution_evidence import check_event, fingerprint

DATE = "2026-11-21"
ARCHIVE = ROOT / "evidence/archives/HERMES_short_query_evidence.zip"
NOTEBOOK = ROOT / f"notebooks/executed/{DATE}_hermes-short-query-cache.ipynb"
ARCHIVE_SHA = "1e06bcb5ee4ef61348f23b9c23d4795676c3fe6fdd98aa5531e8215f9a9c54ce"
NOTEBOOK_SHA = "c422b6d286393c22f49533c0a6ff3a4ca4f9a4d537534be9cb18677c6700c79e"
ARMS = ("full_contextual", "masked_contextual", "short_contextual", "compact_recent4")
SOURCES = dict(zip(
    ("stage_runner.py", "provenance/stage_base.py", "provenance/matched_base.py",
     "stage_context_utils.py", "context_input_utils.py", "diagnostic_pilot_utils.py"),
    ("RUNNER_SOURCE", "STAGE_BASE_SOURCE", "MATCHED_BASE_SOURCE", "CONTEXT_SOURCE", "INPUT_SOURCE", "UTILS_SOURCE"),
))


def audit(tables, audit_path):
    assert sha(ARCHIVE.read_bytes()) == ARCHIVE_SHA
    assert sha(NOTEBOOK.read_bytes()) == NOTEBOOK_SHA
    with ZipFile(ARCHIVE) as archive:
        manifest, payload = check_manifest(archive)

    def read(name):
        return json.loads(payload[name])

    config, completion, reference = (read(n) for n in ("exports/run_config.json", "exports/completion.json", "reference.json"))
    assert reference == reference_data()
    assert config["previous_archive_sha256"] == reference["matched_archive_sha256"]
    assert manifest["status"] == completion["status"] == "complete"
    assert config["reference_sha256"] == sha(payload["reference.json"])
    assert config["sources"].keys() == SOURCES.keys()
    assert all(sha(payload[n]) == config["sources"][n] for n in SOURCES)
    assert config["sources"]["stage_runner.py"] == read("launch.json")["runner_sha256"]
    assert fingerprint({k: v for k, v in config.items() if k != "config_hash"}) == config["config_hash"] == completion["config_hash"]
    assert config["repeats"] == 2 and config["arms"] == list(ARMS)
    assert (completion["main_questions"], completion["main_answers"], completion["timed_calls"]) == (40, 160, 320)

    notebook = json.loads(NOTEBOOK.read_bytes())
    embedded_cell = next(c for c in notebook["cells"] if c["id"] == "short-query-08")
    embedded = {}
    for node in ast.parse("".join(embedded_cell["source"])).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in (*SOURCES.values(), "REFERENCE_JSON"):
                assert name not in embedded
                embedded[name] = ast.literal_eval(node.value).encode()
    assert all(embedded[var] == payload[n] for n, var in SOURCES.items())
    assert embedded["REFERENCE_JSON"] == payload["reference.json"]
    outputs = [o for c in notebook["cells"] for o in c.get("outputs", [])]
    assert not any(o["output_type"] == "error" for o in outputs)
    text = "\n".join("".join(o["text"]) for o in outputs if o["output_type"] == "stream")
    assert f"SHA256={ARCHIVE_SHA}" in text and "SHORT_QUERY_COMPLETE" in text
    log = payload["captured_logs/short_query.log"].decode()
    assert "SHORT_QUERY_PREFLIGHT_PASS" in log and log.count("QUESTION_BLOCK_DONE ") == 40
    preflight = read("exports/short_preflight.json")
    assert read("exports/matched_preflight.json")["status"] == "pass"
    for dtype in ("torch.float32", "torch.float16"):
        assert [r["step"] for r in preflight[dtype]] == [0, 1, 2]
        assert all(r["argmax_equal"] for r in preflight[dtype])
    assert all(r["exceeds_active_bound"] == 0 for r in preflight["torch.float32"])

    old = {(r["sample_id"], r["qa_index"], r["arm"]): r["choice"] for r in reference["baseline_choices"]}
    videos = {v["sample_id"]: v for v in read("exports/replay_annotations.json")}
    assert videos.keys() == {v["sample_id"] for v in reference["videos"]}
    for v in reference["videos"]:
        assert all(videos[v["sample_id"]][k] == v[k] for k in ("video_id", "conversations", "video_sha256"))
    rows, questions = {}, {}
    replays = 0
    for sid, video in sorted(videos.items()):
        feature = read(f"provenance/features_{sid}.json")
        assert feature["feature_fingerprint"] == fingerprint(feature["metadata"])
        assert feature["timestamp_rows"] == video["timestamps"] and feature["origin"] == "verified_historical_cache"
        for qi, qa in enumerate(video["conversations"]):
            name = f"sample_{sid}_q{qi}.json"
            prep, checkpoint = read(f"preparation/{name}"), read(f"checkpoints/{name}")
            end = math.ceil(qa["end_time"] * .5)
            frames = list(range(max(0, end - 20), end))
            history, recent = frames[:-4], frames[-4:]
            assert prep["history_indices"] == history and prep["recent_indices"] == recent
            assert all(float(video["timestamps"][i]["start_s"]) <= qa["end_time"] + 1e-6 for i in frames)
            groups = expected_groups(recent, "original", feature["metadata"])
            assert prep["visible_recent_groups"] == prep["hidden_recent_groups"] == prep["compact_groups"] == groups
            stop = 13 + 196 * len(history)
            assert prep["prompt_tokens"] == 13 and prep["history_stop"] == stop
            visible, hidden = prep["visible_construction_mask_trace"], prep["hidden_construction_mask_trace"]
            assert len(visible) == len(hidden) == (len(groups) if history else 0)
            past = stop
            for a, b, group in zip(visible, hidden, groups):
                query = 196 * len(group)
                check_event(a, query, past, 13)
                check_event(b, query, past, stop)
                assert a == dict(b, blocked_stop=13)
                past += query
            assert fingerprint(checkpoint["rows"]) == checkpoint["rows_sha256"]
            assert checkpoint["key"] == fingerprint(dict(config=config["config_hash"], feature=feature["feature_fingerprint"],
                                                          sample=sid, qa=qi, h=history, r=recent))
            assert [r["method"] for r in checkpoint["rows"]] == list(ARMS)
            rng = random.Random(config["seed"] + sid * 31 + qi)
            orders = []
            for repeat in range(-int((sid, qi) == (204, 0)), 2):
                order = list(ARMS)
                rng.shuffle(order)
                if repeat >= 0:
                    orders.append(order)
            assert prep["timing_orders"] == orders
            questions[sid, qi] = prep
            for i, row in enumerate(checkpoint["rows"]):
                check_prediction(dict(row, answer_s=row["answer_median_s"]), video, qi)
                assert row["sample_id"] == sid and row["feature_fingerprint"] == feature["feature_fingerprint"]
                physical = 13 + 196 * (len(recent) if i >= 2 else len(frames))
                logical = 13 + 196 * (len(recent) if i == 3 else len(frames))
                assert (row["physical_tokens"], row["query_logical_start"]) == (physical, logical)
                assert row["kv_bytes"] == physical * 12288
                assert row["choices_by_repeat"] == [row["pred_choice"]] * 2
                assert len(row["answer_seconds"]) == len(row["traces"]) == 2
                assert all(math.isfinite(t) and t > 0 for t in row["answer_seconds"])
                assert row["answer_median_s"] == statistics.median(row["answer_seconds"])
                for trace in row["traces"]:
                    past, pos = physical, logical
                    assert 1 <= len(trace) <= 17
                    for j, event in enumerate(trace):
                        query = event["query_tokens"]
                        assert query > 1 if j == 0 else query == 1
                        assert (event["past_tokens"], event["logical_start"]) == (past, pos)
                        past, pos = past + query, pos + query
                key = (sid, qi, row["method"])
                expected = old.get(key)
                assert row["historical_choice"] == expected
                if expected is None:
                    assert row["historical_matches"] is None
                else:
                    assert row["historical_matches"] is True and row["pred_choice"] == expected
                    replays += 1
                rows[key] = row
    assert len(questions) == sum(n.startswith("checkpoints/") for n in payload) == 40
    assert len(rows) == 160 and replays == 120
    answers = table(payload["exports/answers.csv"])
    keys = [(int(r["sample_id"]), int(r["qa_index"]), r["method"]) for r in answers]
    assert len(keys) == len(set(keys)) and set(keys) == rows.keys()
    compare_csv([rows[k] for k in keys], answers)

    summary, comparisons, details = [], [], []
    for arm in ARMS:
        subset = [rows[sid, qi, arm] for sid, qi in questions]
        summary.append(dict(arm=arm, answered=len(subset), correct=sum(x["correct"] for x in subset),
                            median_kv_bytes=statistics.median(x["kv_bytes"] for x in subset),
                            median_answer_s=statistics.median(x["answer_median_s"] for x in subset)))
    for baseline in (ARMS[1], ARMS[3], ARMS[0]):
        subset = []
        for (sid, qi), prep in questions.items():
            left, right = rows[sid, qi, ARMS[2]], rows[sid, qi, baseline]
            saved = right["answer_median_s"] - left["answer_median_s"]
            extraction = prep["short_materialize_s"]
            assert math.isfinite(extraction) and extraction > 0
            equal_decode = all(len(a) == len(b) for a, b in zip(left["traces"], right["traces"], strict=True))
            subset.append(dict(baseline=baseline, sample_id=sid, qa_index=qi, short_correct=left["correct"],
                               baseline_correct=right["correct"], short_choice=left["pred_choice"],
                               baseline_choice=right["pred_choice"], same_answer_text=left["pred_answer"] == right["pred_answer"],
                               kv_bytes_saved=right["kv_bytes"] - left["kv_bytes"], answer_s_saved=saved,
                               extraction_s=extraction, net_s_saved=saved - extraction, equal_decode_count=equal_decode))
        details.extend(subset)
        equal = [r["answer_s_saved"] for r in subset if r["equal_decode_count"]]
        comparisons.append(dict(left=ARMS[2], right=baseline, questions=len(subset),
                                left_only_correct=sum(r["short_correct"] > r["baseline_correct"] for r in subset),
                                right_only_correct=sum(r["short_correct"] < r["baseline_correct"] for r in subset),
                                choice_changes=sum(r["short_choice"] != r["baseline_choice"] for r in subset),
                                median_kv_bytes_saved=statistics.median(r["kv_bytes_saved"] for r in subset),
                                median_paired_answer_s_saved=statistics.median(r["answer_s_saved"] for r in subset),
                                equal_decode_count_questions=len(equal),
                                median_equal_decode_answer_s_saved=statistics.median(equal) if equal else None))
    compare_csv(summary, table(payload["exports/summary.csv"]))
    compare_csv(comparisons, table(payload["exports/paired.csv"]))
    economics = []
    for arm in (ARMS[1], ARMS[3], ARMS[0]):
        subset = [r for r in details if r["baseline"] == arm]
        economics.append(dict(baseline=arm, exact_answer_text=sum(r["same_answer_text"] for r in subset),
                              median_extraction_s=statistics.median(r["extraction_s"] for r in subset),
                              median_net_s_saved=statistics.median(r["net_s_saved"] for r in subset),
                              positive_net_questions=sum(r["net_s_saved"] > 0 for r in subset)))
    tables.mkdir(parents=True, exist_ok=True)
    for name, records in (("summary", summary), ("paired", comparisons), ("query_cost", economics), ("question_pairs", details)):
        write_csv(tables / f"{name}.csv", records)
    result = dict(status="pass", archive_sha256=ARCHIVE_SHA, notebook_sha256=NOTEBOOK_SHA, payloads=len(payload),
                  questions=40, answers=160, timed_calls=320, exact_reference_replays=replays,
                  elapsed_post_model_s=completion["elapsed_post_model_s"],
                  fp16_exceedances=sum(r["exceeds_active_bound"] for r in preflight["torch.float16"]),
                  summary=summary, comparisons=comparisons, query_cost=economics)
    audit_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(ROOT / f"results/short_query_{DATE}", ROOT / f"evidence/audits/short_query_{DATE}_audit.json"), indent=2))
