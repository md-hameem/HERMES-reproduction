"""Verify the frozen AutoDL clip comparison without executing uploaded content."""
import ast
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from zipfile import ZipFile

from make_confirmation_manifest import OUTPUT, ROOT, build
from validate_causal_pilot_evidence import check_manifest, check_prediction, sha, table, write_csv
from validate_context_diagnostic_evidence import expected_groups
from validate_input_diagnostic_evidence import compare_csv
from validate_short_query_evidence import SOURCES as SHORT_SOURCES
from validate_stage_attribution_evidence import fingerprint

DATE = "2026-11-21"
ARCHIVE = ROOT / "evidence/archives/HERMES_short_query_confirmation_evidence.zip"
NOTEBOOK = ROOT / f"notebooks/executed/{DATE}_hermes-short-query-confirmation.ipynb"
ARCHIVE_SHA = "291114a32391bbb039e176355688b4b4af269f35ad7a129a73205c73d05575e3"
NOTEBOOK_SHA = "57873f9a0c2b714cadcc9c00a14ed9f747b09d2c08b7d55648f14a34dd623d12"
ARMS = ("compact_recent4", "latest_two_short_readout", "all_history_short_readout")
SOURCES = {
    **SHORT_SOURCES,
    "provenance/short_base.py": "SHORT_BASE_SOURCE",
    "provenance/selection_base.py": "SELECTION_BASE_SOURCE",
}


def audit(tables, receipt):
    assert sha(ARCHIVE.read_bytes()) == ARCHIVE_SHA
    assert sha(NOTEBOOK.read_bytes()) == NOTEBOOK_SHA
    with ZipFile(ARCHIVE) as zipped:
        manifest, data = check_manifest(zipped)

    def read(name):
        return json.loads(data[name])

    frozen, config, complete = (read(name) for name in ("reference.json", "exports/run_config.json", "exports/completion.json"))
    assert frozen == json.loads(OUTPUT.read_bytes()) == build()
    assert manifest["status"] == complete["status"] == "complete"
    assert (complete["main_questions"], complete["main_answers"]) == (150, 450)
    assert config["manifest_sha256"] == sha(data["reference.json"])
    assert config["sources"] == {n: sha(data[n]) for n in SOURCES}
    assert read("launch.json")["runner_sha256"] == config["sources"]["stage_runner.py"]
    assert fingerprint({k: v for k, v in config.items() if k != "config_hash"}) == config["config_hash"] == complete["config_hash"]
    assert config["arms"] == list(ARMS)
    notebook = json.loads(NOTEBOOK.read_bytes())
    source = next(c["source"] for c in notebook["cells"] if c["id"] == "confirmation-08")
    embedded = {}
    for node in ast.parse("".join(source)).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in (*SOURCES.values(), "REFERENCE_JSON"):
                assert name not in embedded
                embedded[name] = ast.literal_eval(node.value).encode()
    assert all(embedded[v] == data[n] for n, v in SOURCES.items())
    assert embedded["REFERENCE_JSON"] == data["reference.json"]
    outputs = [o for c in notebook["cells"] for o in c.get("outputs", [])]
    assert not any(o["output_type"] == "error" for o in outputs)
    streams = "\n".join("".join(o["text"]) for o in outputs if o["output_type"] == "stream")
    assert f"SHA256={ARCHIVE_SHA}" in streams and "Status: complete | Main answers: 450" in streams
    log = data["captured_logs/confirmation.log"].decode()
    assert log.count("VIDEO_BLOCK_DONE ") == 30 and "SHORT_QUERY_CONFIRMATION_COMPLETE" in log
    preflight = read("preflight.json")
    assert preflight["status"] == "pass" and preflight["source_cache_unchanged"]
    assert preflight["feature_ingestion"] == "bitwise_equal_KV_and_positions"

    assets = read("asset_manifest.json")
    expected_ids = [c["sample_id"] for sh in frozen["shards"] for c in sh["clips"]]
    assert [v["sample_id"] for v in assets["videos"]] == expected_ids
    assert len({v["sha256"] for v in assets["videos"]}) == 30
    transfer = 0
    for shard in frozen["shards"]:
        media = read(f"provenance/media/{shard['shard']}.json")
        assert media["archive"] == shard["archive"]
        assert media["videos"] == [v for v in assets["videos"] if v["sample_id"] in {c["sample_id"] for c in shard["clips"]}]
        transfer += media["transferred_bytes"]
    raw = (ROOT / frozen["annotation_path"]).read_bytes().replace(b"\r\n", b"\n")
    assert sha(raw) == frozen["annotation_sha256"]
    annotations = {int(re.search(r"sample_(\d+)", v["video_path"])[1]): v for v in json.loads(raw)}
    rows, question_order = {}, []
    nulls = 0
    for asset in assets["videos"]:
        sid = asset["sample_id"]
        video = annotations[sid]
        provenance = read(f"provenance/features_{sid}.json")
        meta, times = provenance["metadata"], provenance["timestamps"]
        assert meta["sample_id"] == sid
        assert meta["identity"] == fingerprint(dict(config=config["config_hash"], source=asset["sha256"]))
        assert all(int(t["sampled_index"]) == i and float(t["start_s"]) <= 2*i + 1e-6 for i, t in enumerate(times))
        assert all(int(a["source_frame"]) < int(b["source_frame"]) for a, b in zip(times, times[1:]))
        for qi, qa in enumerate(video["conversations"]):
            name = f"sample_{sid}_q{qi}.json"
            prep, check = read(f"preparation/{name}"), read(f"checkpoints/{name}")
            end = math.ceil(.5 * float(qa["end_time"]))
            frames = list(range(max(0, end - 20), end))
            history, recent = frames[:-4], frames[-4:]
            assert all(float(times[i]["start_s"]) <= qa["end_time"] + 1e-6 for i in frames)
            assert (prep["history_indices"], prep["recent_indices"]) == (history, recent)
            order = list(ARMS)
            random.Random(2024 + 31*sid + qi).shuffle(order)
            assert prep["answer_order"] == order
            assert check["key"] == fingerprint(dict(config=config["config_hash"], feature=fingerprint(meta),
                                                    sample=sid, qa=qi, h=history, r=recent))
            assert check["rows_sha256"] == fingerprint(check["rows"])
            assert [r["method"] for r in check["rows"]] == list(ARMS)
            if not history:
                nulls += 1
                assert len({r["pred_choice"] for r in check["rows"]}) == 1
            for i, row in enumerate(check["rows"]):
                check_prediction(row, video, qi)
                assert row["sample_id"] == sid and row["recent_indices"] == recent
                assert row["selected_history_indices"] == ([], history[-2:], history)[i]
                assert row["recent_groups"] == expected_groups(recent, "original", meta)
                assert all(math.isfinite(row[k]) and row[k] >= 0
                           for k in ("construction_s", "extraction_s", "answer_s"))
                past = 13 + 196*len(recent)
                pos = past if i == 0 else 13 + 196*len(frames)
                assert row["query_kv_bytes"] == 12288*past
                assert 1 <= len(row["position_trace"]) <= 17
                for j, event in enumerate(row["position_trace"]):
                    n = event["query_tokens"]
                    assert n > 1 if j == 0 else n == 1
                    assert (event["past_tokens"], event["logical_start"]) == (past, pos)
                    past, pos = past+n, pos+n
                rows[sid, qi, row["method"]] = row
            question_order.append((sid, qi))
    assert len(question_order) == sum(n.startswith("checkpoints/") for n in data) == 150 and len(rows) == 450
    answers = table(data["exports/answers.csv"])
    keys = [(int(r["sample_id"]), int(r["qa_index"]), r["method"]) for r in answers]
    assert len(keys) == len(set(keys)) and set(keys) == rows.keys()
    compare_csv([rows[k] for k in keys], answers)

    summaries = []
    for arm in ARMS:
        subset = [rows[sid, qi, arm] for sid, qi in question_order]
        stats = {k: statistics.median(r[k] for r in subset)
                 for k in ("query_kv_bytes", "construction_s", "extraction_s", "answer_s")}
        summaries.append(dict(arm=arm, answers=len(subset), correct=sum(r["correct"] for r in subset),
                              **{"median_" + k: v for k, v in stats.items()}))
    compare_csv(summaries, table(data["exports/summary.csv"]))
    comparisons, changed = [], []
    for a, b in ((ARMS[2], ARMS[0]), (ARMS[1], ARMS[0]), (ARMS[2], ARMS[1])):
        by_video = defaultdict(list)
        costs = []
        choices = 0
        for sid, qi in question_order:
            left, right = rows[sid, qi, a], rows[sid, qi, b]
            by_video[sid].append(left["correct"] - right["correct"])
            costs.append(sum(right[k] - left[k] for k in ("construction_s", "extraction_s", "answer_s")))
            if left["pred_choice"] != right["pred_choice"]:
                choices += 1
                changed.append(dict(candidate=a, baseline=b, sample_id=sid, qa_index=qi, task=left["task"],
                                    candidate_choice=left["pred_choice"], baseline_choice=right["pred_choice"],
                                    gold=left["gold_choice"], outcome=left["correct"] - right["correct"]))
        values = [v for block in by_video.values() for v in block]
        comparisons.append(dict(candidate=a, baseline=b, wins=values.count(1), losses=values.count(-1),
                                choice_changes=choices, videos_positive=sum(sum(v)>0 for v in by_video.values()),
                                videos_negative=sum(sum(v)<0 for v in by_video.values()),
                                median_prepared_s_saved=statistics.median(costs)))
        if (a, b) == (ARMS[2], ARMS[0]):
            rng = random.Random(2024)
            # Every frozen clip has five questions, so bootstrap the per-video sums.
            assert {len(v) for v in by_video.values()} == {5}
            totals = [sum(v) for v in by_video.values()]
            draws = sorted(100*sum(rng.choices(totals, k=30))/150 for _ in range(10000))
            loo = [100*(sum(values)-x)/145 for x in totals]
            primary = dict(candidate=a, baseline=b, wins=values.count(1), losses=values.count(-1),
                           videos=30, questions=150, difference_pp=100*sum(values)/150,
                           ci_low_pp=draws[250], ci_high_pp=draws[9750],
                           leave_one_video_min_pp=min(loo), leave_one_video_max_pp=max(loo))
    assert primary == read("exports/paired_primary.json")
    tables.mkdir(parents=True, exist_ok=True)
    write_csv(tables / "summary.csv", summaries)
    write_csv(tables / "paired.csv", comparisons)
    write_csv(tables / "choice_changes.csv", changed)
    write_csv(tables / "per_video.csv", [
        dict(sample_id=sid, **{arm: sum(rows[v, qi, arm]["correct"] for v, qi in question_order if v == sid) for arm in ARMS})
        for sid in expected_ids
    ])
    result = dict(status="pass", archive_sha256=ARCHIVE_SHA, notebook_sha256=NOTEBOOK_SHA, payloads=len(data),
                  clips=30, exact_unique_media_sha256=30, source_overlap="unverified", questions=150, answers=450,
                  empty_history_questions=nulls, transferred_bytes=transfer,
                  elapsed_post_model_s=complete["elapsed_post_model_s"], summary=summaries,
                  primary=primary, paired=comparisons)
    receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(ROOT / f"results/short_query_confirmation_{DATE}",
                           ROOT / f"evidence/audits/short_query_confirmation_{DATE}_audit.json"), indent=2))
