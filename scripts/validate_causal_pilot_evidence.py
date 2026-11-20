"""Audit the supplied causal-scorer pilot without executing its embedded code.

Recompute identities, predictions, exported tables, timing totals, sampling
boundaries and historical replay agreement. Raw videos and feature tensors are
absent, so their hashes and the GPU computation cannot be independently rerun.
"""

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("fifo", "hermes", "causal")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stable(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def close(a, b):
    assert math.isfinite(float(a)) and math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9), (a, b)


def table(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def write_csv(path, rows):
    assert rows
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def check_manifest(archive):
    names = archive.namelist()
    assert len(names) == len(set(names)), "Duplicate archive member names"
    for item in archive.infolist():
        path = PurePosixPath(item.filename)
        assert not path.is_absolute() and ".." not in path.parts and "\\" not in item.filename
        assert ((item.external_attr >> 16) & 0o170000) != 0o120000
    assert archive.testzip() is None
    manifest = json.loads(archive.read("evidence_manifest.json"))
    records = manifest["files"]
    assert len(records) == len({r["path"] for r in records})
    assert {r["path"] for r in records} | {"evidence_manifest.json"} == set(names)
    payload = {}
    for row in records:
        data = archive.read(row["path"])
        assert len(data) == row["bytes"] and sha(data) == row["sha256"], row["path"]
        payload[row["path"]] = data
    return manifest, payload


def check_prediction(row, video, qi):
    q = video["conversations"][qi]
    gold = "ABCDEFGH"[q["choices"].index(q["answer"])]
    assert row["video_id"] == str(video["video_id"]) and row["qa_index"] == qi
    assert row["question"] == q["question"] and row["choices"] == q["choices"]
    assert row["gold_answer"] == q["answer"] and row["gold_choice"] == gold
    assert row["task"] == q["task"] and row["query_time_s"] == q["end_time"]
    # BaseVQA stores generated text without newlines; preserve its exact parser.
    text = row["pred_answer"].strip()
    parsed = text[text.index(")") - 1:text.index(")")] if ")" in text else text[:1]
    assert row["pred_choice"] == parsed
    assert row["correct"] == int(row["pred_choice"] == gold)
    assert math.isfinite(row["answer_s"]) and row["answer_s"] >= 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "evidence/archives/HERMES_causal_scorer_pilot_evidence.zip")
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks/executed/2026-11-20_hermes-causal-scorer-pilot.ipynb")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/audits/causal_pilot_2026-11-20_audit.json")
    parser.add_argument("--tables", type=Path, default=ROOT / "results/causal_pilot_2026-11-20")
    args = parser.parse_args()
    with ZipFile(args.archive) as archive:
        manifest, data = check_manifest(archive)
    record = lambda name: json.loads(data[name])
    config, completion = record("exports/run_config.json"), record("exports/completion.json")
    assert manifest["status"] == completion["status"] == "complete"
    assert config["methods"] == completion["methods"] == list(METHODS)
    raw_config = {k: v for k, v in config.items() if k not in ("config_hash", "soft_budget_hours_this_invocation")}
    assert sha(stable(raw_config).encode()) == config["config_hash"] == completion["config_hash"]
    assert config["model_revision"] == "74dd0bf867a4cda7950c17663794267c60cf4b40"
    assert config["torch"] == "2.10.0+cu128" and config["transformers"] == "4.45.0.dev0"
    assert config["kv_size"] == 4000 and config["sample_fps"] == .5 and config["seed"] == 2024
    for path, expected in config["source_sha256"].items():
        assert sha((ROOT / path).read_bytes().replace(b"\r\n", b"\n")) == expected, path
    runner = data["pilot_runner.py"].replace(b"\r\n", b"\n")
    assert sha(runner) == config["runner_sha256"] == "6b52ce8cc8b8ac068dfe076f1b2bc575ef318df8b4853b87acf025f524d9faa9"
    ast.parse(runner)
    historical = subprocess.run(["git", "show", "39be0c5:scripts/autodl_failure_pilot.py"], cwd=ROOT, capture_output=True)
    if historical.returncode == 0:
        assert runner == historical.stdout.replace(b"\r\n", b"\n"), "Packaged runner differs from known pre-heartbeat revision"
    launch = record("launch.json")
    assert launch["runner_sha256"] == config["runner_sha256"] and launch["repository_revision"] == "2a1385e0abc94debb64761b0465713c799a81c1c"

    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    errors = [o for c in notebook["cells"] for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, errors
    embedded = ["".join(c["source"]) for c in notebook["cells"] if "".join(c["source"]).startswith("RUNNER_SOURCE =")]
    assert len(embedded) == 1 and ast.literal_eval(ast.parse(embedded[0]).body[0].value).encode() == runner
    rendered = "\n".join("".join(o.get("text", [])) for c in notebook["cells"] for o in c.get("outputs", []))
    assert sha(args.archive.read_bytes()) in rendered and "PILOT_COMPLETE" in rendered
    preflight = record("preflight.json")
    assert preflight["status"] == "pass" and preflight["source_cache_unchanged"] is True
    assert preflight["feature_ingestion"] == "bitwise_equal_KV_and_positions"
    references = preflight["causal_scorer_comparisons"]
    assert [r["stage"] for r in references] == ["before_pruning", "after_position_sensitive_pruning"]
    assert references[0]["cache_lengths"] == [3149] * 24 and references[1]["cache_lengths"] == [269] * 24
    assert all(0 <= r["max_attention_error"] <= .004 for r in references)

    full_annotations = json.loads((ROOT / "data/streamingbench/streamingbench_realtime.json").read_text(encoding="utf-8"))
    originals = {str(v["video_id"]): v for v in full_annotations}
    videos = record("exports/replay_annotations.json")
    assert len(videos) == completion["videos"] == completion["planned_videos"] == 8
    assert len({str(v["video_id"]) for v in videos}) == len(videos)
    assert config["conversations"] == [{"video_id": v["video_id"], "conversations": v["conversations"]} for v in videos]
    selection = record("selection.json")["cases"]
    assert [r["case_id"] for r in selection] == config["selected_case_ids"]
    assert len(selection) == len({r["video_id"] for r in selection}) == 8
    assert Counter(r["outcome"] for r in selection) == {o: 2 for o in ("hermes_only", "fifo_only", "both_wrong", "both_correct")}
    selected = {(r["video_id"], int(r["qa_index"])) for r in selection}
    old_inventory = {(r["video_id"], int(r["qa_index"])): r for r in table((ROOT / "results/failure_pilot/paired_inventory.csv").read_bytes())}
    for r in selection:
        expected = old_inventory[(r["video_id"], int(r["qa_index"]))]
        assert all(str(value) == expected[key] for key, value in r.items()), r["case_id"]

    assets = record("asset_manifest.json")
    assert {str(r["sample_id"]): r["sha256"] for r in assets["videos"]} == config["assets"]
    assert all(re.fullmatch(r"[a-f0-9]{64}", r["sha256"]) and r["bytes"] > 0 for r in assets["videos"])
    assert assets["mode"] == "range_selected_members" and assets["whole_archive_sha256_verified"] is False
    predictions, runs, features, stage_by_question = [], [], [], {}
    for video in videos:
        vid = str(video["video_id"])
        sid = int(re.search(r"sample_(\d+)", video["video_path"])[1])
        assert video["conversations"] == originals[vid]["conversations"]
        assert sid == int(re.search(r"sample_(\d+)", originals[vid]["video_path"])[1])
        meta = record(f"features/sample_{sid}/complete.json")
        assert meta["source_video_sha256"] == config["assets"][str(sid)] and meta["sample_id"] == sid
        assert meta["sampling_step"] == int(meta["source_fps_rounded"] / .5)
        assert meta["preparation_s"] >= 0
        features.append(meta)
        timestamps = table(data[f"features/sample_{sid}/sampled_timestamps.csv"])
        assert len(timestamps) == meta["sampled_frames_available"]
        for i, ts in enumerate(timestamps):
            assert int(ts["sampled_index"]) == i and int(ts["source_frame"]) == i * meta["sampling_step"]
            assert float(ts["end_s"]) >= float(ts["start_s"]) >= 0
            if i:
                assert float(ts["start_s"]) >= float(timestamps[i - 1]["start_s"])
        cursor = 0
        for chunk in meta["chunks"]:
            assert chunk["start"] == cursor and 0 < chunk["stop"] - cursor <= 16
            assert chunk["tokens"] == 196 * (chunk["stop"] - cursor)
            assert re.fullmatch(r"[a-f0-9]{64}", chunk["sha256"])
            cursor = chunk["stop"]
        assert cursor == math.ceil(video["conversations"][-1]["end_time"] * .5)
        for method in METHODS:
            run = record(f"checkpoints/{method}/sample_{sid}.json")
            assert run["status"] == "complete" and run["method"] == method and run["config_hash"] == config["config_hash"]
            assert len(run["predictions"]) == len(video["conversations"])
            assert sha(stable(run["predictions"]).encode()) == run["prediction_sha256"]
            for qi, row in enumerate(run["predictions"]):
                check_prediction(row, video, qi)
                assert row["sample_id"] == sid
                predictions.append({"method": method, **row})
            assert len(run["events"]) == len(meta["chunks"])
            previous_len, ever_pruned = 13, False
            for event, chunk in zip(run["events"], meta["chunks"]):
                assert (event["qa_index"], event["sampled_start"], event["sampled_stop"]) == (chunk["qa_index"], chunk["start"], chunk["stop"])
                assert all(math.isfinite(event[k]) and event[k] >= 0 for k in ("ingest_s", "compression_s"))
                lengths = event["cache_lengths"]
                assert len(lengths) == 24 and len(set(lengths)) == 1 and 13 <= max(lengths) <= 4013
                ever_pruned |= max(lengths) < previous_len + chunk["tokens"]
                previous_len = max(lengths)
                stage_by_question[(method, vid, event["qa_index"])] = ever_pruned
            assert run["replay_wall_s"] >= sum(e["ingest_s"] + e["compression_s"] for e in run["events"]) + sum(r["answer_s"] for r in run["predictions"])
            runs.append(dict(method=method, sample_id=sid, video_id=vid,
                             correct=sum(r["correct"] for r in run["predictions"]), questions=len(run["predictions"]),
                             events=len(run["events"]), replay_wall_s=run["replay_wall_s"],
                             ingest_s=sum(e["ingest_s"] for e in run["events"]),
                             compression_s=sum(e["compression_s"] for e in run["events"]),
                             answer_s=sum(r["answer_s"] for r in run["predictions"]),
                             peak_allocated_gib=run["peak_allocated_gib"], peak_reserved_gib=run["peak_reserved_gib"]))

    indexed = {(r["method"], r["video_id"], r["qa_index"]): r for r in predictions}
    assert len(indexed) == len(predictions) == 120
    assert {n for n in data if n.startswith("checkpoints/")} == {
        f"checkpoints/{m}/sample_{int(re.search(r'sample_(\d+)', v['video_path'])[1])}.json"
        for m in METHODS for v in videos}
    exported = table(data["exports/predictions.csv"])
    assert len(exported) == 120
    assert {(r["method"], r["video_id"], int(r["qa_index"])) for r in exported} == set(indexed)
    for row in exported:
        actual = indexed[(row["method"], row["video_id"], int(row["qa_index"]))]
        for key, value in actual.items():
            if isinstance(value, list):
                assert ast.literal_eval(row[key]) == value
            elif isinstance(value, (int, float)):
                close(row[key], value)
            else:
                assert row[key] == value
    metrics = []
    for method in METHODS:
        rows = [r for r in predictions if r["method"] == method]
        group = [r for r in runs if r["method"] == method]
        target = [r for r in rows if (r["video_id"], r["qa_index"]) in selected]
        metrics.append(dict(method=method, videos=len(group), questions=len(rows), correct=sum(r["correct"] for r in rows),
                            accuracy_pct=100 * sum(r["correct"] for r in rows) / len(rows),
                            selected_targets=len(target), target_correct=sum(r["correct"] for r in target),
                            total_replay_s=sum(r["replay_wall_s"] for r in group),
                            peak_allocated_gib=max(r["peak_allocated_gib"] for r in group),
                            total_ingest_s=sum(r["ingest_s"] for r in group),
                            total_compression_s=sum(r["compression_s"] for r in group),
                            total_answer_s=sum(r["answer_s"] for r in group)))
    summaries = table(data["exports/summary.csv"])
    assert len(summaries) == len(completion["summary"]) == len(METHODS)
    for i, metric in enumerate(metrics):
        assert metric["method"] == summaries[i]["method"] == completion["summary"][i]["method"]
        for key in summaries[i]:
            if key != "method":
                close(summaries[i][key], metric[key])
                close(completion["summary"][i][key], metric[key])

    casebook = []
    for video in videos:
        vid = str(video["video_id"])
        sid = int(re.search(r"sample_(\d+)", video["video_path"])[1])
        times = table(data[f"features/sample_{sid}/sampled_timestamps.csv"])
        for qi, qa in enumerate(video["conversations"]):
            f, h, c = [indexed[(m, vid, qi)] for m in METHODS]
            last = float(times[math.ceil(qa["end_time"] * .5) - 1]["start_s"])
            assert last <= qa["end_time"], (sid, qi, last, qa["end_time"])
            prior_pruning = stage_by_question.get(("hermes", vid, qi), stage_by_question.get(("hermes", vid, qi - 1), False))
            casebook.append(dict(case_id=f"sample_{sid:03d}_q{qi}", sample_id=sid, video_id=vid, qa_index=qi,
                                 query_time_s=qa["end_time"], last_sampled_start_s=last, task=qa["task"],
                                 required_ability=qa["required_ability"], question=qa["question"],
                                 choices_json=json.dumps(qa["choices"], ensure_ascii=False), gold_answer=qa["answer"], gold_choice=h["gold_choice"],
                                 fifo_choice=f["pred_choice"], hermes_choice=h["pred_choice"], causal_choice=c["pred_choice"],
                                 fifo_correct=f["correct"], hermes_correct=h["correct"], causal_correct=c["correct"],
                                 shared_failure=all(r["correct"] == 0 for r in (f, h, c)),
                                 hermes_had_prior_token_eviction=prior_pruning, selected_target=(vid, qi) in selected,
                                 support_status="unannotated", failure_cause="unclassified"))
    pairs = []
    for a, b in (("causal", "hermes"), ("hermes", "fifo"), ("causal", "fifo")):
        ar, br = [{(r["video_id"], r["qa_index"]): r for r in predictions if r["method"] == m} for m in (a, b)]
        pairs.append(dict(method_a=a, method_b=b, questions=len(ar),
                          a_only_correct=sum(ar[k]["correct"] == 1 and br[k]["correct"] == 0 for k in ar),
                          b_only_correct=sum(ar[k]["correct"] == 0 and br[k]["correct"] == 1 for k in ar),
                          choice_changes=sum(ar[k]["pred_choice"] != br[k]["pred_choice"] for k in ar),
                          correctness_changes=sum(ar[k]["correct"] != br[k]["correct"] for k in ar),
                          inference_scope="Outcome-selected development sample; no population significance claim"))
    exported_pairs = table(data["exports/paired_questions.csv"])
    assert len(exported_pairs) == len(casebook)
    by_key = {(r["video_id"], r["qa_index"]): r for r in casebook}
    assert {(r["video_id"], int(r["qa_index"])) for r in exported_pairs} == set(by_key)
    for row in exported_pairs:
        expected = by_key[(row["video_id"], int(row["qa_index"]))]
        for key, value in row.items():
            target = expected["selected_target"] if key == "is_selected_target" else expected[key]
            assert str(target) == value
    history = []
    with ZipFile(ROOT / "evidence/archives/HERMES_AutoDL_five_stage_evidence.zip") as old:
        for method in ("fifo", "hermes"):
            previous = {(r["video_id"], int(r["qa_index"])): r for r in table(old.read(f"exports/full_{method}_4000_predictions.csv"))}
            new = [r for r in predictions if r["method"] == method]
            history.append(dict(method=method, questions=len(new),
                                old_correct=sum(int(previous[(r["video_id"], r["qa_index"])]["correct"]) for r in new),
                                new_correct=sum(r["correct"] for r in new),
                                choice_changes=sum(previous[(r["video_id"], r["qa_index"])]["pred_choice"] != r["pred_choice"] for r in new)))
    reviews = table(data["review/evidence_review.csv"])
    assert len(reviews) == len(selection) and {r["case_id"] for r in reviews} == set(config["selected_case_ids"])
    assert all(r["review_status"] == "pending" for r in reviews)
    log = data["captured_logs/pilot.log"].decode("utf-8", errors="replace")
    assert len(re.findall(r"^METHOD_DONE ", log, re.M)) == 24 and len(re.findall(r"^BLOCK_DONE ", log, re.M)) == 8
    assert not re.search(r"^REUSE ", log, re.M) and "PREFLIGHT_PASS" in log and "PILOT_COMPLETE" in log

    args.tables.mkdir(parents=True, exist_ok=True)
    for filename in ("summary.csv", "predictions.csv", "paired_questions.csv"):
        target = args.tables / filename
        if target.exists():
            assert target.read_bytes() == data["exports/" + filename]
        target.write_bytes(data["exports/" + filename])
    (args.tables / "run_report.md").write_bytes(data["exports/report.md"])
    for name, rows in (("audited_method_metrics", metrics), ("audited_video_metrics", runs),
                       ("question_casebook", casebook), ("pairwise_outcomes", pairs), ("historical_replay_comparison", history)):
        write_csv(args.tables / (name + ".csv"), rows)
    review_queue = []
    for row in casebook:
        discordant = row["fifo_correct"] != row["hermes_correct"]
        if not row["shared_failure"] and not discordant:
            continue
        early = not row["hermes_had_prior_token_eviction"]
        reason = ("Shared failure before eviction: investigate sampled evidence, perception and part/time grounding" if early else
                  "HERMES/FIFO correctness discordance: locate true supporting evidence and test dependence on it" if discordant else
                  "Shared failure after eviction: cause unproven; establish support visibility and oracle recoverability")
        review_queue.append(dict(case_id=row["case_id"], sample_id=row["sample_id"], qa_index=row["qa_index"],
                                 priority="first" if early or discordant else "second", reason=reason,
                                 question=row["question"], query_time_s=row["query_time_s"],
                                 support_intervals_seconds_json="", visible_in_original_sampled_prefix="pending",
                                 oracle_support_result="pending", cause="unclassified", notes=""))
    # Keep human annotations separate from regenerated queue data.
    write_csv(args.tables / "next_review_queue.csv", review_queue)
    task_rows = []
    for task in sorted({r["task"] for r in casebook}):
        group = [r for r in casebook if r["task"] == task]
        task_rows.append(dict(task=task, questions=len(group),
                              **{m + "_correct": sum(r[m + "_correct"] for r in group) for m in METHODS},
                              shared_failures=sum(r["shared_failure"] for r in group)))
    write_csv(args.tables / "task_outcomes.csv", task_rows)
    audit = dict(status="pass", archive_sha256=sha(args.archive.read_bytes()), notebook_sha256=sha(args.notebook.read_bytes()),
                 manifest_payloads=len(manifest["files"]), archive_files=len(data) + 1, video_checkpoints=len(runs),
                 distinct_videos=len(videos), questions_per_method=40, answers=len(predictions),
                 runner_sha256=config["runner_sha256"], runner_matches_revision="39be0c5", config_hash=config["config_hash"],
                 runner_history_byte_comparison_available=historical.returncode == 0,
                 environment=dict(torch=config["torch"], transformers=config["transformers"], gpu=config["gpu"]),
                 preflight=preflight, method_metrics=metrics, pairwise_outcomes=pairs, historical_replay=history,
                 shared_failures=sum(r["shared_failure"] for r in casebook),
                 shared_failures_before_any_eviction=sum(r["shared_failure"] and not r["hermes_had_prior_token_eviction"] for r in casebook),
                 shared_preparation_s=sum(f["preparation_s"] for f in features),
                 chunks_per_method=sum(len(f["chunks"]) for f in features),
                 encoded_frames_per_method=sum(sum(c["stop"] - c["start"] for c in f["chunks"]) for f in features),
                 completion_elapsed_s=completion["elapsed_s"], completion_elapsed_scope="Post-model-load/preflight/features/replay phase; excludes dataset download, installation, model loading and final packaging",
                 asset_transfer_bytes=assets["transferred_bytes"], whole_dataset_archive_sha256_verified=False,
                 future_sampled_start_bound_violations=0, retained_review_rows_pending=len(reviews),
                 limits=["Outcome-selected development sample, not untouched confirmation", "Raw videos/feature tensors absent; GPU computation not independently rerun",
                         "Native check is one 16-frame prefix before/after forced pruning, not all scored chunks", "Reported timings share vision preparation; no full-system speed claim",
                         "Executed notebook has extra network/environment cells and nonconsecutive counts; no claim of a fresh single-pass replay",
                         "Contact sheets are sparse; support intervals/token survival remain unverified"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"PASS {len(manifest['files'])} payloads, {len(runs)} checkpoints, {len(predictions)} answers, all summary/timing/identity checks")
    print(json.dumps(dict(methods=metrics, pairs=pairs, shared_failures=audit["shared_failures"], before_eviction=audit["shared_failures_before_any_eviction"]), indent=2))


if __name__ == "__main__":
    main()
