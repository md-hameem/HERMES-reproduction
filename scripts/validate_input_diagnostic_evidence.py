"""Independent saved-evidence audit; never execute archived Python or notebook cells."""
import argparse
import ast
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile

from validate_causal_pilot_evidence import check_manifest, check_prediction, close, sha, stable, table, write_csv

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("question_only", "recent_4", "recent_20", "native_prefix", "support_oracle")
ARCHIVE_SHA = "f184271bd8929f2ca5dead71a1ec60e4e1b555f33f2ed66d9909e3d7e0ffa1e8"
NOTEBOOK_SHA = "24df662b86955295cae4aa24a559806b72021ded6ce6a0815a0eb1ce321313bf"


def expected_frames(video, qi, method):
    end = math.ceil(float(video["conversations"][qi]["end_time"])*.5)
    if method in ("question_only", "support_oracle"):
        return []
    return list(range(max(0,end-int(method.split("_")[1])),end)) if method.startswith("recent_") else list(range(end))


def validate_row(row, video, qi, method, feature_hash):
    frames = expected_frames(video, qi, method)
    qa = video["conversations"][qi]
    assert row["sample_id"] == video["sample_id"] and row["method"] == method
    assert row["sampled_indices"] == frames and row["visual_tokens"] == len(frames)*196
    assert row["source_frame_indices"] == [int(video["timestamps"][i]["source_frame"]) for i in frames]
    assert row["sampled_start_seconds"] == [float(video["timestamps"][i]["start_s"]) for i in frames]
    assert all(t <= float(qa["end_time"])+1e-6 for t in row["sampled_start_seconds"])
    assert row["fresh_state"] is True and row["no_compression"] is True
    assert row["feature_fingerprint"] == feature_hash
    assert row["video_id"] == str(video["video_id"]) and row["qa_index"] == qi
    assert row["question"] == qa["question"] and row["choices"] == qa["choices"]
    assert row["gold_answer"] == qa["answer"] and row["gold_choice"] == "ABCDEFGH"[qa["choices"].index(qa["answer"])]
    assert row["query_time_s"] == qa["end_time"] and row["task"] == qa["task"]
    for key in ("ingest_s", "answer_s", "wall_s", "peak_allocated_gib"):
        assert math.isfinite(row[key]) and row[key] >= 0
    if row["status"] == "answered":
        check_prediction(row, video, qi)
        assert not row["skip_reason"] and row["cache_lengths_before_answer"] == [13+len(frames)*196]*24
        assert row["peak_allocated_gib"] > 0
        assert row["wall_s"] + 1e-6 >= row["ingest_s"]+row["answer_s"]
        assert method != "support_oracle"
        if method == "native_prefix":
            assert len(frames)*196 <= 8192
    else:
        assert row["status"] == "skipped" and row["correct"] is None
        assert row["pred_answer"] == row["pred_choice"] == ""
        if method == "native_prefix":
            assert row["skip_reason"] in ("configured_native_cap","context_limit") and len(frames)*196 > 8192
            if row["skip_reason"] == "context_limit":
                assert len(frames)*196 > 32768
        else:
            assert method == "support_oracle" and row["skip_reason"] == "not_requested"


def compare_csv(rows, csv_rows):
    assert len(rows) == len(csv_rows)
    for row, saved in zip(rows, csv_rows):
        for key, value in saved.items():
            original = row.get(key, "")
            if isinstance(original, (list,dict)):
                assert ast.literal_eval(value) == original, key
            elif isinstance(original, float):
                close(value, original)
            else:
                assert value == ("" if original is None else str(original)), (key,value,original)


def audit(archive_path, notebook_path, output, tables):
    assert sha(archive_path.read_bytes()) == ARCHIVE_SHA
    assert sha(notebook_path.read_bytes()) == NOTEBOOK_SHA
    with ZipFile(archive_path) as z:
        manifest, data = check_manifest(z)
    record = lambda name: json.loads(data[name])
    config, completion, reference = record("exports/run_config.json"), record("exports/completion.json"), record("reference.json")
    assert manifest["status"] == completion["status"] == "complete"
    assert sha(stable({k:v for k,v in config.items() if k != "config_hash"}).encode()) == config["config_hash"] == completion["config_hash"]
    assert config["methods"] == list(METHODS) and config["fresh_state"] is True
    assert config["native"] == 1 and config["native_max_tokens"] == 8192 and config["support"] == record("support_oracle.json") == []
    assert config["torch"] == "2.10.0+cu128" and config["transformers"] == "4.45.0.dev0"
    assert config["model_revision"] == "74dd0bf867a4cda7950c17663794267c60cf4b40"
    assert config["sample_fps"] == .5 and config["seed"] == 2024 and config["max_new_tokens"] == 16
    assert sha(data["reference.json"]) == config["reference_sha256"]
    assert reference["archive_sha256"] == "af4b7392e90bd3f433ad20182a9058d8c8027a8ede936ed5d759572be04e0361"
    with ZipFile(ROOT/"evidence/archives/HERMES_causal_scorer_pilot_evidence.zip") as z:
        _, prior = check_manifest(z)
    for path, expected in reference["source_sha256"].items():
        assert sha((ROOT/path).read_bytes().replace(b"\r\n",b"\n")) == expected
    for path,key,local in [("diagnostic_runner.py","runner_sha256","scripts/autodl_input_diagnostic.py"),
                           ("diagnostic_pilot_utils.py","utils_sha256","scripts/autodl_failure_pilot.py")]:
        ast.parse(data[path])
        assert sha(data[path]) == config[key]
        assert data[path] == (ROOT/local).read_bytes().replace(b"\r\n",b"\n")
    launch = record("launch.json")
    assert launch["runner_sha256"] == config["runner_sha256"] and launch["repository_revision"] == "2a1385e0abc94debb64761b0465713c799a81c1c"
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert not [o for c in nb["cells"] for o in c.get("outputs",[]) if o.get("output_type") == "error"]
    embedded = {}
    for cell in nb["cells"]:
        source = "".join(cell["source"])
        if source.startswith("RUNNER_SOURCE ="):
            for node in ast.parse(source).body:
                if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                    key = node.targets[0].id
                    if key in ("RUNNER_SOURCE","UTILS_SOURCE","REFERENCE_JSON"):
                        embedded[key] = ast.literal_eval(node.value).encode()
    assert embedded == dict(RUNNER_SOURCE=data["diagnostic_runner.py"],UTILS_SOURCE=data["diagnostic_pilot_utils.py"],REFERENCE_JSON=data["reference.json"])
    printed = "\n".join("".join(o.get("text",[])) for c in nb["cells"] for o in c.get("outputs",[]))
    assert ARCHIVE_SHA in printed and "DIAGNOSTIC_COMPLETE" in printed
    preflight = record("preflight.json")
    assert preflight["status"] == "pass" and preflight["source_cache_unchanged"] is True
    assert preflight["feature_ingestion"] == "bitwise_equal_KV_and_positions"
    assert [r["cache_lengths"] for r in preflight["causal_scorer_comparisons"]] == [[3149]*24,[269]*24]
    assert all(r["max_attention_error"] == 0 for r in preflight["causal_scorer_comparisons"])
    originals = {str(v["video_id"]):v for v in json.loads((ROOT/"data/streamingbench/streamingbench_realtime.json").read_text(encoding="utf-8"))}
    old_predictions = table(prior["exports/predictions.csv"])
    assert reference["predictions"] == old_predictions
    old = {(int(r["sample_id"]),int(r["qa_index"]),r["method"]):r for r in old_predictions}
    assets = record("asset_manifest.json")
    assert assets["mode"] == "verified_existing_videos" and len(assets["videos"]) == 8
    asset_map = {v["sample_id"]:v for v in assets["videos"]}
    videos = reference["videos"]
    assert len(videos) == len({v["sample_id"] for v in videos}) == 8
    replay = record("exports/replay_annotations.json")
    assert [v["sample_id"] for v in replay] == [v["sample_id"] for v in videos]
    rows = []
    checkpoint_names = set()
    for vi,video in enumerate(videos):
        sid = video["sample_id"]
        assert video["conversations"] == replay[vi]["conversations"] == originals[str(video["video_id"])]["conversations"]
        meta = json.loads(prior[f"features/sample_{sid}/complete.json"])
        timestamps = table(prior[f"features/sample_{sid}/sampled_timestamps.csv"])
        assert video["feature_meta"] == meta and video["timestamps"] == timestamps
        provenance = record(f"provenance/features_{sid}.json")
        assert provenance["origin"] == "verified_historical_cache" and provenance["metadata"] == meta
        assert provenance["timestamp_rows"] == timestamps and provenance["feature_fingerprint"] == sha(stable(meta).encode())
        assert meta["source_video_sha256"] == video["video_sha256"] == asset_map[sid]["sha256"]
        for qi in range(5):
            for method in METHODS:
                name = f"checkpoints/{method}/sample_{sid}_q{qi}.json"
                checkpoint_names.add(name)
                value = record(name)
                row = value["row"]
                validate_row(row,video,qi,method,provenance["feature_fingerprint"])
                identity = {k:row[k] for k in ("sample_id","video_id","qa_index","method","question","choices","gold_answer","gold_choice","query_time_s","task")}
                assert value["row_sha256"] == sha(stable(row).encode())
                expected_id = dict(config=config["config_hash"], feature=provenance["feature_fingerprint"],question=identity,frames=row["sampled_indices"])
                assert value["checkpoint_identity"] == sha(stable(expected_id).encode())
                rows.append(row)
    assert checkpoint_names == {n for n in data if n.startswith("checkpoints/")}
    assert len(rows) == completion["rows"] == 200 and completion["mandatory_answered"] == 120
    by_key = {(r["sample_id"],r["qa_index"],r["method"]):r for r in rows}
    csv_rows = table(data["exports/predictions.csv"])
    ordered = [by_key[(int(r["sample_id"]),int(r["qa_index"]),r["method"])] for r in csv_rows]
    assert len({(r["sample_id"],r["qa_index"],r["method"]) for r in ordered}) == 200
    compare_csv(ordered,csv_rows)
    summary = []
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        answered = [r for r in subset if r["status"] == "answered"]
        summary.append(dict(method=method,planned=40,recorded=len(subset),answered=len(answered),skipped=len(subset)-len(answered),
            correct=sum(r["correct"] for r in answered),accuracy_pct=100*sum(r["correct"] for r in answered)/len(answered) if answered else "",
            measured_wall_s=sum(r["wall_s"] for r in answered)))
    compare_csv(summary,table(data["exports/summary.csv"]))
    paired = []
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method and r["status"] == "answered"]
        for baseline in ("fifo","hermes","causal"):
            pairs = [(r,old[r["sample_id"],r["qa_index"],baseline]) for r in subset]
            paired.append(dict(method=method,historical_baseline=baseline,matched_questions=len(pairs),
                diagnostic_correct=sum(r["correct"] for r,b in pairs),historical_correct=sum(int(b["correct"]) for r,b in pairs),
                diagnostic_only_correct=sum(bool(r["correct"]) and not int(b["correct"]) for r,b in pairs),
                historical_only_correct=sum(not r["correct"] and bool(int(b["correct"])) for r,b in pairs),
                comparison_scope="Exact shared identities; historical session; changed input/history; descriptive development comparison"))
    compare_csv(paired,table(data["exports/paired_historical.csv"]))
    within = []
    for a,b in combinations(METHODS[:4],2):
        pairs = [(by_key[sid,qi,a],by_key[sid,qi,b]) for sid,qi in {(r["sample_id"],r["qa_index"]) for r in rows}
                 if by_key[sid,qi,a]["status"] == by_key[sid,qi,b]["status"] == "answered"]
        within.append(dict(method_a=a,method_b=b,matched_questions=len(pairs),a_correct=sum(x["correct"] for x,y in pairs),
            b_correct=sum(y["correct"] for x,y in pairs),a_only_correct=sum(bool(x["correct"]) and not y["correct"] for x,y in pairs),
            b_only_correct=sum(not x["correct"] and bool(y["correct"]) for x,y in pairs),choice_changes=sum(x["pred_choice"] != y["pred_choice"] for x,y in pairs)))
    cases, per_video = [], []
    for video in videos:
        sid = video["sample_id"]
        for qi,q in enumerate(video["conversations"]):
            case = dict(sample_id=sid,qa_index=qi,question=q["question"],task=q["task"],query_time_s=q["end_time"],gold=q["answer"],
                        historical_hermes_choice=old[sid,qi,"hermes"]["pred_choice"],historical_hermes_correct=int(old[sid,qi,"hermes"]["correct"]))
            for method in METHODS[:4]:
                r = by_key[sid,qi,method]
                case.update({method+"_choice":r["pred_choice"],method+"_correct":r["correct"],method+"_status":r["status"]})
            cases.append(case)
        for method in METHODS[:4]:
            selected = [r for r in rows if r["sample_id"] == sid and r["method"] == method and r["status"] == "answered"]
            per_video.append(dict(sample_id=sid,method=method,answered=len(selected),correct=sum(r["correct"] for r in selected)))
        per_video.append(dict(sample_id=sid,method="historical_hermes",answered=5,correct=sum(int(old[sid,qi,"hermes"]["correct"]) for qi in range(5))))
    review = table(data["review/support_review.csv"])
    assert len(review) == 40 and all(r["review_status"] == "pending" for r in review)
    pages = [n for n in data if n.startswith("review/") and n.endswith(".jpg")]
    assert len(pages) == 57
    log = data["captured_logs/diagnostic.log"].decode()
    assert log.count("QUESTION_DONE ") == 200 and log.count("VIDEO_DONE ") == 8 and log.count("FEATURE_REUSE_VERIFIED ") == 8
    assert "\nREUSE " not in log and "DIAGNOSTIC_COMPLETE" in log
    tables.mkdir(parents=True,exist_ok=True)
    for source,target in [("summary.csv","summary.csv"),("predictions.csv","predictions.csv"),("paired_historical.csv","paired_historical.csv"),("report.md","run_report.md")]:
        (tables/target).write_bytes(data["exports/"+source])
    write_csv(tables/"within_run_pairs.csv",within)
    write_csv(tables/"question_casebook.csv",cases)
    write_csv(tables/"per_video_metrics.csv",per_video)
    task_rows = []
    for task in sorted({r["task"] for r in rows}):
        for method in METHODS[:3]:
            subset = [r for r in rows if r["task"] == task and r["method"] == method]
            task_rows.append(dict(task=task,method=method,questions=len(subset),correct=sum(r["correct"] for r in subset)))
    write_csv(tables/"task_outcomes.csv",task_rows)
    figures = tables/"figures"
    figures.mkdir(exist_ok=True)
    for source,target in [("review/sample_204_q0/page_01.jpg","sample_204_q0_prefix.jpg"),
                          ("review/sample_204_q3/page_06.jpg","sample_204_q3_recent_page.jpg")]:
        (figures/target).write_bytes(data[source])
    native_choice_changes = {baseline:sum(r["pred_choice"] != old[r["sample_id"],r["qa_index"],baseline]["pred_choice"]
        for r in rows if r["method"] == "native_prefix" and r["status"] == "answered") for baseline in ("fifo","hermes","causal")}
    result = dict(status="pass",archive_sha256=ARCHIVE_SHA,notebook_sha256=NOTEBOOK_SHA,payloads=len(data),checkpoints=200,
        answered=sum(r["status"] == "answered" for r in rows),skipped=dict(Counter(r["skip_reason"] for r in rows if r["status"] == "skipped")),
        methods=summary,within_run_pairs=within,native_historical_pairs=[r for r in paired if r["method"] == "native_prefix"],
        native_historical_choice_changes=native_choice_changes,
        feature_reuse_videos=8,gpu_preflight="pass; zero maximum attention error on checked prefix before/after pruning",
        review_rows_pending=40,review_pages=57,retained_notebook_errors=0,
        elapsed_post_model_s=completion["elapsed_post_model_s"],
        limits=["Saved evidence audit; raw videos/feature tensors and GPU computation cannot be independently rerun from ZIP",
                "Outcome-selected development sample; no population significance or equivalence claim",
                "Native answers cover 14 early feasible questions, not all 40",
                "No verified-support oracle or controlled past-answer intervention executed",
                "Changed input extent, positions and history do not isolate eviction causation",
                "Historical comparisons are not same-session efficiency comparisons"])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8",newline="\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,default=ROOT/"evidence/archives/HERMES_input_diagnostic_evidence.zip")
    parser.add_argument("--notebook",type=Path,default=ROOT/"notebooks/executed/2026-11-20_hermes-input-diagnostic.ipynb")
    parser.add_argument("--output",type=Path,default=ROOT/"evidence/audits/input_diagnostic_2026-11-20_audit.json")
    parser.add_argument("--tables",type=Path,default=ROOT/"results/input_diagnostic_2026-11-20")
    args = parser.parse_args()
    result = audit(args.archive,args.notebook,args.output,args.tables)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
