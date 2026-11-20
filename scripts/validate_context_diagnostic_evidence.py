"""Audit context intervention evidence without executing archived source/cells."""
import argparse
import ast
import json
import math
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from validate_causal_pilot_evidence import check_manifest,check_prediction,sha,stable,table,write_csv
from validate_input_diagnostic_evidence import compare_csv

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE_SHA="fee3165e323c0f7106688a37cb185ec390d4f831e9debdfbf0352bd72881b774"
NOTEBOOK_SHA="26f2ec995b044b7d1595c73945eebc0285675882a7f93bd7719b53dfa6bf2055"
CASES=((204,0),(203,2),(204,3),(209,0),(208,4),(202,0),(208,0))
ARMS=("question_only","recent_4_original","recent_20_original","full_prefix_original","full_prefix_framewise",
      "recent_20_fixed4","recent_20_opening_fill","recent_20_local_fill","full_prefix_fixed4","full_prefix_opening_fill","full_prefix_local_fill","candidate_support")
CONTRASTS=(("full_prefix_original","full_prefix_framewise"),("full_prefix_original","full_prefix_fixed4"),
    ("recent_20_original","recent_20_fixed4"),("recent_20_fixed4","recent_20_opening_fill"),
    ("recent_20_fixed4","recent_20_local_fill"),("full_prefix_fixed4","full_prefix_opening_fill"),("full_prefix_fixed4","full_prefix_local_fill"))


def fingerprint(value):
    return sha(stable(value).encode())


def expected_plan(video,qi,arm):
    end=math.ceil(video["conversations"][qi]["end_time"]*.5)
    tail=list(range(max(0,end-4),end))
    base=list(range(end)) if arm.startswith("full_prefix_") else list(range(max(0,end-20),end))
    frames=list(base)
    if arm=="question_only": frames=[]
    elif arm=="recent_4_original": frames=tail
    elif arm=="candidate_support": frames={(204,0):[6,7],(204,3):[69]}.get((video["sample_id"],qi),[])
    elif arm.endswith("_fill"): frames=[0 if arm.endswith("_opening_fill") else tail[0]]*(len(base)-len(tail))+tail
    partition="original" if arm=="question_only" or arm.endswith("_original") else "framewise" if arm.endswith("_framewise") else "fixed4"
    replacement=arm.endswith("_fill")
    return dict(arm=arm,sampled_indices=frames,partition=partition,counterfactual=replacement,
        protected_tail_slots=list(range(max(0,len(frames)-4),len(frames))) if replacement else [],candidate_only=arm=="candidate_support",
        skip_reason="candidate_not_defined" if arm=="candidate_support" and not frames else "",
        intervention_effective=replacement and frames!=base)


def expected_groups(frames,partition,meta):
    if partition in ("fixed4","framewise"):
        count=4 if partition=="fixed4" else 1
        return [frames[i:i+count] for i in range(0,len(frames),count)]
    owner={i:c["file"] for c in meta["chunks"] for i in range(c["start"],c["stop"])}
    groups=[]
    for i in frames:
        if groups and i==groups[-1][-1]+1 and owner[i]==owner[groups[-1][-1]]: groups[-1].append(i)
        else: groups.append([i])
    return groups


def validate_row(row,video,qi,arm,meta):
    plan=expected_plan(video,qi,arm)
    frames=plan["sampled_indices"]
    assert row["plan"]==plan and row["sample_id"]==video["sample_id"] and row["method"]==arm
    assert row["visual_tokens"]==len(frames)*196
    assert row["source_frame_indices"]==[int(video["timestamps"][i]["source_frame"]) for i in frames]
    assert row["source_start_seconds"]==[float(video["timestamps"][i]["start_s"]) for i in frames]
    assert frames==sorted(frames) and all(type(i) is int and 0<=i<math.ceil(video["conversations"][qi]["end_time"]*.5) for i in frames)
    assert all(t<=video["conversations"][qi]["end_time"]+1e-6 for t in row["source_start_seconds"])
    assert row["input_identity"]==fingerprint(dict(features=fingerprint(meta),indices=frames))
    qa=video["conversations"][qi]
    identity=dict(sample_id=video["sample_id"],video_id=str(video["video_id"]),qa_index=qi,method=arm,
        question=qa["question"],choices=qa["choices"],gold_answer=qa["answer"],gold_choice="ABCDEFGH"[qa["choices"].index(qa["answer"])],
        query_time_s=qa["end_time"],task=qa["task"])
    assert all(row[k]==v for k,v in identity.items())
    for key in ("ingest_s","answer_s","wall_s","peak_allocated_gib"):
        assert math.isfinite(row[key]) and row[key]>=0
    if row["status"]=="answered":
        check_prediction(row,video,qi)
        assert not row["skip_reason"] and not plan["skip_reason"]
        assert row["prompt_tokens"]==13 and row["context_limit"]==32768 and row["question_tokens"]>0
        assert row["cache_lengths_before_answer"]==[13+len(frames)*196]*24
        assert row["protected_tail_token_starts"]==[13+i*196 for i in plan["protected_tail_slots"]]
        assert row["ingestion_groups"]==expected_groups(frames,plan["partition"],meta)
        assert row["peak_allocated_gib"]>0 and row["wall_s"]+1e-6>=row["ingest_s"]+row["answer_s"]
    else:
        assert row["status"]=="skipped" and row["correct"] is None and row["pred_answer"]==row["pred_choice"]==""
        if arm=="candidate_support": assert row["skip_reason"]=="candidate_not_defined" and not frames
        else: assert arm.startswith("full_prefix_") and (video["sample_id"],qi)==(208,4) and row["skip_reason"]=="configured_native_cap" and len(frames)*196>16384
    return identity


def audit(archive,notebook,output,tables):
    assert sha(archive.read_bytes())==ARCHIVE_SHA and sha(notebook.read_bytes())==NOTEBOOK_SHA
    with ZipFile(archive) as z: manifest,data=check_manifest(z)
    record=lambda name:json.loads(data[name])
    config,completion,reference=record("exports/run_config.json"),record("exports/completion.json"),record("reference.json")
    assert manifest["status"]==completion["status"]=="complete"
    assert config["config_hash"]==completion["config_hash"]==fingerprint({k:v for k,v in config.items() if k!="config_hash"})
    assert config["cases"]==[list(x) for x in CASES] and config["arms"]==list(ARMS)
    assert config["max_visual_tokens"]==16384 and config["no_compression"] is True and config["fresh_state"] is True
    assert config["model_revision"]=="74dd0bf867a4cda7950c17663794267c60cf4b40" and config["seed"]==2024
    assert config["torch"]=="2.10.0+cu128" and config["transformers"]=="4.45.0.dev0"
    assert sha(data["reference.json"])==config["reference_sha256"]
    local={"context_runner.py":"scripts/autodl_context_diagnostic.py","context_input_utils.py":"scripts/autodl_input_diagnostic.py","diagnostic_pilot_utils.py":"scripts/autodl_failure_pilot.py"}
    for name,path in local.items():
        assert sha(data[name])==config["sources"][name]
        ast.parse(data[name])
        assert data[name]==(ROOT/path).read_bytes().replace(b"\r\n",b"\n")
    with ZipFile(ROOT/"evidence/archives/HERMES_input_diagnostic_evidence.zip") as z: _,prior=check_manifest(z)
    old_reference=json.loads(prior["reference.json"])
    expected=old_reference.copy()
    ids={sid for sid,qi in CASES}
    expected["videos"]=[v for v in old_reference["videos"] if v["sample_id"] in ids]
    expected["predictions"]=[r for r in old_reference["predictions"] if int(r["sample_id"]) in ids]
    expected["input_archive_sha256"]="f184271bd8929f2ca5dead71a1ec60e4e1b555f33f2ed66d9909e3d7e0ffa1e8"
    expected["input_rows"]=[json.loads(blob)["row"] for name,blob in prior.items() if name.startswith("checkpoints/") and
        (json.loads(blob)["row"]["sample_id"],json.loads(blob)["row"]["qa_index"]) in CASES]
    assert reference==expected
    for path,expected_hash in reference["source_sha256"].items(): assert sha((ROOT/path).read_bytes().replace(b"\r\n",b"\n"))==expected_hash
    originals={str(v["video_id"]):v for v in json.loads((ROOT/"data/streamingbench/streamingbench_realtime.json").read_text(encoding="utf-8"))}
    videos={v["sample_id"]:v for v in reference["videos"]}
    assert len(videos)==5
    replay=record("exports/replay_annotations.json")
    assets=record("asset_manifest.json")
    assert assets["mode"]=="verified_existing_videos" and len(assets["videos"])==5
    for video in videos.values():
        sid=video["sample_id"]
        assert video["conversations"]==originals[str(video["video_id"])]["conversations"]
        assert next(v for v in replay if v["sample_id"]==sid)["conversations"]==video["conversations"]
        provenance=record(f"provenance/features_{sid}.json")
        assert provenance["origin"]=="verified_historical_cache" and provenance["metadata"]==video["feature_meta"]
        assert provenance["timestamp_rows"]==video["timestamps"] and provenance["feature_fingerprint"]==fingerprint(video["feature_meta"])
        assert next(v for v in assets["videos"] if v["sample_id"]==sid)["sha256"]==video["video_sha256"]
    launch=record("launch.json")
    assert launch["runner_sha256"]==config["sources"]["context_runner.py"] and launch["repository_revision"]=="2a1385e0abc94debb64761b0465713c799a81c1c"
    preflight=record("preflight.json")
    assert preflight["status"]=="pass" and preflight["feature_ingestion"]=="bitwise_equal_KV_and_positions" and preflight["source_cache_unchanged"] is True
    assert [r["cache_lengths"] for r in preflight["causal_scorer_comparisons"]]==[[3149]*24,[269]*24]
    assert all(r["max_attention_error"]==0 for r in preflight["causal_scorer_comparisons"])
    nb=json.loads(notebook.read_text(encoding="utf-8"))
    assert not [o for c in nb["cells"] for o in c.get("outputs",[]) if o.get("output_type")=="error"]
    embedded={}
    for cell in nb["cells"]:
        source="".join(cell["source"])
        if source.startswith("RUNNER_SOURCE ="):
            for node in ast.parse(source).body:
                if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ("RUNNER_SOURCE","INPUT_SOURCE","UTILS_SOURCE","REFERENCE_JSON"):
                    embedded[node.targets[0].id]=ast.literal_eval(node.value).encode()
    assert embedded==dict(RUNNER_SOURCE=data["context_runner.py"],INPUT_SOURCE=data["context_input_utils.py"],UTILS_SOURCE=data["diagnostic_pilot_utils.py"],REFERENCE_JSON=data["reference.json"])
    printed="\n".join("".join(o.get("text",[])) for cell in nb["cells"] for o in cell.get("outputs",[]))
    assert ARCHIVE_SHA in printed and "CONTEXT_COMPLETE" in printed
    rows=[];names=set()
    for sid,qi in CASES:
        video=videos[sid];meta=video["feature_meta"]
        for arm in ARMS:
            name=f"checkpoints/{arm}/sample_{sid}_q{qi}.json";names.add(name)
            value=record(name);row=value["row"]
            identity=validate_row(row,video,qi,arm,meta)
            assert value["row_sha256"]==fingerprint(row)
            assert value["identity"]==fingerprint(dict(config=config["config_hash"],question=identity,plan=row["plan"],features=fingerprint(meta)))
            rows.append(row)
    assert names=={name for name in data if name.startswith("checkpoints/")}
    assert completion["case_blocks"]==completion["planned_blocks"]==7 and completion["rows"]==84
    assert completion["answered"]==sum(r["status"]=="answered" for r in rows)==74
    assert math.isfinite(completion["elapsed_post_model_s"]) and completion["elapsed_post_model_s"]>=sum(r["wall_s"] for r in rows)
    by_key={(r["sample_id"],r["qa_index"],r["method"]):r for r in rows}
    csv_rows=table(data["exports/predictions.csv"])
    assert len(csv_rows)==len({(r["sample_id"],r["qa_index"],r["method"]) for r in csv_rows})==84
    compare_csv([by_key[int(r["sample_id"]),int(r["qa_index"]),r["method"]] for r in csv_rows],csv_rows)
    summary=[]
    for arm in ARMS:
        subset=[r for r in rows if r["method"]==arm];answered=[r for r in subset if r["status"]=="answered"]
        summary.append(dict(arm=arm,recorded=len(subset),answered=len(answered),skipped=len(subset)-len(answered),correct=sum(r["correct"] for r in answered)))
    compare_csv(summary,table(data["exports/summary.csv"]))
    mapping={"question_only":"question_only","recent_4_original":"recent_4","recent_20_original":"recent_20","full_prefix_original":"native_prefix"}
    old={(r["sample_id"],r["qa_index"],r["method"]):r for r in reference["input_rows"]}
    replay=[]
    for saved in table(data["exports/replay_checks.csv"]):
        sid,qi,arm=int(saved["sample_id"]),int(saved["qa_index"]),saved["arm"]
        x,y=by_key[sid,qi,arm],old[sid,qi,mapping[arm]]
        comparable=x["status"]==y["status"]=="answered"
        replay.append(dict(sample_id=sid,qa_index=qi,arm=arm,comparable=comparable,previous_choice=y["pred_choice"],new_choice=x["pred_choice"],choice_matches=x["pred_choice"]==y["pred_choice"] if comparable else ""))
    assert len(replay)==len({(r["sample_id"],r["qa_index"],r["arm"]) for r in replay})==28
    compare_csv(replay,table(data["exports/replay_checks.csv"]))
    contrasts=[]
    for sid,qi in CASES:
        for a,b in CONTRASTS:
            x,y=by_key[sid,qi,a],by_key[sid,qi,b]
            same=x["plan"]["sampled_indices"]==y["plan"]["sampled_indices"]
            answered=x["status"]==y["status"]=="answered"
            contrasts.append(dict(sample_id=sid,qa_index=qi,arm_a=a,arm_b=b,both_answered=answered,same_content=same,
                same_length=len(x["plan"]["sampled_indices"])==len(y["plan"]["sampled_indices"]),choice_changed=x["pred_choice"]!=y["pred_choice"] if answered else "",
                same_tail_sources=x["plan"]["sampled_indices"][-4:]==y["plan"]["sampled_indices"][-4:],replacement_effective=y["plan"]["intervention_effective"],
                a_correct=x["correct"],b_correct=y["correct"],interpretation="Identical-input null control" if same and x["plan"]["partition"]==y["plan"]["partition"] else
                "Chunking/numerical control" if same else "Counterfactual earlier-content replacement; repeated filler is a confound"))
    compare_csv(contrasts,table(data["exports/contrasts.csv"]))
    assert all(r["same_length"] and r["same_tail_sources"] for r in contrasts)
    log=data["captured_logs/context.log"].decode()
    assert log.count("ARM_DONE ")==84 and log.count("CASE_DONE ")==7 and log.count("FEATURE_REUSE_VERIFIED ")==5 and "\nREUSE " not in log
    assert sum(n.endswith(".jpg") for n in data)==27 and sum(n.endswith(".png") for n in data)==3
    pair_summary=[]
    for a,b in CONTRASTS:
        subset=[r for r in contrasts if r["arm_a"]==a and r["arm_b"]==b and r["both_answered"]]
        pair_summary.append(dict(arm_a=a,arm_b=b,matched_questions=len(subset),effective_replacements=sum(r["replacement_effective"] for r in subset),
            choice_changes=sum(r["choice_changed"] for r in subset),b_only_correct=sum(not r["a_correct"] and bool(r["b_correct"]) for r in subset),
            a_only_correct=sum(bool(r["a_correct"]) and not r["b_correct"] for r in subset)))
    cases=[]
    for sid,qi in CASES:
        row=dict(sample_id=sid,qa_index=qi,question=videos[sid]["conversations"][qi]["question"],gold=videos[sid]["conversations"][qi]["answer"])
        for arm in ARMS:
            r=by_key[sid,qi,arm];row.update({arm+"_choice":r["pred_choice"],arm+"_correct":r["correct"],arm+"_status":r["status"]})
        cases.append(row)
    tables.mkdir(parents=True,exist_ok=True)
    for source,target in [("summary.csv","summary.csv"),("predictions.csv","predictions.csv"),("replay_checks.csv","replay_checks.csv"),("contrasts.csv","contrasts.csv"),("report.md","run_report.md")]:
        (tables/target).write_bytes(data["exports/"+source])
    write_csv(tables/"audited_pair_summary.csv",pair_summary);write_csv(tables/"question_casebook.csv",cases)
    figures=tables/"figures";figures.mkdir(exist_ok=True)
    for source,target in [("review/sample_204_q0/candidate_sampled_0007_original.png","sample_204_q0_frame_7.png"),
        ("review/sample_204_q3/candidate_sampled_0069_original.png","sample_204_q3_frame_69.png"),
        ("review/sample_203_q2/page_04.jpg","sample_203_q2_last_page.jpg")]:
        (figures/target).write_bytes(data[source])
    result=dict(status="pass",archive_sha256=ARCHIVE_SHA,notebook_sha256=NOTEBOOK_SHA,payloads=len(data),case_blocks=7,checkpoints=84,answered=74,
        skip_reasons=dict(Counter(r["skip_reason"] for r in rows if r["status"]=="skipped")),methods=summary,
        comparable_replays=sum(r["comparable"] for r in replay),replay_choice_mismatches=sum(r["choice_matches"] is False for r in replay if r["comparable"]),
        contrast_rows=len(contrasts),answered_contrasts=sum(r["both_answered"] for r in contrasts),pair_summary=pair_summary,
        feature_reuse_videos=5,retained_notebook_errors=0,gpu_preflight="pass; zero maximum native attention error on checked prefix before/after pruning",
        review_pages=27,candidate_images=3,elapsed_post_model_s=completion["elapsed_post_model_s"],
        limitations=["Selected development cases; repeated arms and multiple contrasts are not independent test examples",
            "Counterfactual repeats preserve positions but alter content/repetition; filler neutrality and a unique internal mechanism are unestablished",
            "Two analyst support probes are not certified human-oracle or population results",
            "Saved evidence audit; raw videos/feature tensors and numerical GPU computation cannot be independently rerun from this ZIP",
            "No deployable selection policy, validated original method or independent confirmation"])
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8",newline="\n")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,default=ROOT/"evidence/archives/HERMES_context_diagnostic_evidence.zip")
    parser.add_argument("--notebook",type=Path,default=ROOT/"notebooks/executed/2026-11-20_hermes-context-diagnostic.ipynb")
    parser.add_argument("--output",type=Path,default=ROOT/"evidence/audits/context_diagnostic_2026-11-20_audit.json")
    parser.add_argument("--tables",type=Path,default=ROOT/"results/context_diagnostic_2026-11-20")
    args=parser.parse_args();print(json.dumps(audit(args.archive,args.notebook,args.output,args.tables),indent=2))


if __name__=="__main__": main()
