"""Small fresh-state context/content and chunk-partition interventions.

All replacements are declared counterfactuals using original pre-query features.
Candidate-support probes are analyst-selected, not certified human oracles.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

try:
    import context_input_utils as d
except ModuleNotFoundError as exc:
    if exc.name != "context_input_utils":
        raise
    import autodl_input_diagnostic as d

p = d.p
ARTIFACT = "HERMES_context_diagnostic_evidence.zip"
CASES = ((204,0),(203,2),(204,3),(209,0),(208,4),(202,0),(208,0))
ARMS = ("question_only", "recent_4_original", "recent_20_original", "full_prefix_original",
        "full_prefix_framewise", "recent_20_fixed4", "recent_20_opening_fill", "recent_20_local_fill",
        "full_prefix_fixed4", "full_prefix_opening_fill", "full_prefix_local_fill", "candidate_support")
SUPPORT_CANDIDATES = {(204,0):[6,7], (204,3):[69]}


def validate_indices(indices, video, qi):
    end = d.prefix_end(video,qi)
    if any(type(i) is not int or i < 0 or i >= end for i in indices):
        raise ValueError("Counterfactual contains an invalid/future sampled index")
    if indices != sorted(indices):
        raise ValueError("Input mapping must be nondecreasing; repeated fillers are explicit")
    query = float(video["conversations"][qi]["end_time"])
    if any(float(video["timestamps"][i]["start_s"]) > query+1e-6 for i in indices):
        raise ValueError("Source frame starts after query")


def arm_plan(video, qi, arm):
    if arm not in ARMS:
        raise ValueError("Unknown intervention arm")
    end = d.prefix_end(video,qi)
    recent = list(range(max(0,end-20),end))
    tail = list(range(max(0,end-4),end))
    if arm == "question_only":
        indices,partition = [],"original"
    elif arm == "candidate_support":
        indices,partition = list(SUPPORT_CANDIDATES.get((video["sample_id"],qi),[])),"fixed4"
    elif arm == "recent_4_original":
        indices,partition = tail,"original"
    else:
        indices = list(range(end)) if arm.startswith("full_prefix_") else recent
        partition = "original" if arm.endswith("_original") else "framewise" if arm.endswith("_framewise") else "fixed4"
        if arm.endswith("_fill"):
            filler = 0 if arm.endswith("_opening_fill") else tail[0]
            indices = [filler]*(len(indices)-len(tail)) + tail
    validate_indices(indices,video,qi)
    replacement = arm.endswith("_fill")
    return dict(arm=arm,sampled_indices=indices,partition=partition,counterfactual=replacement,
                protected_tail_slots=list(range(max(0,len(indices)-4),len(indices))) if replacement else [],
                candidate_only=arm == "candidate_support",
                skip_reason="candidate_not_defined" if arm == "candidate_support" and not indices else "",
                intervention_effective=replacement and indices != (list(range(end)) if arm.startswith("full_prefix_") else recent))


def input_groups(indices, partition, meta):
    if partition in ("fixed4","framewise"):
        size = 4 if partition == "fixed4" else 1
        return [indices[i:i+size] for i in range(0,len(indices),size)]
    slices = d.feature_slices(indices,meta["chunks"])
    by_file = {c["file"]:c for c in meta["chunks"]}
    return [list(range(by_file[c["file"]]["start"]+c["start"],by_file[c["file"]]["start"]+c["stop"])) for c in slices]


class FeatureBank:
    def __init__(self,directory,meta):
        import torch
        self.frames = {}
        for chunk in meta["chunks"]:
            value = torch.load(directory/chunk["file"],map_location="cpu",weights_only=True)
            count = chunk["stop"]-chunk["start"]
            if value.ndim != 3 or value.shape[:2] != (1,count*196) or value.dtype != torch.float16:
                raise ValueError("Feature layout differs from original per-frame token blocks")
            for offset,index in enumerate(range(chunk["start"],chunk["stop"])):
                if index in self.frames:
                    raise ValueError("Overlapping feature chunks")
                self.frames[index] = value[:,offset*196:(offset+1)*196]

    def get(self,indices):
        import torch
        return torch.cat([self.frames[i] for i in indices],dim=1).contiguous()


def valid_checkpoint(value,key,identity,plan):
    try:
        row = value["row"]
        return (value["identity"] == key and value["row_sha256"] == d.fingerprint(row)
            and all(row[k] == v for k,v in identity.items()) and row["plan"] == plan
            and ((row["status"] == "answered" and row["correct"] == int(row["pred_choice"] == identity["gold_choice"]))
              or (row["status"] == "skipped" and row["correct"] is None and bool(row["skip_reason"]))))
    except (KeyError,TypeError,ValueError):
        return False


def run_arm(model,analyzer,video,qi,arm,bank,meta,config_hash,args):
    import torch
    plan = arm_plan(video,qi,arm)
    identity = d.question_identity(video,qi,arm)
    key = d.fingerprint(dict(config=config_hash,question=identity,plan=plan,features=d.fingerprint(meta)))
    dest = args.work/"checkpoints"/arm/f"sample_{video['sample_id']}_q{qi}.json"
    if dest.is_file():
        value = json.loads(dest.read_text(encoding="utf-8"))
        if not valid_checkpoint(value,key,identity,plan):
            raise d.RunConflict("Checkpoint identity/correctness mismatch; preserve and choose new WORK")
        print(f"REUSE {arm} sample={video['sample_id']} q={qi}",flush=True)
        return value["row"]
    indices = plan["sampled_indices"]
    p.report_progress("context_arm",sample_id=video["sample_id"],qa_index=qi,arm=arm,input_slots=len(indices))
    row = dict(**identity,plan=plan,status="skipped",skip_reason=plan["skip_reason"],pred_answer="",pred_choice="",correct=None,
               visual_tokens=len(indices)*196,source_frame_indices=[int(video["timestamps"][i]["source_frame"]) for i in indices],
               source_start_seconds=[float(video["timestamps"][i]["start_s"]) for i in indices],
               input_identity=d.fingerprint(dict(features=d.fingerprint(meta),indices=indices)),
               ingest_s=0.0,answer_s=0.0,wall_s=0.0,peak_allocated_gib=0.0,cache_lengths_before_answer=[])
    started = time.perf_counter()
    if not row["skip_reason"]:
        random.seed(p.SEED)
        torch.manual_seed(p.SEED)
        p.reset(model)
        try:
            d.assert_fresh_cache(model,0)
            prompt = analyzer.format_mcqa_prompt(identity["question"],identity["choices"])
            qtokens = len(model.processor.tokenizer(prompt["prompt"]).input_ids)
            row["prompt_tokens"] = model.visual_start_idx
            row["question_tokens"] = qtokens
            row["context_limit"] = model.language_model.config.max_position_embeddings
            reason = d.context_reason(len(indices),model.visual_start_idx,qtokens,row["context_limit"],args.max_visual_tokens,
                                      "native_prefix" if arm.startswith("full_prefix_") else arm)
            if not reason and arm.startswith("full_prefix_"):
                base = sum(k.numel()*k.element_size()+v.numel()*v.element_size() for k,v in model.kv_cache)
                needed = base/model.visual_start_idx*(len(indices)*196+qtokens+48)
                if torch.cuda.mem_get_info()[0] < needed*1.5+2*1024**3:
                    reason = "estimated_vram_limit"
            if reason:
                row["skip_reason"] = reason
            else:
                groups = input_groups(indices,plan["partition"],meta)
                row["ingestion_groups"] = groups
                p.sync()
                torch.cuda.reset_peak_memory_stats()
                before = time.perf_counter()
                for group in groups:
                    p.ingest_features(model,bank.get(group),len(group))
                p.sync()
                row["ingest_s"] = time.perf_counter()-before
                d.assert_fresh_cache(model,len(indices))
                row["cache_lengths_before_answer"] = model._get_cache_seq_len_per_layer()
                row["protected_tail_token_starts"] = [model.visual_start_idx+i*196 for i in plan["protected_tail_slots"]]
                before = time.perf_counter()
                answer = model.question_answering(prompt,max_new_tokens=16)
                p.sync()
                prediction = analyzer.extract_characters_regex(answer)
                row.update(status="answered",pred_answer=answer.replace("\n",""),pred_choice=prediction,
                           correct=int(prediction == identity["gold_choice"]),answer_s=time.perf_counter()-before,
                           peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        except torch.cuda.OutOfMemoryError:
            if not arm.startswith("full_prefix_"):
                raise
            row.update(status="skipped",skip_reason="cuda_oom",correct=None)
        finally:
            model._layer_position_ids.clear()
            model.clear_cache()
    row["wall_s"] = time.perf_counter()-started
    value = dict(identity=key,row=row,row_sha256=d.fingerprint(row))
    if not valid_checkpoint(value,key,identity,plan):
        raise ValueError("New arm checkpoint failed validation")
    p.atomic_json(dest,value)
    print(f"ARM_DONE {arm} sample={video['sample_id']} q={qi} status={row['status']} correct={row['correct']} reason={row['skip_reason']}",flush=True)
    return row


def export(rows,reference,work):
    fields = ["sample_id","qa_index","method","question","choices","gold_answer","gold_choice","status","skip_reason","pred_choice","correct",
              "plan","input_identity","cache_lengths_before_answer","protected_tail_token_starts","ingest_s","answer_s","wall_s","peak_allocated_gib"]
    p.write_csv(work/"exports/predictions.csv",[{k:r.get(k,"") for k in fields} for r in rows])
    summary = []
    for arm in ARMS:
        cases = [r for r in rows if r["method"] == arm]
        answers = [r for r in cases if r["status"] == "answered"]
        summary.append(dict(arm=arm,recorded=len(cases),answered=len(answers),skipped=len(cases)-len(answers),correct=sum(r["correct"] for r in answers)))
    p.write_csv(work/"exports/summary.csv",summary)
    old = {(r["sample_id"],r["qa_index"],r["method"]):r for r in reference["input_rows"]}
    replay = []
    arm_mapping = {"question_only":"question_only","recent_4_original":"recent_4","recent_20_original":"recent_20","full_prefix_original":"native_prefix"}
    for row in rows:
        if row["method"] in arm_mapping:
            prior = old[row["sample_id"],row["qa_index"],arm_mapping[row["method"]]]
            comparable = row["status"] == prior["status"] == "answered"
            replay.append(dict(sample_id=row["sample_id"],qa_index=row["qa_index"],arm=row["method"],comparable=comparable,
                               previous_choice=prior["pred_choice"],new_choice=row["pred_choice"],
                               choice_matches=row["pred_choice"] == prior["pred_choice"] if comparable else ""))
    p.write_csv(work/"exports/replay_checks.csv",replay)
    by_key = {(r["sample_id"],r["qa_index"],r["method"]):r for r in rows}
    pairs = []
    contrasts = [("full_prefix_original","full_prefix_framewise"),("full_prefix_original","full_prefix_fixed4"),
                 ("recent_20_original","recent_20_fixed4"),("recent_20_fixed4","recent_20_opening_fill"),
                 ("recent_20_fixed4","recent_20_local_fill"),("full_prefix_fixed4","full_prefix_opening_fill"),
                 ("full_prefix_fixed4","full_prefix_local_fill")]
    for sid,qi in CASES:
        for a,b in contrasts:
            if (sid,qi,a) not in by_key or (sid,qi,b) not in by_key:
                continue
            x,y = by_key[sid,qi,a],by_key[sid,qi,b]
            answered = x["status"] == y["status"] == "answered"
            fixed_content = x["plan"]["sampled_indices"] == y["plan"]["sampled_indices"]
            fixed_positions = len(x["plan"]["sampled_indices"]) == len(y["plan"]["sampled_indices"])
            same_tail = x["plan"]["sampled_indices"][-4:] == y["plan"]["sampled_indices"][-4:]
            if b.endswith("_fill") and (not fixed_positions or not same_tail):
                raise ValueError("Counterfactual changed protected tail length/source positions")
            pairs.append(dict(sample_id=sid,qa_index=qi,arm_a=a,arm_b=b,both_answered=answered,
                same_content=fixed_content,same_length=fixed_positions,choice_changed=x["pred_choice"] != y["pred_choice"] if answered else "",
                same_tail_sources=same_tail,replacement_effective=y["plan"]["intervention_effective"],
                a_correct=x["correct"],b_correct=y["correct"],
                interpretation=("Identical-input null control" if fixed_content and x["plan"]["partition"] == y["plan"]["partition"]
                    else "Chunking/numerical control" if fixed_content else "Counterfactual earlier-content replacement; repeated filler is a confound")))
    if pairs:
        p.write_csv(work/"exports/contrasts.csv",pairs)
    text = ["# Context/partition diagnostic", "", "Seven selected development cases; not an algorithm evaluation or population accuracy estimate.",
            "", "| Arm | Answered | Skipped | Correct |", "|---|---:|---:|---:|"]
    text += [f"| {r['arm']} | {r['answered']} | {r['skipped']} | {r['correct']} |" for r in summary]
    mismatches = sum(r["choice_matches"] is False for r in replay if r["comparable"])
    text += ["",f"Historical replay choice mismatches: {mismatches}. Inspect mismatches before interpreting interventions.",
        "Inspect exact-content chunking controls before attributing differences to semantics.",
        "Replacement arms keep last-four source frames and their token positions fixed at equal input length; they repeat pre-query fillers, which are not certified neutral.",
        "When there are no earlier slots, replacement arms are identical-input controls, not independent evidence.",
        "Candidate support at 204/q0 [6,7] and 204/q3 [69] comes from analyst-reviewed saved sheets; no human oracle certification is claimed.",
        "Gold is used only for scoring. All answers reset state and use no compression/past answers. Resource-limited full prefixes are skipped without truncation.",
        "Question wall times exclude model/setup/shared preparation. No publication-ready improvement is implied."]
    (work/"exports/report.md").write_text("\n".join(text)+"\n",encoding="utf-8")


def export_review(video,qi,work):
    from decord import VideoReader
    from PIL import Image,ImageDraw
    reader = VideoReader(video["video_path"],num_threads=1)
    directory = work/"review"/f"sample_{video['sample_id']}_q{qi}"
    directory.mkdir(parents=True,exist_ok=True)
    end = d.prefix_end(video,qi)
    for start in range(0,end,12):
        target = directory/f"page_{start//12+1:02d}.jpg"
        if target.exists():
            continue
        chosen = list(range(start,min(start+12,end)))
        frames = reader.get_batch([int(video["timestamps"][i]["source_frame"]) for i in chosen]).asnumpy()
        canvas = Image.new("RGB",(1024,520),"white")
        draw = ImageDraw.Draw(canvas)
        draw.text((4,4),f"sample={video['sample_id']} q={qi} | ORIGINAL PREFIX: no filler frames | support review pending",fill="black")
        for n,(frame,index) in enumerate(zip(frames,chosen)):
            picture = Image.fromarray(frame)
            picture.thumbnail((256,140))
            x,y = n%4*256,25+n//4*164
            canvas.paste(picture,(x,y))
            draw.text((x+3,y+143),f"sampled {index} | {video['timestamps'][index]['start_s']}s",fill="black")
        canvas.save(target,quality=85)
    for index in SUPPORT_CANDIDATES.get((video["sample_id"],qi),[]):
        target = directory/f"candidate_sampled_{index:04d}_original.png"
        if not target.exists():
            source = int(video["timestamps"][index]["source_frame"])
            Image.fromarray(reader[source].asnumpy()).save(target)


def package(work,status):
    captured = work/"captured_logs"
    captured.mkdir(exist_ok=True,parents=True)
    for source in (work/"logs").glob("*"):
        if source.is_file():
            shutil.copy2(source,captured/source.name)
    if (work/"progress.json").is_file():
        shutil.copy2(work/"progress.json",captured/"progress_snapshot.json")
    files = []
    for area in ("exports","checkpoints","provenance","review","captured_logs"):
        files += [x for x in (work/area).rglob("*") if x.is_file() and ".ipynb_checkpoints" not in x.parts]
    for name in ("context_runner.py","context_input_utils.py","diagnostic_pilot_utils.py","reference.json","launch.json","asset_manifest.json","preflight.json"):
        if (work/name).exists():
            files.append(work/name)
    manifest = [dict(path=x.relative_to(work).as_posix(),bytes=x.stat().st_size,sha256=p.digest(x)) for x in sorted(set(files))]
    p.atomic_json(work/"evidence_manifest.json",dict(status=status,files=manifest))
    target = work/ARTIFACT
    with ZipFile(target.with_suffix(".tmp"),"w",compression=ZIP_DEFLATED) as z:
        for row in manifest:
            z.write(work/row["path"],row["path"])
        z.write(work/"evidence_manifest.json","evidence_manifest.json")
    os.replace(target.with_suffix(".tmp"),target)
    print(f"EVIDENCE_ZIP={target}\nSHA256={p.digest(target)}",flush=True)


def execute(args):
    import numpy as np
    import torch
    import transformers
    reference = json.loads((args.work/"reference.json").read_text(encoding="utf-8"))
    d.check_reference(reference,args.repo)
    if args.model.name != p.MODEL_SHA or torch.__version__ != "2.10.0+cu128" or transformers.__version__ != "4.45.0.dev0" or not torch.cuda.is_available():
        raise ValueError("Pinned model/CUDA environment required")
    config = dict(reference_sha256=p.digest(args.work/"reference.json"),sources={name:p.digest(args.work/name) for name in
        ("context_runner.py","context_input_utils.py","diagnostic_pilot_utils.py")},cases=CASES,arms=ARMS,
        max_visual_tokens=args.max_visual_tokens,seed=p.SEED,model_revision=p.MODEL_SHA,
        torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(0),
        no_compression=True,fresh_state=True,candidate_support="Analyst sheet candidates, not certified oracle")
    config_hash = d.fingerprint(config)
    path = args.work/"exports/run_config.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8"))["config_hash"] != config_hash:
        raise d.RunConflict("Run configuration changed; use a new WORK directory")
    p.atomic_json(path,dict(**config,config_hash=config_hash))
    previous_failure = args.work/"exports/failure.json"
    if previous_failure.exists():
        destination = args.work/"logs"/f"previous_failure_{time.time_ns()}.json"
        destination.parent.mkdir(exist_ok=True,parents=True)
        os.replace(previous_failure,destination)
    videos = copy.deepcopy(reference["videos"])
    d.acquire_videos(videos,args.reuse,args.work)
    p.atomic_json(args.work/"exports/replay_annotations.json",videos)
    sys.path.insert(0,str(args.repo))
    from inference.llavaov_hermes import load_model
    from video_qa.hermes_vqa import HermesVQA
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(p.SEED)
    np.random.seed(p.SEED)
    torch.manual_seed(p.SEED)
    p.report_progress("loading_model")
    model,processor = load_model(model_path=str(args.model),kv_size=4000,streaming=True,sample_fps=p.FPS)
    model.eval()
    analyzer = HermesVQA(anno=[],save_dir=str(args.work),sample_fps=p.FPS,qa_model=model,qa_processor=processor)
    by_id = {v["sample_id"]:v for v in videos}
    rows,assets = [],{}
    started = time.perf_counter()
    completed = 0
    with torch.inference_mode():
        p.preflight(model,by_id[204],args.work)
        for index,(sid,qi) in enumerate(CASES):
            if index and args.max_hours and time.perf_counter()-started > args.max_hours*3600:
                break
            video = by_id[sid]
            if sid not in assets:
                directory,meta,_ = d.acquire_features(model,video,args.reuse,args.work,d.fingerprint(dict(config=config_hash,video=video["video_sha256"])))
                assets[sid] = (FeatureBank(directory,meta),meta)
            bank,meta = assets[sid]
            order = list(ARMS)
            random.Random(p.SEED+sid*10+qi).shuffle(order)
            for arm in order:
                rows.append(run_arm(model,analyzer,video,qi,arm,bank,meta,config_hash,args))
            export_review(video,qi,args.work)
            export(rows,reference,args.work)
            completed += 1
            elapsed = time.perf_counter()-started
            print(f"CASE_DONE {completed}/{len(CASES)} elapsed_min={elapsed/60:.2f} rough_remaining_min={(len(CASES)-completed)*elapsed/completed/60:.2f}",flush=True)
    status = "complete" if completed == len(CASES) else "partial_time_budget"
    if status == "complete":
        if len(rows) != len(CASES)*len(ARMS) or any(r["status"] != "answered" for r in rows if r["method"] in ARMS[:3]):
            raise ValueError("Core replay control matrix incomplete")
    p.atomic_json(args.work/"exports/completion.json",dict(status=status,case_blocks=completed,planned_blocks=len(CASES),
        rows=len(rows),answered=sum(r["status"] == "answered" for r in rows),config_hash=config_hash,
        elapsed_post_model_s=time.perf_counter()-started,timing_scope="Post-model preflight, verified feature work, arms, review/export; excludes setup/download/model load/final packaging"))
    p.report_progress(status,case_blocks=completed)
    package(args.work,status)
    print("CONTEXT_COMPLETE" if status == "complete" else "CONTEXT_PARTIAL",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo","work","model","reuse"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--max-visual-tokens",type=int,default=16384)
    parser.add_argument("--max-hours",type=float,default=.75)
    args = parser.parse_args()
    for name in ("repo","work","model","reuse"):
        setattr(args,name,getattr(args,name).resolve())
    d.validate_workspaces(args.work,args.reuse)
    if args.max_hours < 0 or args.max_visual_tokens < 3920:
        parser.error("Invalid budget/context cap")
    with p.ProgressMonitor(args.work):
        try:
            execute(args)
        except d.RunConflict:
            raise
        except Exception as exc:
            p.atomic_json(args.work/"exports/failure.json",dict(status="failed",type=type(exc).__name__,detail=str(exc)[-1500:]))
            p.atomic_json(args.work/"exports/completion.json",dict(status="failed"))
            package(args.work,"failed_partial")
            raise


if __name__ == "__main__":
    main()
