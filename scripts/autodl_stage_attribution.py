"""Factor history visibility during recent encoding and question/answer readout.

No production inference files are changed. Masking H during R construction is
validated against physically independent R construction at identical RoPE positions.
"""
import argparse
import contextlib
import copy
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED,ZipFile

try:
    import stage_context_utils as c
except ModuleNotFoundError as exc:
    if exc.name != "stage_context_utils": raise
    import autodl_context_diagnostic as c

d,p=c.d,c.p
ARMS=("A_history_both","B_independent_recent","C_hide_history_readout","D_independent_hide_readout","compact_recent4")
ARTIFACT="HERMES_stage_attribution_evidence.zip"


def split_frames(video,qi,long_prefix=False):
    end=d.prefix_end(video,qi)
    selected=list(range(0 if long_prefix else max(0,end-20),end))
    recent=selected[-4:]
    history=selected[:-len(recent)] if recent else selected
    d.validate_frames(selected,video,qi,video["timestamps"])
    return history,recent


def cache_length(cache):
    if cache is None:return 0
    if hasattr(cache,"get_seq_length"):return int(cache.get_seq_length())
    return int(cache[0][0].shape[2])


def causal_mask(batch,q,past,dtype,device,blocked_start,blocked_stop):
    import torch
    if not 0<=blocked_start<=blocked_stop<=past:
        raise ValueError("History mask span must lie entirely in physical past cache")
    mask=torch.zeros((batch,1,q,past+q),dtype=dtype,device=device)
    minimum=torch.finfo(dtype).min
    mask[:,:,:,blocked_start:blocked_stop]=minimum
    future=torch.ones((q,q),dtype=torch.bool,device=device).triu(1)
    mask[:,:,:,past:].masked_fill_(future,minimum)
    return mask


class HistoryMask:
    """Apply one explicit causal+H mask to every LM call, including all decode steps."""
    def __init__(self,model,start,stop,force=False):
        self.model,self.start,self.stop,self.force=model,start,stop,force
        self.events=[];self.handle=None

    def hook(self,module,args,kwargs):
        if args:raise ValueError("Expected named LM inputs for audited mask")
        if kwargs.get("attention_mask") is not None:raise ValueError("Refuse to overwrite another attention mask")
        value=kwargs.get("inputs_embeds")
        if value is None:value=kwargs["input_ids"]
        batch,q=value.shape[:2]
        past=cache_length(kwargs.get("past_key_values"))
        mask=causal_mask(batch,q,past,self.model.dtype,self.model.device,self.start,self.stop)
        kwargs["attention_mask"]=mask
        self.events.append(dict(query_tokens=q,past_tokens=past,blocked_start=self.start,blocked_stop=self.stop,
                               shape=list(mask.shape),policy="causal plus hidden historical physical span"))
        return args,kwargs

    def __enter__(self):
        if self.stop>self.start or self.force:
            self.handle=self.model.language_model.register_forward_pre_hook(self.hook,with_kwargs=True)
        return self

    def __exit__(self,*exc):
        if self.handle is not None:self.handle.remove()
        return False


def snapshot(model):
    return dict(kv=[(k.clone(),v.clone()) for k,v in model.kv_cache],positions=[x.clone() for x in model._position_ids_cache],
                prompt=model.visual_start_idx)


def restore(model,state):
    model._layer_position_ids.clear()
    model.kv_cache=[(k.clone(),v.clone()) for k,v in state["kv"]]
    model._position_ids_cache=[x.clone() for x in state["positions"]]
    model.visual_start_idx=state["prompt"]
    model.conv_history=[]
    model.token_activity_cache=[None]*model.num_layers


def prefix_state(state,length):
    return dict(kv=[(k[:,:,:length].contiguous().clone(),v[:,:,:length].contiguous().clone()) for k,v in state["kv"]],
                positions=[x[:length].clone() for x in state["positions"]],prompt=state["prompt"])


def drop_history(state,start,stop):
    import torch
    n=state["kv"][0][0].shape[2]
    keep=torch.cat((torch.arange(start,device=state["positions"][0].device),torch.arange(stop,n,device=state["positions"][0].device)))
    return dict(kv=[(k.index_select(2,keep),v.index_select(2,keep)) for k,v in state["kv"]],
                positions=[x.index_select(0,keep) for x in state["positions"]],prompt=state["prompt"])


def assert_state(model,state):
    import torch
    for (k,v),(expected_k,expected_v),pos,expected_pos in zip(model.kv_cache,state["kv"],model._position_ids_cache,state["positions"]):
        torch.testing.assert_close(k,expected_k,atol=0,rtol=0)
        torch.testing.assert_close(v,expected_v,atol=0,rtol=0)
        torch.testing.assert_close(pos,expected_pos,atol=0,rtol=0)


def ingest(model,bank,indices,meta,logical_start=None):
    groups=c.input_groups(indices,"original",meta)
    for group in groups:
        if logical_start is None:
            p.ingest_features(model,bank.get(group),len(group))
        else:
            # Native independent reference: physical cache omits H while R's logical positions retain its gap.
            original=model._get_next_start_pos_per_layer
            try:
                model._get_next_start_pos_per_layer=lambda:[logical_start]*model.num_layers
                p.ingest_features(model,bank.get(group),len(group))
            finally:model._get_next_start_pos_per_layer=original
            logical_start+=len(group)*196
    return groups


def prepare_states(model,bank,meta,history,recent):
    import torch
    p.reset(model)
    full_groups=ingest(model,bank,history+recent,meta)
    a=snapshot(model)
    d.assert_fresh_cache(model,len(history)+len(recent))
    prompt=model.visual_start_idx
    stop=prompt+len(history)*196
    if history:
        restore(model,prefix_state(a,stop))
        with HistoryMask(model,prompt,stop) as mask:
            recent_groups=ingest(model,bank,recent,meta)
        b=snapshot(model)
        d.assert_fresh_cache(model,len(history)+len(recent))
        for (ak,av),(bk,bv) in zip(a["kv"],b["kv"]):
            torch.testing.assert_close(ak[:,:,:stop],bk[:,:,:stop],atol=0,rtol=0)
            torch.testing.assert_close(av[:,:,:stop],bv[:,:,:stop],atol=0,rtol=0)
        events=mask.events
    else:
        b=a;recent_groups=full_groups;events=[]
    p.reset(model)
    compact_groups=ingest(model,bank,recent,meta)
    compact=snapshot(model)
    d.assert_fresh_cache(model,len(recent))
    return dict(A=a,B=b,compact=compact),dict(history_stop=stop,prompt_tokens=prompt,full_groups=full_groups,
        independent_recent_groups=recent_groups,independent_encoding_mask_trace=events,compact_groups=compact_groups,
        state_bytes={name:sum(k.numel()*k.element_size()+v.numel()*v.element_size() for k,v in state["kv"]) for name,state in [("A",a),("B",b),("compact",compact)]},
        note="A/B/compact control snapshots are prepared once per question and held simultaneously; answer-arm timing is not standalone system cost")


def probe(model,state,ids,start,stop,force=False,steps=2,attentions=False):
    import torch
    restore(model,state)
    cache=model.kv_cache
    logical=int(state["positions"][0][-1])+1
    logits=[];weights=[]
    with HistoryMask(model,start,stop,force=force) as mask:
        try:
            for step in range(steps+1):
                current=ids if step==0 else ids[:,:1]
                n=current.shape[1]
                model._set_rotary_required_len(logical+n)
                for layer in range(model.num_layers):model._layer_position_ids[layer]=model._build_position_ids(logical,n,current.shape[0])
                output=model.language_model(input_ids=current,past_key_values=cache,position_ids=model._layer_position_ids[0],
                                            use_cache=True,output_attentions=attentions,return_dict=True)
                cache=output.past_key_values
                logits.append(output.logits.detach().clone())
                if attentions:weights.append(output.attentions)
                logical+=n
        finally:model._layer_position_ids.clear()
    return logits,weights,mask.events


def error_statistics(actual,expected,atol,rtol):
    import torch
    a,b=actual.float(),expected.float()
    if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Nonfinite or shape-mismatched preflight tensors")
    delta=(a-b).abs()
    return dict(elements=a.numel(),max_abs=float(delta.max()),rms=float(delta.square().mean().sqrt()),
                relative_l2=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-12)),
                exceeds_original_005_bound=int((delta>(.005+.005*b.abs())).sum()),
                exceeds_active_bound=int((delta>(atol+rtol*b.abs())).sum()))


class PrecisionProxy:
    """Override only public dtype; all state/methods stay on the original model."""
    def __init__(self,model,dtype):
        object.__setattr__(self,"_wrapped",model);object.__setattr__(self,"_dtype",dtype)
    def __getattr__(self,name):
        return object.__getattribute__(self,"_dtype") if name=="dtype" else getattr(object.__getattribute__(self,"_wrapped"),name)
    def __setattr__(self,name,value):setattr(object.__getattribute__(self,"_wrapped"),name,value)


@contextlib.contextmanager
def fp32_structural_reference(model):
    """Temporarily promote LM parameters, preserving original mixed buffer dtypes/values.

    Vision weights are never cast. Cached rotary buffers are restored exactly,
    rather than blindly calling half() on float32 frequency buffers afterward.
    """
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    lm=model.language_model
    parameter_dtypes={name:value.dtype for name,value in lm.named_parameters()}
    buffers={name:value.clone() for name,value in lm.named_buffers()}
    rotary=[(layer.self_attn.rotary_emb,layer.self_attn.rotary_emb.max_seq_len_cached,
             getattr(layer.self_attn.rotary_emb,"_required_seq_len",0)) for layer in lm.model.layers]
    try:
        for value in lm.parameters():
            if value.is_floating_point() and value.dtype not in (torch.float16,torch.bfloat16,torch.float32):
                raise ValueError("Unexpected parameter dtype for reversible FP32 reference")
        model.clear_cache()
        for value in lm.parameters():
            if value.is_floating_point():value.data=value.data.float()
        # Both reference paths use the same high-precision math implementation.
        with sdpa_kernel(SDPBackend.MATH):yield PrecisionProxy(model,torch.float32)
    finally:
        for name,value in lm.named_parameters():value.data=value.data.to(parameter_dtypes[name])
        for name,value in buffers.items():
            parent,_,leaf=name.rpartition(".")
            setattr(lm.get_submodule(parent) if parent else lm,leaf,value)
        for module,length,required in rotary:
            module.max_seq_len_cached=length;module._required_seq_len=required
        model.clear_cache();model.kv_cache=None
        model._layer_position_ids.clear();model._position_ids_cache=[None]*model.num_layers
        model.token_activity_cache=[None]*model.num_layers


def _preflight_pass(model,bank,meta,ids,atol=.02,rtol=.005,kv_atol=.005,kv_rtol=.005,diagnostics=None,save=None):
    """Check physical masking against independent construction and compact readout."""
    import torch
    p.reset(model);sink=snapshot(model);s=sink["prompt"]
    states,detail=prepare_states(model,bank,meta,[0],[1])
    stop=s+196
    restore(model,sink)
    ingest(model,bank,[1],meta,logical_start=stop)
    independent=snapshot(model)
    construction_error=0.0
    comparisons=[]
    for layer,((fk,fv),(ik,iv)) in enumerate(zip(states["B"]["kv"],independent["kv"])):
        for kind,actual,expected in (("K",fk[:,:,stop:],ik[:,:,s:]),("V",fv[:,:,stop:],iv[:,:,s:])):
            comparisons.append(dict(layer=layer,kind=kind,**error_statistics(actual,expected,kv_atol,kv_rtol)))
    if diagnostics is not None:
        diagnostics.update(dtype=str(model.dtype),kv_atol=kv_atol,kv_rtol=kv_rtol,construction_by_layer=comparisons)
        if save:save()
    for layer,((fk,fv),(ik,iv)) in enumerate(zip(states["B"]["kv"],independent["kv"])):
        for kind,actual,expected in (("K",fk[:,:,stop:],ik[:,:,s:]),("V",fv[:,:,stop:],iv[:,:,s:])):
            torch.testing.assert_close(actual,expected,atol=kv_atol,rtol=kv_rtol,
                                       msg=lambda msg: f"Construction preflight layer {layer} {kind}: {msg}")
            construction_error=max(construction_error,float((actual-expected).abs().max()))
    records=[]
    plain,_,_=probe(model,states["A"],ids,0,0)
    explicit,_,all_events=probe(model,states["A"],ids,0,0,force=True)
    for name,left,right in [("all_visible_explicit_causal_matches_native",plain,explicit)]:
        maximum=0.0
        for a,b in zip(left,right):
            torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
            assert torch.equal(a[:,-1].argmax(-1),b[:,-1].argmax(-1)),"Preflight argmax changed"
            maximum=max(maximum,float((a-b).abs().max()))
        records.append(dict(check=name,max_logit_error=maximum,argmax_equal=True))
    for label,state in (("contextual_recent",states["A"]),("independent_recent",states["B"])):
        masked,_,events=probe(model,state,ids,s,stop)
        compact,_,_=probe(model,drop_history(state,s,stop),ids,0,0)
        maximum=0.0
        for a,b in zip(masked,compact):
            torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
            assert torch.equal(a[:,-1].argmax(-1),b[:,-1].argmax(-1)),"Masked/compact preflight argmax changed"
            maximum=max(maximum,float((a-b).abs().max()))
        assert len(events)==3 and [e["query_tokens"] for e in events][1:]==[1,1]
        records.append(dict(check=label+"_masked_matches_physically_removed_history",max_logit_error=maximum,
                            argmax_equal=True,query_and_decode_mask_calls=len(events)))
    _,weights,events=probe(model,states["B"],ids,s,stop,attentions=True)
    for step,layers in enumerate(weights):
        for attention in layers:
            assert not torch.any(attention[...,s:stop]),"Historical attention leaked"
            q=attention.shape[-2]
            assert not torch.any(torch.triu(attention[...,-q:],diagonal=1)),"Future attention leaked"
    # Query path must leave the stored control source unchanged.
    restore(model,states["A"])
    source=snapshot(model)
    probe(model,states["A"],ids,s,stop)
    for (a,b),(x,y) in zip(states["A"]["kv"],source["kv"]):
        torch.testing.assert_close(a,x,atol=0,rtol=0);torch.testing.assert_close(b,y,atol=0,rtol=0)
    model.clear_cache()
    return dict(status="pass",history_tokens=196,recent_tokens=196,prompt_tokens=s,
        independent_construction_max_kv_error=construction_error,comparisons=records,
        all_layer_hidden_history_attention_zero=True,all_layer_future_attention_zero=True,
        source_snapshots_unchanged=True,attention_checks_steps=len(events),layers=model.num_layers,
        tolerances=dict(kv_atol=kv_atol,kv_rtol=kv_rtol,logits_atol=atol,logits_rtol=rtol),
        scope="One two-frame visual prefix; explicit history mask at question prefill and two forced decode steps; not every production prefix")


def numerical_preflight(model,bank,meta,ids,atol=.02,rtol=.005,diagnostic_path=None):
    import torch
    diagnostics=dict(policy="v2_strict_fp32_structure_then_fp16_production",fp32={},production={})
    def save():
        if diagnostic_path is not None:p.atomic_json(diagnostic_path,diagnostics)
    if model.dtype in (torch.float16,torch.bfloat16):
        print("STAGE_PREFLIGHT strict FP32 structural reference (math attention)",flush=True)
        with fp32_structural_reference(model) as reference:
            structural=_preflight_pass(reference,bank,meta,ids,atol=.001,rtol=.001,kv_atol=.0001,kv_rtol=.0001,
                                       diagnostics=diagnostics["fp32"],save=save)
        diagnostics["fp32"]["result"]=structural;save()
        print("STAGE_PREFLIGHT restored production precision; checking FP16 numerical bounds",flush=True)
        result=_preflight_pass(model,bank,meta,ids,atol=atol,rtol=rtol,kv_atol=.01,kv_rtol=.01,
                               diagnostics=diagnostics["production"],save=save)
        result["fp32_structural_reference"]=structural
        result["production_precision_restored"]=True
    else:
        result=_preflight_pass(model,bank,meta,ids,atol=atol,rtol=rtol,diagnostics=diagnostics["production"],save=save)
    result["validation_policy"]=diagnostics["policy"]
    result["numerical_limits"]="FP32 checks mask and position equivalence. Production precision has a separate K/V tolerance. Zero hidden-history/future attention, argmax and per-question replay gates are mandatory."
    diagnostics["production"]["result"]=result;save()
    return result


def checkpoint_valid(value,key,identity,arm,history,recent):
    try:
        row=value["row"]
        return (value["identity"]==key and value["row_sha256"]==d.fingerprint(row) and all(row[k]==v for k,v in identity.items())
            and row["method"]==arm and row["history_indices"]==history and row["recent_indices"]==recent
            and ((row["status"]=="answered" and row["correct"]==int(row["pred_choice"]==row["gold_choice"]))
                 or (row["status"]=="skipped" and row["correct"] is None and bool(row["skip_reason"]))))
    except (KeyError,ValueError,TypeError):return False


def run_answer(model,analyzer,video,qi,arm,scope,states,detail,history,recent,feature_hash,config_hash,args,old_choice=None):
    import torch
    identity=d.question_identity(video,qi,arm)
    key=d.fingerprint(dict(config=config_hash,scope=scope,question=identity,history=history,recent=recent,features=feature_hash))
    dest=args.work/"checkpoints"/scope/arm/f"sample_{video['sample_id']}_q{qi}.json"
    if dest.is_file():
        value=json.loads(dest.read_text(encoding="utf-8"))
        if not checkpoint_valid(value,key,identity,arm,history,recent):raise d.RunConflict("Stage checkpoint differs; preserve and use new WORK")
        if value["row"].get("reference_choice_matches") is False:
            raise ValueError("Previously saved replay mismatch remains unresolved; do not bypass the attribution gate")
        print(f"REUSE {scope} {arm} sample={video['sample_id']} q={qi}",flush=True)
        return value["row"]
    construction=arm in (ARMS[1],ARMS[3])
    hidden=arm in (ARMS[2],ARMS[3])
    state=states["compact" if arm==ARMS[4] else "B" if construction else "A"]
    s=state["prompt"];stop=s+len(history)*196
    p.report_progress("stage_answer",sample_id=video["sample_id"],qa_index=qi,arm=arm,scope=scope)
    row=dict(**identity,scope=scope,history_indices=history,recent_indices=recent,history_empty=not history,
        independent_construction=construction,history_readout_hidden=hidden,factor_cell=arm!=ARMS[4],
        history_physically_present=bool(history) and arm!=ARMS[4],status="answered",skip_reason="",
        feature_fingerprint=feature_hash,cache_lengths=[x[0].shape[2] for x in state["kv"]],
        recent_logical_start=s+(0 if arm==ARMS[4] else len(history)*196),
        recent_physical_start=s+(0 if arm==ARMS[4] else len(history)*196),
        actual_visual_indices=recent if arm==ARMS[4] else history+recent,mask_trace=[],
        query_logical_start=int(state["positions"][0][-1])+1,
        pred_choice="",pred_answer="",correct=None,answer_s=0.0,answer_wall_s=0.0,peak_allocated_gib=0.0,
        reference_choice=old_choice if old_choice is not None else "",reference_choice_matches="")
    started=time.perf_counter()
    restore(model,state)
    if model.conv_history:raise ValueError("Previous answer history leaked")
    random.seed(p.SEED);torch.manual_seed(p.SEED)
    torch.cuda.reset_peak_memory_stats()
    try:
        prompt=analyzer.format_mcqa_prompt(identity["question"],identity["choices"])
        qlen=len(model.processor.tokenizer(prompt["prompt"]).input_ids)
        context=model.language_model.config.max_position_embeddings
        if int(state["positions"][0][-1])+1+qlen+48>context:
            if scope=="main":raise ValueError("Main context limit exceeded")
            row.update(status="skipped",skip_reason="context_limit",correct=None)
        else:
            p.sync();before=time.perf_counter()
            with HistoryMask(model,s,stop if hidden else s) as mask:
                answer=model.question_answering(prompt,max_new_tokens=16)
            p.sync();row["answer_s"]=time.perf_counter()-before
            row["mask_trace"]=mask.events
            if hidden and history:
                assert mask.events and all(e["blocked_start"]==s and e["blocked_stop"]==stop for e in mask.events)
                assert all(e["query_tokens"]==1 for e in mask.events[1:])
            assert_state(model,state)
            choice=analyzer.extract_characters_regex(answer)
            row.update(pred_choice=choice,pred_answer=answer.replace("\n",""),correct=int(choice==identity["gold_choice"]),
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
            if old_choice is not None:row["reference_choice_matches"]=choice==old_choice
    finally:model._layer_position_ids.clear();model.clear_cache()
    row["answer_wall_s"]=time.perf_counter()-started
    value=dict(identity=key,row=row,row_sha256=d.fingerprint(row))
    if not checkpoint_valid(value,key,identity,arm,history,recent):raise ValueError("Stage checkpoint failed validation")
    p.atomic_json(dest,value)
    print(f"STAGE_DONE {scope} {arm} sample={video['sample_id']} q={qi} correct={row['correct']} replay={row['reference_choice_matches']}",flush=True)
    if row["reference_choice_matches"] is False:
        raise ValueError("No-op/compact historical replay mismatch: evidence saved; stop attribution and audit numerics")
    return row


def export(rows,work):
    fields=["scope","sample_id","qa_index","method","question","choices","gold_choice","pred_choice","correct","status","skip_reason",
        "history_indices","recent_indices","actual_visual_indices","history_empty","factor_cell","history_physically_present",
        "independent_construction","history_readout_hidden","cache_lengths","query_logical_start",
        "recent_logical_start","mask_trace","reference_choice","reference_choice_matches","answer_s","answer_wall_s","peak_allocated_gib"]
    p.write_csv(work/"exports/predictions.csv",[{k:r.get(k,"") for k in fields} for r in rows])
    summary=[]
    for scope in ("main","long_204_q3"):
        for arm in (ARMS if scope=="main" else ARMS[:4]):
            subset=[r for r in rows if r["scope"]==scope and r["method"]==arm]
            if not subset:continue
            answers=[r for r in subset if r["status"]=="answered"]
            summary.append(dict(scope=scope,arm=arm,recorded=len(subset),answered=len(answers),correct=sum(r["correct"] for r in answers),
                history_empty_questions=sum(r["history_empty"] for r in answers)))
    p.write_csv(work/"exports/summary.csv",summary)
    pairs=[]
    keys={(r["scope"],r["sample_id"],r["qa_index"]):{} for r in rows}
    for r in rows:keys[r["scope"],r["sample_id"],r["qa_index"]][r["method"]]=r
    for a,b in [(ARMS[0],ARMS[1]),(ARMS[0],ARMS[2]),(ARMS[1],ARMS[3]),(ARMS[2],ARMS[3]),(ARMS[3],ARMS[4])]:
        for scope in ("main","long_204_q3"):
            matched=[(v[a],v[b]) for k,v in keys.items() if k[0]==scope and a in v and b in v and v[a]["status"]==v[b]["status"]=="answered"]
            if matched:pairs.append(dict(scope=scope,arm_a=a,arm_b=b,questions=len(matched),a_only_correct=sum(bool(x["correct"]) and not y["correct"] for x,y in matched),
                b_only_correct=sum(not x["correct"] and bool(y["correct"]) for x,y in matched),choice_changes=sum(x["pred_choice"]!=y["pred_choice"] for x,y in matched),
                history_empty_pairs=sum(x["history_empty"] for x,y in matched)))
    if pairs:p.write_csv(work/"exports/paired.csv",pairs)
    text=["# Stage attribution screen","","Inspected development questions; no original-method or population claim.",
        "","| Scope | Arm | Answered | Correct |","|---|---|---:|---:|"]
    text += [f"| {x['scope']} | {x['arm']} | {x['answered']} | {x['correct']} |" for x in summary]
    text += ["","A: H visible in encoding/readout. B: H masked during R encoding only. C: H masked at every question/decode call only. D: H masked at both stages. Compact: fresh recent-4 at compact positions.",
        "B/D use a physical H mask validated against independent cache construction at preserved logical positions; they are not complete DSCache implementations.",
        "H remains physically stored in A-D. C may retain historical information mixed indirectly into R. History-empty questions are null controls.",
        "Long 204/q3 results are separate from the main 40-question matrix. A and compact must replay earlier choices or execution stops.",
        "Shared cache preparation/snapshots are logged separately; per-answer timing/peak allocation includes shared control snapshots and is not standalone algorithm efficiency."]
    (work/"exports/report.md").write_text("\n".join(text)+"\n",encoding="utf-8")


def package(work,status):
    captured=work/"captured_logs";captured.mkdir(parents=True,exist_ok=True)
    for x in (work/"logs").glob("*"):
        if x.is_file():shutil.copy2(x,captured/x.name)
    if (work/"progress.json").exists():shutil.copy2(work/"progress.json",captured/"progress_snapshot.json")
    files=[]
    for area in ("exports","checkpoints","provenance","preparation","captured_logs"):
        files += [x for x in (work/area).rglob("*") if x.is_file() and ".ipynb_checkpoints" not in x.parts]
    for name in ("stage_runner.py","stage_context_utils.py","context_input_utils.py","diagnostic_pilot_utils.py","reference.json","launch.json","asset_manifest.json","preflight.json","stage_preflight.json"):
        if (work/name).exists():files.append(work/name)
    manifest=[dict(path=x.relative_to(work).as_posix(),bytes=x.stat().st_size,sha256=p.digest(x)) for x in sorted(set(files))]
    p.atomic_json(work/"evidence_manifest.json",dict(status=status,files=manifest))
    target=work/ARTIFACT
    with ZipFile(target.with_suffix(".tmp"),"w",compression=ZIP_DEFLATED) as z:
        for x in manifest:z.write(work/x["path"],x["path"])
        z.write(work/"evidence_manifest.json","evidence_manifest.json")
    os.replace(target.with_suffix(".tmp"),target)
    print(f"EVIDENCE_ZIP={target}\nSHA256={p.digest(target)}",flush=True)


def execute(args):
    import numpy as np
    import torch
    import transformers
    reference=json.loads((args.work/"reference.json").read_text(encoding="utf-8"))
    d.check_reference(reference,args.repo)
    if args.model.name!=p.MODEL_SHA or torch.__version__!="2.10.0+cu128" or transformers.__version__!="4.45.0.dev0" or not torch.cuda.is_available():raise ValueError("Pinned model/CUDA environment required")
    config=dict(reference_sha256=p.digest(args.work/"reference.json"),sources={n:p.digest(args.work/n) for n in
        ("stage_runner.py","stage_context_utils.py","context_input_utils.py","diagnostic_pilot_utils.py")},arms=ARMS,
        model_revision=p.MODEL_SHA,torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(0),
        seed=p.SEED,main_window=20,recent_window=4,long_case=args.long_case,mask_policy="all layers, prefill and every decode; physical H retained",shared_control_snapshots=True)
    config_hash=d.fingerprint(config);path=args.work/"exports/run_config.json"
    if path.exists() and json.loads(path.read_text())["config_hash"]!=config_hash:raise d.RunConflict("Stage configuration changed; use new WORK")
    p.atomic_json(path,dict(**config,config_hash=config_hash))
    failure=args.work/"exports/failure.json"
    if failure.exists():
        dest=args.work/"logs"/f"previous_failure_{time.time_ns()}.json";dest.parent.mkdir(parents=True,exist_ok=True);os.replace(failure,dest)
    videos=copy.deepcopy(reference["videos"]);d.acquire_videos(videos,args.reuse,args.work)
    p.atomic_json(args.work/"exports/replay_annotations.json",videos)
    sys.path.insert(0,str(args.repo))
    from inference.llavaov_hermes import load_model
    from video_qa.hermes_vqa import HermesVQA
    torch.set_num_threads(min(8,os.cpu_count() or 1));torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(p.SEED);np.random.seed(p.SEED);torch.manual_seed(p.SEED)
    p.report_progress("loading_stage_model")
    model,processor=load_model(model_path=str(args.model),kv_size=4000,streaming=True,sample_fps=p.FPS);model.eval()
    analyzer=HermesVQA(anno=[],save_dir=str(args.work),sample_fps=p.FPS,qa_model=model,qa_processor=processor)
    old={(r["sample_id"],r["qa_index"],r["method"]):r for r in reference["input_rows"]}
    rows=[];done=0;started=time.perf_counter();preflight_done=False;long_done=not args.long_case
    with torch.inference_mode():
        # Start with the already inspected 204 case, then preserve a deterministic video order.
        videos.sort(key=lambda v:(v["sample_id"]!=204,v["sample_id"]))
        for video in videos:
            directory,meta,_=d.acquire_features(model,video,args.reuse,args.work,d.fingerprint(dict(config=config_hash,video=video["video_sha256"])))
            bank=c.FeatureBank(directory,meta);feature_hash=d.fingerprint(meta)
            if not preflight_done:
                p.preflight(model,video,args.work)
                ids=processor.tokenizer("What is visible?",return_tensors="pt").input_ids.to(model.device)
                p.report_progress("stage_mask_position_preflight")
                p.atomic_json(args.work/"stage_preflight.json",numerical_preflight(model,bank,meta,ids,
                    diagnostic_path=args.work/"exports/stage_preflight_diagnostics.json"))
                preflight_done=True
            for qi in range(len(video["conversations"])):
                if done and args.max_hours and time.perf_counter()-started>args.max_hours*3600:break
                h,r=split_frames(video,qi)
                torch.cuda.reset_peak_memory_stats()
                before=time.perf_counter();states,detail=prepare_states(model,bank,meta,h,r);p.sync()
                detail.update(shared_preparation_s=time.perf_counter()-before,history_indices=h,recent_indices=r,
                              preparation_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
                p.atomic_json(args.work/"preparation/main"/f"sample_{video['sample_id']}_q{qi}.json",detail)
                # Replay gates run first; no scientific factor comparison after a mismatch.
                for arm in (ARMS[0],ARMS[4],ARMS[1],ARMS[2],ARMS[3]):
                    previous=old[video["sample_id"],qi,"recent_20" if arm==ARMS[0] else "recent_4"] if arm in (ARMS[0],ARMS[4]) else None
                    rows.append(run_answer(model,analyzer,video,qi,arm,"main",states,detail,h,r,feature_hash,config_hash,args,
                                           previous["pred_choice"] if previous else None))
                del states
                if args.long_case and (video["sample_id"],qi)==(204,3):
                    h,r=split_frames(video,qi,True)
                    torch.cuda.reset_peak_memory_stats()
                    before=time.perf_counter();states,detail=prepare_states(model,bank,meta,h,r);p.sync()
                    detail.update(shared_preparation_s=time.perf_counter()-before,history_indices=h,recent_indices=r,
                                  preparation_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
                    p.atomic_json(args.work/"preparation/long_204_q3.json",detail)
                    for arm in ARMS[:4]:
                        rows.append(run_answer(model,analyzer,video,qi,arm,"long_204_q3",states,detail,h,r,feature_hash,config_hash,args,
                                               reference["long_reference_choice"] if arm==ARMS[0] else None))
                    del states;long_done=True
                done+=1;export(rows,args.work)
                elapsed=time.perf_counter()-started
                print(f"QUESTION_BLOCK_DONE {done}/40 elapsed_min={elapsed/60:.2f} rough_remaining_min={(40-done)*elapsed/done/60:.2f}",flush=True)
            del bank
            if done<40 and args.max_hours and time.perf_counter()-started>args.max_hours*3600:break
    status="complete" if done==40 and long_done else "partial_time_budget"
    if status=="complete" and (sum(x["scope"]=="main" for x in rows)!=200 or any(x["status"]!="answered" for x in rows if x["scope"]=="main")):raise ValueError("Main stage matrix incomplete")
    p.atomic_json(args.work/"exports/completion.json",dict(status=status,main_questions=done,main_planned=40,main_answers=sum(x["scope"]=="main" and x["status"]=="answered" for x in rows),
        long_case_done=long_done,long_case_enabled=bool(args.long_case),rows=len(rows),config_hash=config_hash,elapsed_post_model_s=time.perf_counter()-started,
        timing_scope="Includes preflight, verified feature work, shared control cache construction, answers and exports; excludes setup/download/model load/final packaging"))
    p.report_progress(status,main_questions=done);package(args.work,status)
    print("STAGE_ATTRIBUTION_COMPLETE" if status=="complete" else "STAGE_ATTRIBUTION_PARTIAL",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ("repo","work","model","reuse"):parser.add_argument("--"+n,type=Path,required=True)
    parser.add_argument("--long-case",type=int,choices=(0,1),default=1)
    parser.add_argument("--max-hours",type=float,default=1.0)
    args=parser.parse_args()
    for n in ("repo","work","model","reuse"):setattr(args,n,getattr(args,n).resolve())
    d.validate_workspaces(args.work,args.reuse)
    if args.max_hours<0:parser.error("Invalid soft budget")
    with p.ProgressMonitor(args.work):
        try:execute(args)
        except d.RunConflict:raise
        except Exception as exc:
            p.atomic_json(args.work/"exports/failure.json",dict(status="failed",type=type(exc).__name__,detail=str(exc)[-1500:]))
            p.atomic_json(args.work/"exports/completion.json",dict(status="failed"));package(args.work,"failed_partial");raise


if __name__=="__main__":main()
