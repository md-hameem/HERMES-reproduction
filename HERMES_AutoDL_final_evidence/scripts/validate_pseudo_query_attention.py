

import os, sys, json, math, argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from inference.llavaov_hermes import load_model, _get_rotary_module
from inference.reindex_1d import apply_rotary_pos_emb_1d
from transformers.cache_utils import DynamicCache

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--output-dir', required=True)
args = parser.parse_args()
outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
assert torch.cuda.is_available()
model, processor = load_model(args.model, kv_size=4000, streaming=True, device='cuda', sample_fps=0.5)
model.encode_init_prompt()
assert model.kv_cache is not None
implementation = model.language_model.config._attn_implementation
assert implementation in ('sdpa','flash_attention_2'), implementation
query = 'Summarize the video narrative, identifying main characters, key events, timeline changes, and the overall theme.'
input_ids = torch.as_tensor([processor.tokenizer(query).input_ids], device=model.device, dtype=torch.long)
assert input_ids.shape[1] > 1, 'A multi-token query is required to test causal masking.'
past_key_values = model.kv_cache
past_len = past_key_values[0][0].shape[2]
assert past_len > 0
start_pos = model._get_next_start_pos_per_layer()
q_len, batch = input_ids.shape[1], input_ids.shape[0]
position_ids = model._build_position_ids(start_pos[0], q_len, batch)
model._layer_position_ids.clear()
for layer_idx in range(model.num_layers):
    model._layer_position_ids[layer_idx] = model._build_position_ids(start_pos[layer_idx], q_len, batch)
model._set_rotary_required_len(max(start_pos)+q_len)

with torch.inference_mode():
    native_out = model.language_model(input_ids=input_ids, use_cache=False,
        past_key_values=DynamicCache.from_legacy_cache(past_key_values), output_attentions=True,
        position_ids=position_ids, cache_position=None, attention_mask=None)
    native = list(native_out.attentions)
    current = model._compute_attention_scores_manually(input_ids, past_key_values)

    # Reproduce the manual projections/RoPE, adding only the missing causal mask.
    inputs_embeds = model.get_input_embeddings()(input_ids)
    hidden_states = inputs_embeds
    config = model.language_model.config
    num_heads, num_kv_heads = config.num_attention_heads, config.num_key_value_heads
    head_dim = config.hidden_size//num_heads
    causal = []
    for layer_idx, layer in enumerate(model.language_model.model.layers):
        past_k, past_v = past_key_values[layer_idx]
        pos = model._build_position_ids(start_pos[layer_idx], q_len, batch)
        norm = layer.input_layernorm(hidden_states)
        attn = layer.self_attn
        q = attn.q_proj(norm).view(batch,q_len,num_heads,head_dim).transpose(1,2)
        k = attn.k_proj(norm).view(batch,q_len,num_kv_heads,head_dim).transpose(1,2)
        v = attn.v_proj(norm).view(batch,q_len,num_kv_heads,head_dim).transpose(1,2)
        cos,sin = _get_rotary_module(model.language_model)(v, pos.max().item()+1)
        cos,sin = cos[pos].unsqueeze(1),sin[pos].unsqueeze(1)
        q,k = apply_rotary_pos_emb_1d(q,cos,sin),apply_rotary_pos_emb_1d(k,cos,sin)
        k,v = torch.cat([past_k,k],dim=2),torch.cat([past_v,v],dim=2)
        if num_kv_heads != num_heads:
            rep = num_heads//num_kv_heads
            k,v = torch.repeat_interleave(k,rep,dim=1),torch.repeat_interleave(v,rep,dim=1)
        logits = torch.matmul(q.float(),k.float().transpose(-2,-1))/math.sqrt(head_dim)
        kv_len = logits.shape[-1]
        q_pos = torch.arange(q_len,device=logits.device).view(-1,1)
        k_pos = torch.arange(kv_len,device=logits.device).view(1,-1)
        allowed = k_pos <= (kv_len-q_len)+q_pos
        logits = logits.masked_fill(~allowed.view(1,1,q_len,kv_len), torch.finfo(logits.dtype).min)
        causal.append(F.softmax(logits,dim=-1,dtype=torch.float32).to(q.dtype))
model._layer_position_ids.clear()
assert len(native)==len(current)==len(causal)==model.num_layers

def metrics(a,b):
    af,bf = a.float(),b.float()
    assert af.shape==bf.shape and torch.isfinite(af).all() and torch.isfinite(bf).all()
    diff=(af-bf).abs()
    cosine=F.cosine_similarity(af.flatten(),bf.flatten(),dim=0).item()
    # Ranking behavior over past tokens after averaging heads and pseudo-query positions.
    past=a.shape[-1]-a.shape[-2]
    av=a.float()[0,:,:,:past].mean(dim=(0,1))
    bv=b.float()[0,:,:,:past].mean(dim=(0,1))
    topk=min(100,past)
    ai=set(torch.topk(av,topk).indices.cpu().tolist())
    bi=set(torch.topk(bv,topk).indices.cpu().tolist())
    return {'max_abs':diff.max().item(),'mean_abs':diff.mean().item(),
            'cosine':cosine,f'past_top{topk}_overlap':len(ai&bi)/topk}

rows=[]
for i,(n,m,c) in enumerate(zip(native,current,causal)):
    assert n.shape==m.shape==c.shape
    for comparison,a,b in [('current_vs_native',m,n),('causal_frozen_vs_native',c,n),('current_vs_causal_frozen',m,c)]:
        rows.append({'layer':i,'comparison':comparison,**metrics(a,b)})
import pandas as pd
frame=pd.DataFrame(rows)
frame.to_csv(outdir/'attention_per_layer_metrics.csv',index=False)
summary=frame.groupby('comparison').agg(max_abs_max=('max_abs','max'),mean_abs_mean=('mean_abs','mean'),
    cosine_min=('cosine','min'),cosine_mean=('cosine','mean')).reset_index()
overlap_cols=[c for c in frame if c.startswith('past_top')]
overlap=frame.groupby('comparison')[overlap_cols].mean().reset_index()
summary=summary.merge(overlap,on='comparison')
summary.to_csv(outdir/'attention_summary.csv',index=False)
provenance={'model_snapshot':Path(args.model).resolve().name,'attention_implementation':implementation,
    'query':query,'query_tokens':q_len,'past_tokens':past_len,'layers':model.num_layers,
    'reference':'native language_model output_attentions=True',
    'current_manual':'frozen input embeddings across layers; no causal mask',
    'causal_frozen':'frozen input embeddings across layers; causal mask added',
    'interpretation_limit':'Single synthetic text-cache/query differential test; not a full-video behavioral ablation.'}
(outdir/'attention_test_provenance.json').write_text(json.dumps(provenance,indent=2))
print(summary.to_string(index=False))
print('ATTENTION_DIFFERENTIAL_TEST_COMPLETE')
    