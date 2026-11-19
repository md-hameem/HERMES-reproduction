

import types
import torch
from logzero import logger
from video_qa.base import MODELS, work
from video_qa.hermes_vqa import HermesVQA
from inference.reindex_1d import (compute_cos_sin_for_positions, rotary_delta,
    apply_rotary_delta_to_keys_only)


def fifo_keep_indices(current_lengths, visual_start_idx, kv_size):
    # Return prefix + newest visual-token indices, independently per layer.
    assert visual_start_idx>=0 and kv_size>0
    all_indices=[]
    for length in current_lengths:
        assert length>=visual_start_idx
        visual=length-visual_start_idx
        keep_visual=min(kv_size,visual)
        prefix=torch.arange(visual_start_idx,dtype=torch.long)
        recent=torch.arange(length-keep_visual,length,dtype=torch.long)
        all_indices.append(torch.cat([prefix,recent]))
    return all_indices


def predict_and_compress_fifo(self):
    # Compression-only entry point: no pseudo queries or HERMES scoring.
    if self.kv_cache is None:return
    lengths=[k.shape[2] for k,v in self.kv_cache]
    current_max=max(lengths)
    if current_max>self.kv_size:
        keep_per_layer=fifo_keep_indices(lengths,self.visual_start_idx,self.kv_size)
        device=self.device
        max_pos_limit=getattr(self.language_model.config,'max_position_embeddings',32768)
        compact_threshold=max_pos_limit-1024
        current_max_pos=max((p[-1].item() for p in self._position_ids_cache if p is not None and p.numel()),default=0)
        should_compact=current_max_pos>compact_threshold if self.streaming else True
        old_positions=[p.clone() if p is not None else torch.arange(n,device=device) for p,n in zip(self._position_ids_cache,lengths)]
        new_cache=[]; after=[]
        for layer_idx,((k_layer,v_layer),keep) in enumerate(zip(self.kv_cache,keep_per_layer)):
            keep=keep.to(device=device,dtype=torch.long)
            assert keep.numel()<=lengths[layer_idx]
            k_kept=torch.index_select(k_layer,2,keep)
            v_kept=torch.index_select(v_layer,2,keep)
            old_pos=old_positions[layer_idx].index_select(0,keep)
            if should_compact:
                new_pos=torch.arange(keep.numel(),device=device,dtype=torch.long)
                cos_old,sin_old=compute_cos_sin_for_positions(self.language_model,len(old_pos),old_pos,k_layer.dtype,device)
                cos_new,sin_new=compute_cos_sin_for_positions(self.language_model,len(new_pos),new_pos,k_layer.dtype,device)
                cos_delta,sin_delta=rotary_delta(cos_old,sin_old,cos_new,sin_new)
                k_kept=apply_rotary_delta_to_keys_only(k_kept,cos_delta,sin_delta)
            else:new_pos=old_pos
            new_cache.append((k_kept.contiguous(),v_kept.contiguous()))
            self._position_ids_cache[layer_idx]=new_pos.clone().contiguous()
            after.append(k_kept.shape[2])
        self.kv_cache=new_cache
        self.token_activity_cache=[None for _ in range(self.num_layers)]
        expected=self.visual_start_idx+min(self.kv_size,max(lengths)-self.visual_start_idx)
        assert max(after)==min(after)==expected,(after,expected)
        print(f'FIFO_COMPRESSION before_max={current_max} after_min={min(after)} after_max={max(after)} budget={self.kv_size} compacted={should_compact}',flush=True)
    print(f'GPU memory usage: {self.get_gpu_memory_usage_gb()} GB',flush=True)


def install_fifo_control():
    original=MODELS['llava_ov_0.5b']['load_func']
    def load_fifo(*args,**kwargs):
        model,processor=original(*args,**kwargs)
        model.predict_and_compress=types.MethodType(predict_and_compress_fifo,model)
        model.fifo_control=True
        logger.info('FIFO control installed: prefix + newest visual tokens; no pseudo-query attention or summary tokens')
        return model,processor
    MODELS['llava_ov_0.5b']['load_func']=load_fifo


if __name__=='__main__':
    install_fifo_control()
    work(HermesVQA)
    