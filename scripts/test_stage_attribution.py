"""Mask/position numerical checks, factor invariants, safety and notebook validation."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import autodl_stage_attribution as s
import make_stage_attribution_notebook as notebook


class StageSafetyTests(unittest.TestCase):
    def test_main_split_is_causal_recent_twenty_and_long_scope_separate(self):
        ref=notebook.reference_data()
        for video in ref["videos"]:
            for qi in range(5):
                h,r=s.split_frames(video,qi)
                self.assertLessEqual(len(h),16);self.assertLessEqual(len(r),4)
                self.assertEqual(h+r,list(range(max(0,s.d.prefix_end(video,qi)-20),s.d.prefix_end(video,qi))))
        video=next(v for v in ref["videos"] if v["sample_id"]==204)
        h,r=s.split_frames(video,3,True)
        self.assertEqual((len(h),len(r)),(67,4))

    def test_checkpoint_preserves_factor_and_frame_identity(self):
        video=notebook.reference_data()["videos"][0]
        identity=s.d.question_identity(video,0,s.ARMS[0])
        row=dict(**identity,history_indices=[0],recent_indices=[1],status="answered",pred_choice=identity["gold_choice"],correct=1)
        value=dict(identity="key",row=row,row_sha256=s.d.fingerprint(row))
        self.assertTrue(s.checkpoint_valid(value,"key",identity,s.ARMS[0],[0],[1]))
        self.assertFalse(s.checkpoint_valid(value,"key",identity,s.ARMS[0],[0],[2]))
        row["correct"]=0;value["row_sha256"]=s.d.fingerprint(row)
        self.assertFalse(s.checkpoint_valid(value,"key",identity,s.ARMS[0],[0],[1]))

    def test_notebook_cells_validate_with_exact_source_and_no_gpu_or_network(self):
        nb=json.loads(notebook.DEST.read_text(encoding="utf-8"));ns={}
        with patch.dict(os.environ,HERMES_STAGE_VALIDATE_ONLY="1"),contextlib.redirect_stdout(io.StringIO()) as output:
            for i,cell in enumerate(nb["cells"]):
                if cell["cell_type"]=="code":exec(compile("".join(cell["source"]),f"stage_cell_{i}","exec"),ns)
        self.assertIn("VALIDATION_PASS",output.getvalue());self.assertNotIn("torch",ns)
        for var,file in [("RUNNER_SOURCE","autodl_stage_attribution.py"),("CONTEXT_SOURCE","autodl_context_diagnostic.py"),
                         ("INPUT_SOURCE","autodl_input_diagnostic.py"),("UTILS_SOURCE","autodl_failure_pilot.py")]:
            self.assertEqual(ns[var],(notebook.ROOT/"scripts"/file).read_text(encoding="utf-8"))
        ref=json.loads(ns["REFERENCE_JSON"])
        self.assertEqual(len(ref["videos"]),8);self.assertEqual(len(ref["input_rows"]),200)
        self.assertEqual(ref["long_reference_choice"],"A")
        s.d.check_reference(ref,notebook.ROOT)

    def test_package_excludes_cache_tensors_and_hashes_mask_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            s.p.atomic_json(root/"preparation/example.json",dict(mask_trace=[dict(blocked_start=13,blocked_stop=209)]))
            (root/"features").mkdir();(root/"features/state.pt").write_bytes(b"not for archive")
            with contextlib.redirect_stdout(io.StringIO()):s.package(root,"failed_partial")
            with ZipFile(root/s.ARTIFACT) as z:
                from validate_causal_pilot_evidence import check_manifest
                manifest,data=check_manifest(z)
                self.assertEqual(manifest["status"],"failed_partial")
                self.assertIn("preparation/example.json",data);self.assertNotIn("features/state.pt",data)

    def test_config_conflict_preserves_existing_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            s.p.atomic_json(root/"new/exports/completion.json",dict(status="complete",preserve=True))
            argv=["stage","--repo",str(root/"repo"),"--work",str(root/"new"),"--model",str(root/"model"),"--reuse",str(root/"old")]
            with patch("sys.argv",argv),patch.object(s,"execute",side_effect=s.d.RunConflict("changed")),\
                 patch.object(s,"package") as package,contextlib.redirect_stdout(io.StringIO()),self.assertRaises(s.d.RunConflict):s.main()
            self.assertTrue(json.loads((root/"new/exports/completion.json").read_text())["preserve"])
            package.assert_not_called()

    def test_exports_never_mix_long_case_with_main_denominators(self):
        rows=[]
        for scope in ("main","long_204_q3"):
            for arm in (s.ARMS if scope=="main" else s.ARMS[:4]):
                rows.append(dict(scope=scope,sample_id=204,qa_index=3,method=arm,status="answered",correct=1,pred_choice="B",history_empty=False))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);s.export(rows,root)
            summary=s.p.read_csv(root/"exports/summary.csv")
            self.assertEqual(len(summary),9)
            self.assertTrue(all(r["answered"]=="1" for r in summary))
            pairs=s.p.read_csv(root/"exports/paired.csv")
            self.assertTrue(all(r["questions"]=="1" for r in pairs))


class StageNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch,transformers
        if transformers.__version__!="4.45.0.dev0":raise unittest.SkipTest("Pinned Transformers required")
        cls.torch=torch;torch.set_num_threads(1)

    def fixture(self):
        torch=self.torch
        from test_autodl_failure_pilot import NativeScorerTests
        model=NativeScorerTests.model(self)
        model.visual_start_idx=5;model.conv_history=[];model.token_activity_cache=[None]*3
        model.clear_cache=lambda:None
        initial=s.snapshot(model)
        class Bank:
            def __init__(self):self.x=torch.randn(1,392,64)
            def get(self,indices):return torch.cat([self.x[:,i*196:(i+1)*196] for i in indices],dim=1)
        return model,Bank(),dict(chunks=[dict(file="fixture",start=0,stop=2)]),initial

    def test_mask_blocks_history_and_future_but_not_prompt_or_current_tokens(self):
        torch=self.torch
        mask=s.causal_mask(1,3,8,torch.float32,torch.device("cpu"),2,6)[0,0]
        minimum=torch.finfo(torch.float32).min
        self.assertTrue(torch.all(mask[:,2:6]==minimum))
        self.assertTrue(torch.all(mask[:,:2]==0))
        self.assertEqual(float(mask[0,8]),0)
        self.assertEqual(float(mask[0,9]),minimum)
        with self.assertRaises(ValueError):s.causal_mask(1,1,4,torch.float32,torch.device("cpu"),2,5)

    def test_complete_numerical_preflight_against_independent_and_compact_references(self):
        torch=self.torch;model,bank,meta,initial=self.fixture()
        with torch.inference_mode(),patch.object(s.p,"reset",side_effect=lambda m:s.restore(m,initial)):
            result=s.numerical_preflight(model,bank,meta,torch.tensor([[2,4,6]]),atol=1e-5,rtol=1e-5)
        self.assertEqual(result["status"],"pass")
        self.assertLess(result["independent_construction_max_kv_error"],1e-5)
        self.assertTrue(result["all_layer_hidden_history_attention_zero"])
        self.assertEqual([r["query_and_decode_mask_calls"] for r in result["comparisons"][1:]],[3,3])

    def run_half_preflight(self, path):
        torch=self.torch
        model,bank,meta,_=self.fixture()
        model.dtype=torch.float16
        lm=model.language_model.half()
        for layer in lm.model.layers:
            layer.self_attn.rotary_emb.inv_freq=layer.self_attn.rotary_emb.inv_freq.float()
        weights={name:value.clone() for name,value in lm.named_parameters()}
        buffers={name:value.clone() for name,value in lm.named_buffers()}
        def reset(target):
            target._layer_position_ids.clear()
            target.kv_cache=list(lm(input_ids=torch.tensor([[1,3,5,7,9]]),use_cache=True).past_key_values)
            target._position_ids_cache=[torch.arange(5) for _ in range(model.num_layers)]
        try:
            with torch.inference_mode(),patch.object(s.p,"reset",side_effect=reset):
                return s.numerical_preflight(model,bank,meta,torch.tensor([[2,4,6]]),diagnostic_path=path)
        finally:
            for name,value in lm.named_parameters():
                torch.testing.assert_close(value,weights[name],atol=0,rtol=0)
            for name,value in lm.named_buffers():
                torch.testing.assert_close(value,buffers[name],atol=0,rtol=0)

    def test_half_preflight_runs_fp32_reference_then_restored_half(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"diagnostics.json"
            result=self.run_half_preflight(path)
            diagnostics=json.loads(path.read_text())
        self.assertEqual(result["status"],"pass")
        self.assertEqual(diagnostics["fp32"]["dtype"],"torch.float32")
        self.assertEqual(diagnostics["production"]["dtype"],"torch.float16")
        self.assertEqual(len(diagnostics["production"]["construction_by_layer"]),6)

    def test_history_leak_stops_at_fp32_and_saves_diagnostics(self):
        causal_mask=s.causal_mask
        def unblocked(batch,q,past,dtype,device,blocked_start,blocked_stop):
            return causal_mask(batch,q,past,dtype,device,0,0)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"diagnostics.json"
            with patch.object(s,"causal_mask",side_effect=unblocked),self.assertRaises(AssertionError):
                self.run_half_preflight(path)
            diagnostics=json.loads(path.read_text())
        self.assertTrue(diagnostics["fp32"]["construction_by_layer"])
        self.assertEqual(diagnostics["production"],{})

    def test_history_cache_is_identical_but_recent_deeper_cache_can_change(self):
        torch=self.torch;model,bank,meta,initial=self.fixture()
        with torch.inference_mode(),patch.object(s.p,"reset",side_effect=lambda m:s.restore(m,initial)):
            states,detail=s.prepare_states(model,bank,meta,[0],[1])
        stop=201
        for (ak,av),(bk,bv) in zip(states["A"]["kv"],states["B"]["kv"]):
            self.assertTrue(torch.equal(ak[:,:,:stop],bk[:,:,:stop]));self.assertTrue(torch.equal(av[:,:,:stop],bv[:,:,:stop]))
        self.assertGreater(float((states["A"]["kv"][-1][0][:,:,stop:]-states["B"]["kv"][-1][0][:,:,stop:]).abs().max()),1e-6)
        self.assertEqual(states["A"]["positions"][0].tolist(),states["B"]["positions"][0].tolist())
        self.assertTrue(detail["independent_encoding_mask_trace"])

    def test_absent_history_is_a_null_factor(self):
        torch=self.torch;model,bank,meta,initial=self.fixture()
        with torch.inference_mode(),patch.object(s.p,"reset",side_effect=lambda m:s.restore(m,initial)):
            states,detail=s.prepare_states(model,bank,meta,[],[0,1])
        for (ak,av),(bk,bv) in zip(states["A"]["kv"],states["B"]["kv"]):self.assertTrue(torch.equal(ak,bk) and torch.equal(av,bv))
        self.assertEqual(detail["independent_encoding_mask_trace"],[])

    def test_real_answer_function_masks_every_decode_call_and_restores_source(self):
        torch=self.torch;model,bank,meta,initial=self.fixture()
        from inference.llavaov_hermes import LlavaOneVision_Hermes as H
        class Tokenizer:
            eos_token_id=-1
            def __call__(self,text):return types.SimpleNamespace(input_ids=[2,4])
            def decode(self,ids,**kwargs):return "A"
        model.processor=types.SimpleNamespace(tokenizer=Tokenizer())
        model._truncate_kv_cache=types.MethodType(H._truncate_kv_cache,model)
        with torch.inference_mode(),patch.object(s.p,"reset",side_effect=lambda m:s.restore(m,initial)):
            states,_=s.prepare_states(model,bank,meta,[0],[1])
            s.restore(model,states["A"])
            with s.HistoryMask(model,5,201) as mask,contextlib.redirect_stdout(io.StringIO()):
                H.question_answering(model,dict(prompt="q",question="q"),max_new_tokens=4)
            s.assert_state(model,states["A"])
        self.assertEqual([e["query_tokens"] for e in mask.events],[2,1,1,1,1])
        self.assertTrue(all(e["blocked_start"]==5 and e["blocked_stop"]==201 for e in mask.events))

    def test_hook_removed_on_failure_and_noop_mask_is_opt_in(self):
        model,_,_,_=self.fixture()
        before=len(model.language_model._forward_pre_hooks)
        with self.assertRaises(RuntimeError):
            with s.HistoryMask(model,5,201):
                self.assertEqual(len(model.language_model._forward_pre_hooks),before+1)
                raise RuntimeError("fixture")
        self.assertEqual(len(model.language_model._forward_pre_hooks),before)
        with s.HistoryMask(model,5,5):self.assertEqual(len(model.language_model._forward_pre_hooks),before)

    def answer_fixture(self):
        model,bank,meta,initial=self.fixture()
        from inference.llavaov_hermes import LlavaOneVision_Hermes as H
        class Tokenizer:
            eos_token_id=-1
            def __call__(self,text):return types.SimpleNamespace(input_ids=[2,4])
            def decode(self,ids,**kwargs):return "A"
        model.processor=types.SimpleNamespace(tokenizer=Tokenizer())
        model._truncate_kv_cache=types.MethodType(H._truncate_kv_cache,model)
        model.question_answering=types.MethodType(H.question_answering,model)
        video=dict(sample_id=204,video_id=335,conversations=[dict(question="q",choices=["yes","no"],answer="yes",end_time=4,task="fixture")])
        analyzer=types.SimpleNamespace(format_mcqa_prompt=lambda q,ch:dict(prompt=q,question=q),extract_characters_regex=lambda x:x)
        return model,bank,meta,initial,video,analyzer

    def test_answer_wrapper_records_prefill_and_every_decode_mask_and_resumes(self):
        torch=self.torch;model,bank,meta,initial,video,analyzer=self.answer_fixture()
        with tempfile.TemporaryDirectory() as directory,torch.inference_mode(),patch.object(s.p,"reset",side_effect=lambda m:s.restore(m,initial)),\
             patch.object(s.p,"sync"),patch.object(torch.cuda,"reset_peak_memory_stats"),patch.object(torch.cuda,"max_memory_allocated",return_value=1024**3),\
             contextlib.redirect_stdout(io.StringIO()):
            states,detail=s.prepare_states(model,bank,meta,[0],[1]);args=types.SimpleNamespace(work=Path(directory))
            row=s.run_answer(model,analyzer,video,0,s.ARMS[2],"main",states,detail,[0],[1],"features","config",args)
            self.assertEqual(row["correct"],1)
            self.assertEqual(row["cache_lengths"],[397]*3)
            self.assertEqual([x["query_tokens"] for x in row["mask_trace"]],[2]+[1]*16)
            self.assertEqual(row["recent_logical_start"],201)
            resumed=s.run_answer(model,analyzer,video,0,s.ARMS[2],"main",states,detail,[0],[1],"features","config",args)
            self.assertEqual(row,resumed)

    def test_saved_noop_mismatch_stops_again_on_resume(self):
        torch=self.torch;model,bank,meta,initial,video,analyzer=self.answer_fixture()
        with tempfile.TemporaryDirectory() as directory,torch.inference_mode(),patch.object(s.p,"reset",side_effect=lambda m:s.restore(m,initial)),\
             patch.object(s.p,"sync"),patch.object(torch.cuda,"reset_peak_memory_stats"),patch.object(torch.cuda,"max_memory_allocated",return_value=1024**3),\
             contextlib.redirect_stdout(io.StringIO()):
            states,detail=s.prepare_states(model,bank,meta,[0],[1]);args=types.SimpleNamespace(work=Path(directory))
            with self.assertRaisesRegex(ValueError,"replay mismatch"):
                s.run_answer(model,analyzer,video,0,s.ARMS[0],"main",states,detail,[0],[1],"features","config",args,"B")
            with self.assertRaisesRegex(ValueError,"Previously saved"):
                s.run_answer(model,analyzer,video,0,s.ARMS[0],"main",states,detail,[0],[1],"features","config",args,"B")


if __name__=="__main__":unittest.main()
