"""Context map invariants, safe counterfactuals, resume and CPU numerical checks."""
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

import autodl_context_diagnostic as c
import make_context_diagnostic_notebook as notebook


class ContextSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = notebook.reference_data()
        cls.videos = {v["sample_id"]:v for v in cls.reference["videos"]}

    def test_all_plans_use_prequery_frames_and_preserve_fixed_length_tail(self):
        for sid,qi in c.CASES:
            video = self.videos[sid]
            for family in ("recent_20","full_prefix"):
                baseline = c.arm_plan(video,qi,family+"_fixed4")
                for filler in ("opening","local"):
                    plan = c.arm_plan(video,qi,family+"_"+filler+"_fill")
                    self.assertEqual(len(plan["sampled_indices"]),len(baseline["sampled_indices"]))
                    self.assertEqual(plan["sampled_indices"][-4:],baseline["sampled_indices"][-4:])
                    c.validate_indices(plan["sampled_indices"],video,qi)
                    self.assertEqual(plan["protected_tail_slots"],list(range(len(plan["sampled_indices"])-4,len(plan["sampled_indices"]))))

    def test_duplicate_fillers_are_explicit_but_future_and_reordered_inputs_fail(self):
        video = self.videos[204]
        c.validate_indices([0,0,0,0,4,5,6,7],video,0)
        for indices in ([8],[True],[2,1],[-1]):
            with self.assertRaises(ValueError):
                c.validate_indices(indices,video,0)
        with self.assertRaisesRegex(ValueError,"Unknown"):
            c.arm_plan(video,0,"unregistered_arm")

    def test_short_prefix_replacement_is_labeled_ineffective(self):
        video = self.videos[208]
        for arm in ("recent_20_opening_fill","recent_20_local_fill","full_prefix_opening_fill"):
            plan = c.arm_plan(video,0,arm)
            self.assertEqual(plan["sampled_indices"],[0,1,2,3])
            self.assertFalse(plan["intervention_effective"])

    def test_candidate_support_is_limited_to_reviewed_sheet_candidates(self):
        self.assertEqual(c.arm_plan(self.videos[204],0,"candidate_support")["sampled_indices"],[6,7])
        self.assertEqual(c.arm_plan(self.videos[204],3,"candidate_support")["sampled_indices"],[69])
        self.assertEqual(c.arm_plan(self.videos[203],2,"candidate_support")["skip_reason"],"candidate_not_defined")

    def test_default_planned_denominators_include_explicit_skips(self):
        recorded,eligible = 0,0
        for sid,qi in c.CASES:
            for arm in c.ARMS:
                plan = c.arm_plan(self.videos[sid],qi,arm)
                recorded += 1
                if not plan["skip_reason"] and not (arm.startswith("full_prefix_") and len(plan["sampled_indices"])*196 > 16384):
                    eligible += 1
        self.assertEqual((recorded,eligible),(84,74))

    def test_group_partitions_do_not_change_content_or_frame_order(self):
        video = self.videos[203]
        original = c.arm_plan(video,2,"full_prefix_original")
        for partition in ("original","fixed4","framewise"):
            groups = c.input_groups(original["sampled_indices"],partition,video["feature_meta"])
            self.assertEqual([i for group in groups for i in group],original["sampled_indices"])
        self.assertEqual(c.input_groups([],"original",video["feature_meta"]),[])

    def test_resume_rejects_changed_plan_or_damaged_correctness(self):
        video = self.videos[204]
        plan = c.arm_plan(video,0,"recent_4_original")
        identity = c.d.question_identity(video,0,"recent_4_original")
        row = dict(**identity,plan=plan,status="answered",pred_choice="C",correct=1)
        value = dict(identity="configuration",row=row,row_sha256=c.d.fingerprint(row))
        self.assertTrue(c.valid_checkpoint(value,"configuration",identity,plan))
        self.assertFalse(c.valid_checkpoint(value,"different",identity,plan))
        row["correct"] = 0
        value["row_sha256"] = c.d.fingerprint(row)
        self.assertFalse(c.valid_checkpoint(value,"configuration",identity,plan))

    def test_notebook_all_cells_validate_and_embed_exact_sources(self):
        value = json.loads(notebook.DEST.read_text(encoding="utf-8"))
        ns = {}
        with patch.dict(os.environ,HERMES_CONTEXT_VALIDATE_ONLY="1"),contextlib.redirect_stdout(io.StringIO()) as out:
            for i,cell in enumerate(value["cells"]):
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]),f"context_cell_{i}","exec"),ns)
        self.assertIn("VALIDATION_PASS",out.getvalue())
        for variable,path in (("RUNNER_SOURCE","autodl_context_diagnostic.py"),("INPUT_SOURCE","autodl_input_diagnostic.py"),("UTILS_SOURCE","autodl_failure_pilot.py")):
            self.assertEqual(ns[variable],(notebook.ROOT/"scripts"/path).read_text(encoding="utf-8"))
        reference = json.loads(ns["REFERENCE_JSON"])
        self.assertEqual(len(reference["videos"]),5)
        self.assertEqual(len(reference["input_rows"]),35)
        c.d.check_reference(reference,notebook.ROOT)
        self.assertNotIn("torch",ns)

    def test_package_contains_hashed_source_maps_and_no_feature_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c.p.atomic_json(root/"exports/completion.json",dict(status="failed"))
            (root/"context_runner.py").write_text("# fixture source")
            (root/"features").mkdir()
            (root/"features/test.pt").write_bytes(b"excluded")
            with contextlib.redirect_stdout(io.StringIO()):
                c.package(root,"failed_partial")
            with ZipFile(root/c.ARTIFACT) as z:
                from validate_causal_pilot_evidence import check_manifest
                manifest,payload = check_manifest(z)
                self.assertEqual(manifest["status"],"failed_partial")
                self.assertIn("context_runner.py",payload)
                self.assertNotIn("features/test.pt",payload)

    def test_earlier_config_conflict_does_not_overwrite_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c.p.atomic_json(root/"new/exports/completion.json",dict(status="complete",sentinel="preserve"))
            argv=["runner","--repo",str(root/"repo"),"--work",str(root/"new"),"--reuse",str(root/"old"),"--model",str(root/"model")]
            with patch("sys.argv",argv),patch.object(c,"execute",side_effect=c.d.RunConflict("changed")),\
                 patch.object(c,"package") as package,contextlib.redirect_stdout(io.StringIO()),self.assertRaises(c.d.RunConflict):
                c.main()
            self.assertEqual(json.loads((root/"new/exports/completion.json").read_text())["sentinel"],"preserve")
            package.assert_not_called()


class ContextNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch,transformers
        if transformers.__version__ != "4.45.0.dev0":
            raise unittest.SkipTest("Pinned Transformers needed for native numerical check")
        cls.torch = torch
        torch.set_num_threads(1)

    def test_feature_bank_repeats_only_declared_source_blocks(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x = torch.arange(4*196*2,dtype=torch.float16).reshape(1,4*196,2)
            torch.save(x,root/"chunk.pt")
            meta = dict(chunks=[dict(file="chunk.pt",start=0,stop=4)])
            bank = c.FeatureBank(root,meta)
            output = bank.get([0,0,2,3])
            self.assertTrue(torch.equal(output[:,:196],x[:,:196]))
            self.assertTrue(torch.equal(output[:,196:392],x[:,:196]))
            self.assertTrue(torch.equal(output[:,392:],x[:,392:]))

    def test_same_input_different_chunk_sizes_match_native_cache(self):
        torch = self.torch
        from test_autodl_failure_pilot import NativeScorerTests
        with torch.inference_mode():
            whole = NativeScorerTests.model(self)
            parts = NativeScorerTests.model(self)
            features = torch.randn(1,392,64)
            c.p.ingest_features(whole,features,2)
            for start in (0,196):
                c.p.ingest_features(parts,features[:,start:start+196],1)
            for (ka,va),(kb,vb) in zip(whole.kv_cache,parts.kv_cache):
                torch.testing.assert_close(ka,kb,atol=1e-6,rtol=1e-5)
                torch.testing.assert_close(va,vb,atol=1e-6,rtol=1e-5)
            for pa,pb in zip(whole._position_ids_cache,parts._position_ids_cache):
                self.assertTrue(torch.equal(pa,pb))

    def test_counterfactual_execution_records_fixed_positions_and_resumes(self):
        torch = self.torch
        reference = notebook.reference_data()
        video = next(v for v in reference["videos"] if v["sample_id"] == 204)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = torch.arange(8*196*2,dtype=torch.float16).reshape(1,8*196,2)
            torch.save(features,root/"chunk.pt")
            meta = dict(chunks=[dict(file="chunk.pt",start=0,stop=8,tokens=1568,sha256=c.p.digest(root/"chunk.pt"))])
            bank = c.FeatureBank(root,meta)
            calls=[]
            def answer(prompt,max_new_tokens):
                self.assertEqual(max_new_tokens,16)
                self.assertNotIn("gold_choice",prompt)
                calls.append(prompt)
                return "C"
            model=types.SimpleNamespace(visual_start_idx=13,num_layers=24,conv_history=[],_layer_position_ids={},
                processor=types.SimpleNamespace(tokenizer=lambda text:types.SimpleNamespace(input_ids=[1,2,3])),
                language_model=types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=32768)),
                kv_cache=[(torch.zeros(1,2,13,4),torch.zeros(1,2,13,4))],
                question_answering=answer,clear_cache=lambda:None,_get_cache_seq_len_per_layer=lambda:[1581]*24)
            analyzer=types.SimpleNamespace(format_mcqa_prompt=lambda q,ch:dict(prompt=q+str(ch),question=q),extract_characters_regex=lambda a:a)
            args=types.SimpleNamespace(work=root,max_visual_tokens=16384)
            with patch.object(c.p,"reset"),patch.object(c.p,"sync"),patch.object(c.p,"ingest_features") as ingest,\
                 patch.object(c.d,"assert_fresh_cache"),patch.object(torch.cuda,"reset_peak_memory_stats"),\
                 patch.object(torch.cuda,"mem_get_info",return_value=(24*1024**3,24*1024**3)),\
                 patch.object(torch.cuda,"max_memory_allocated",return_value=1024**3),contextlib.redirect_stdout(io.StringIO()):
                row=c.run_arm(model,analyzer,video,0,"full_prefix_opening_fill",bank,meta,"config",args)
                self.assertEqual(row["correct"],1)
                self.assertEqual(row["protected_tail_token_starts"],[797,993,1189,1385])
                self.assertEqual(row["plan"]["sampled_indices"],[0,0,0,0,4,5,6,7])
                observed=torch.cat([call.args[1] for call in ingest.call_args_list],dim=1)
                self.assertTrue(torch.equal(observed,bank.get([0,0,0,0,4,5,6,7])))
                resumed=c.run_arm(model,analyzer,video,0,"full_prefix_opening_fill",bank,meta,"config",args)
                self.assertEqual(resumed,row)
                self.assertEqual(len(calls),1)

    def test_contrast_export_does_not_call_identical_short_fill_a_new_effect(self):
        reference=notebook.reference_data()
        video=next(v for v in reference["videos"] if v["sample_id"] == 208)
        rows=[]
        for arm in c.ARMS:
            plan=c.arm_plan(video,0,arm)
            identity=c.d.question_identity(video,0,arm)
            rows.append(dict(**identity,plan=plan,status="skipped" if plan["skip_reason"] else "answered",skip_reason=plan["skip_reason"],
                             correct=None if plan["skip_reason"] else 0,pred_choice="D",wall_s=0))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            c.export(rows,reference,root)
            contrasts=c.p.read_csv(root/"exports/contrasts.csv")
            fills=[r for r in contrasts if r["arm_b"].endswith("_fill")]
            self.assertTrue(fills)
            self.assertTrue(all(r["same_content"] == "True" and r["replacement_effective"] == "False" for r in fills))


if __name__ == "__main__":
    unittest.main()
