"""Safety, paired-export, resume and small native-decoder checks for input controls."""
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

import autodl_input_diagnostic as d
import make_input_diagnostic_notebook as notebook


def fixture_video():
    return dict(sample_id=204, video_id=335, video_path="sample_204_real.mp4",
                conversations=[dict(question="What color?", choices=["Red", "Yellow"], answer="Yellow", end_time=7, task="test")],
                timestamps=[dict(sampled_index=str(i), source_frame=str(i*60), start_s=str(i*2), end_s=str(i*2+.03)) for i in range(8)])


class DiagnosticSafetyTests(unittest.TestCase):
    def test_partial_asset_reuse_downloads_only_missing_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            old,work=root/"old",root/"new"
            (old/"videos").mkdir(parents=True)
            existing=old/"videos/sample_204_real.mp4"
            existing.write_bytes(b"existing video fixture")
            other=root/"downloaded.mp4"
            other.write_bytes(b"new video fixture")
            videos=[dict(sample_id=204,video_path="sample_204_real.mp4",video_sha256=d.p.digest(existing)),
                    dict(sample_id=202,video_path="sample_202_real.mp4",video_sha256=d.p.digest(other))]
            downloaded=dict(mode="fixture",videos=[dict(sample_id=202,path=str(other),bytes=other.stat().st_size,sha256=d.p.digest(other))])
            with patch.object(d.p,"prepare_assets",return_value=downloaded) as prepare:
                d.acquire_videos(videos,old,work)
            self.assertEqual([v["sample_id"] for v in prepare.call_args.args[0]],[202])
            self.assertEqual(existing.read_bytes(),b"existing video fixture")
            self.assertEqual(len(json.loads((work/"asset_manifest.json").read_text())["videos"]),2)

    def test_config_conflict_preserves_previous_completion_and_does_not_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            work=root/"new"
            d.p.atomic_json(work/"exports/completion.json",dict(status="complete",sentinel="preserve"))
            argv=["runner","--repo",str(root/"repo"),"--work",str(work),"--reuse",str(root/"old"),"--model",str(root/"model")]
            with patch("sys.argv",argv),patch.object(d,"execute",side_effect=d.RunConflict("changed")),\
                 patch.object(d,"package") as package,contextlib.redirect_stdout(io.StringIO()),self.assertRaises(d.RunConflict):
                d.main()
            self.assertEqual(json.loads((work/"exports/completion.json").read_text())["sentinel"],"preserve")
            package.assert_not_called()

    def test_output_workspace_cannot_overwrite_reusable_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            d.validate_workspaces(root/"new", root/"old")
            for work,reuse in ((root,root),(root/"old/new",root/"old"),(root/"old",root/"old/new")):
                with self.assertRaises(ValueError):
                    d.validate_workspaces(work,reuse)

    def test_setup_logger_streams_output_and_propagates_failure(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, HERMES_DIAGNOSTIC_VALIDATE_ONLY="1"):
            ns = {}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(notebook.CONFIG,ns)
                exec(notebook.helpers(),ns)
            ns["WORK"] = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                ns["logged"]([ns["sys"].executable,"-c","print('SETUP_FIXTURE')"],"test.log")
                with self.assertRaisesRegex(RuntimeError,"Command failed"):
                    ns["logged"]([ns["sys"].executable,"-c","raise SystemExit(3)"],"test.log")
            self.assertIn("SETUP_FIXTURE",out.getvalue())
            self.assertIn("SETUP_FIXTURE",(Path(directory)/"logs/test.log").read_text())

    def test_existing_environment_is_reused_only_after_exact_probe_passes(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,HERMES_DIAGNOSTIC_VALIDATE_ONLY="1"):
            ns = {}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(notebook.CONFIG,ns)
            ns.update(VALIDATE_ONLY=False,ENV={},REUSE=Path(directory)/"old",WORK=Path(directory)/"new")
            candidate = ns["REUSE"]/"venv/bin/python"
            candidate.parent.mkdir(parents=True)
            candidate.write_text("fixture")
            calls=[]
            ns.update(probe=lambda *a,**k:types.SimpleNamespace(returncode=0),
                      package_environment=lambda env:dict(env),logged=lambda *a,**k:calls.append(a))
            with contextlib.redirect_stdout(io.StringIO()):
                exec(notebook.install(),ns)
            self.assertEqual(ns["PYTHON"],candidate)
            self.assertEqual(len(calls),2)  # exact environment probe log and pip freeze; no install

    def test_selection_respects_prefix_and_resets_do_not_expand_recent_window(self):
        video = fixture_video()
        for method, expected in [("question_only", []), ("recent_4", [0,1,2,3]), ("recent_20", [0,1,2,3]), ("native_prefix", [0,1,2,3])]:
            frames, reason = d.selected_frames(method, video, 0, {}, True)
            self.assertEqual(frames, expected)
            self.assertIsNone(reason)
            d.validate_frames(frames, video, 0, video["timestamps"])
        self.assertEqual(d.selected_frames("support_oracle", video, 0, {}, True), ([], "not_requested"))

    def test_future_duplicate_unsorted_and_noninteger_frames_rejected(self):
        video = fixture_video()
        for bad in ([4], [-1], [1,1], [2,1], [True], [1.0]):
            with self.subTest(indices=bad), self.assertRaises(ValueError):
                d.validate_frames(bad, video, 0, video["timestamps"])
        timestamps = copy.deepcopy(video["timestamps"])
        timestamps[3]["start_s"] = "8.0"
        with self.assertRaisesRegex(ValueError, "after query"):
            d.validate_frames([3], video, 0, timestamps)

    def test_oracle_requires_review_and_cannot_use_future_or_duplicate_cases(self):
        video = fixture_video()
        item = dict(sample_id=204, qa_index=0, sampled_indices=[1,3], verified=True, notes="Human-checked fixture")
        self.assertEqual(d.validate_oracles([item], [video])[(204,0)], item)
        for change in (dict(verified=False), dict(notes=""), dict(sampled_indices=[4]), dict(sampled_indices=[]), dict(qa_index=-1)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                d.validate_oracles([{**item, **change}], [video])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            d.validate_oracles([item, item], [video])

    def test_context_reserves_prompt_question_and_generation_without_truncating(self):
        self.assertIsNone(d.context_reason(20, 13, 90, 32768, 8192, "recent_20"))
        self.assertEqual(d.context_reason(42, 13, 90, 32768, 8192, "native_prefix"), "configured_native_cap")
        self.assertEqual(d.context_reason(20, 13, 90, 4000, 8192, "recent_20"), "context_limit")

    def test_feature_slices_preserve_original_identity_across_boundaries(self):
        chunks = [dict(file="a.pt", start=0, stop=4), dict(file="b.pt", start=4, stop=8)]
        self.assertEqual(d.feature_slices([2,3,4,6,7], chunks), [dict(file="a.pt", start=2, stop=4),
                         dict(file="b.pt", start=0, stop=1), dict(file="b.pt", start=2, stop=4)])
        with self.assertRaises(ValueError):
            d.feature_slices([8], chunks)
        with self.assertRaises(ValueError):
            d.feature_slices([2], chunks+[dict(file="overlap.pt",start=1,stop=3)])

    def test_checkpoint_cannot_reuse_different_config_question_or_features(self):
        identity = d.question_identity(fixture_video(), 0, "recent_4")
        row = dict(**identity, status="answered", sampled_indices=[0,1,2,3], pred_choice="B", correct=1)
        value = dict(checkpoint_identity="config-and-features", row=row, row_sha256=d.fingerprint(row))
        self.assertTrue(d.checkpoint_valid(value, "config-and-features", identity, [0,1,2,3]))
        self.assertFalse(d.checkpoint_valid(value, "other-features", identity, [0,1,2,3]))
        self.assertFalse(d.checkpoint_valid(value, "config-and-features", {**identity, "question":"Changed"}, [0,1,2,3]))
        value["row"]["correct"] = 0
        value["row_sha256"] = d.fingerprint(value["row"])
        self.assertFalse(d.checkpoint_valid(value, "config-and-features", identity, [0,1,2,3]))

    def test_rejected_historical_features_are_not_silently_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature = root/"chunk.pt"
            feature.write_bytes(b"fixture")
            meta = dict(chunks=[dict(file="chunk.pt", start=0, stop=1, tokens=196, sha256=d.p.digest(feature))])
            d.verify_feature_files(root, meta)
            feature.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                d.verify_feature_files(root, meta)

    def test_native_subset_comparison_excludes_unanswered_questions(self):
        video = fixture_video()
        identity = d.question_identity(video, 0, "native_prefix")
        old = [dict(sample_id="204", qa_index="0", method=m, correct="0") for m in ("fifo","hermes","causal")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = dict(**identity, status="answered", pred_choice="B", correct=1, wall_s=1.0)
            d.export([row], dict(videos=[video], predictions=old), root)
            paired = d.p.read_csv(root/"exports/paired_historical.csv")
            self.assertEqual(paired[9]["matched_questions"], "1")
            self.assertEqual(paired[9]["diagnostic_only_correct"], "1")
            row.update(status="skipped", correct=None, skip_reason="context_limit")
            d.export([row], dict(videos=[video], predictions=old), root)
            paired = d.p.read_csv(root/"exports/paired_historical.csv")
            self.assertEqual(paired[9]["matched_questions"], "0")

    def test_package_hashes_snapshot_logs_and_excludes_asset_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"logs").mkdir()
            (root/"logs/diagnostic.log").write_text("fixture log")
            d.p.atomic_json(root/"progress.json", dict(stage="fixture"))
            (root/"features").mkdir()
            (root/"features/big.pt").write_bytes(b"not for export")
            with contextlib.redirect_stdout(io.StringIO()):
                d.package(root, "failed_partial")
            with ZipFile(root/d.ARTIFACT) as archive:
                manifest = json.loads(archive.read("evidence_manifest.json"))
                self.assertEqual(manifest["status"], "failed_partial")
                self.assertNotIn("features/big.pt", archive.namelist())
                self.assertIn("captured_logs/progress_snapshot.json", archive.namelist())
                for item in manifest["files"]:
                    import hashlib
                    self.assertEqual(hashlib.sha256(archive.read(item["path"])).hexdigest(), item["sha256"])

    def test_all_notebook_cells_validate_without_network_or_gpu(self):
        value = json.loads(notebook.DEST.read_text(encoding="utf-8"))
        namespace = {}
        with patch.dict(os.environ, HERMES_DIAGNOSTIC_VALIDATE_ONLY="1"), contextlib.redirect_stdout(io.StringIO()) as out:
            for i, cell in enumerate(value["cells"]):
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), f"notebook_cell_{i}", "exec"), namespace)
        self.assertIn("VALIDATION_PASS", out.getvalue())
        self.assertNotIn("torch", namespace)
        self.assertEqual(namespace["RUNNER_SOURCE"],(notebook.ROOT/"scripts/autodl_input_diagnostic.py").read_text(encoding="utf-8"))
        self.assertEqual(namespace["UTILS_SOURCE"],(notebook.ROOT/"scripts/autodl_failure_pilot.py").read_text(encoding="utf-8"))
        reference = json.loads(namespace["REFERENCE_JSON"])
        self.assertEqual(len(reference["videos"]), 8)
        self.assertEqual(len(reference["predictions"]), 120)
        d.check_reference(reference, notebook.ROOT)
        self.assertEqual(d.fingerprint(reference), d.fingerprint(notebook.reference_data()))


class DiagnosticNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import transformers
        if transformers.__version__ != "4.45.0.dev0":
            raise unittest.SkipTest("Pinned Transformers required for native decoder test")
        cls.torch = torch
        torch.set_num_threads(1)

    def model(self):
        from test_autodl_failure_pilot import NativeScorerTests
        obj = NativeScorerTests.model(self)
        obj.visual_start_idx, obj.conv_history = 5, []
        return obj

    def test_fragmented_feature_ingestion_matches_native_and_has_contiguous_positions(self):
        torch = self.torch
        with torch.inference_mode():
            whole, fragments = self.model(), self.model()
            features = torch.randn(1,196,64)
            d.p.ingest_features(whole, features, 1)
            for start, stop in ((0,57),(57,150),(150,196)):
                d.p.ingest_features(fragments, features[:,start:stop], 0)
            d.assert_fresh_cache(whole, 1)
            d.assert_fresh_cache(fragments, 1)
            for (ka,va),(kb,vb) in zip(whole.kv_cache, fragments.kv_cache):
                torch.testing.assert_close(ka,kb,atol=1e-6,rtol=1e-5)
                torch.testing.assert_close(va,vb,atol=1e-6,rtol=1e-5)
            fragments.conv_history = [("old question", "old answer")]
            with self.assertRaisesRegex(ValueError, "retained history"):
                d.assert_fresh_cache(fragments, 1)

    def test_safe_feature_slice_uses_original_frame_token_block(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = torch.arange(3*196*2,dtype=torch.float16).reshape(1,3*196,2)
            torch.save(value,root/"chunk.pt")
            result = d.read_features(root,dict(file="chunk.pt",start=1,stop=2))
            self.assertTrue(torch.equal(result,value[:,196:392]))
            torch.save(torch.zeros(1,10,2),root/"bad.pt")
            with self.assertRaises(ValueError):
                d.read_features(root,dict(file="bad.pt",start=0,stop=1))

    def test_question_execution_resume_and_native_skip_keep_complete_identity(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(torch.zeros(1,4*196,64,dtype=torch.float16),root/"chunk.pt")
            meta = dict(chunks=[dict(file="chunk.pt",start=0,stop=4,tokens=784,sha256=d.p.digest(root/"chunk.pt"))])
            args = types.SimpleNamespace(work=root,native=True,native_max_tokens=196)
            calls=[]
            def answer(prompt, max_new_tokens):
                self.assertEqual(max_new_tokens,16)
                self.assertNotIn("gold_choice",prompt)
                calls.append(prompt)
                return "B"
            model = types.SimpleNamespace(visual_start_idx=13,num_layers=3,conv_history=[],
                processor=types.SimpleNamespace(tokenizer=lambda text:types.SimpleNamespace(input_ids=[1,2])),
                language_model=types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=2048)),
                question_answering=answer,clear_cache=lambda:None,_get_cache_seq_len_per_layer=lambda:[797]*3)
            analyzer=types.SimpleNamespace(format_mcqa_prompt=lambda q,c:dict(prompt=q+str(c),question=q),
                                          extract_characters_regex=lambda text:text)
            with patch.object(d.p,"reset"),patch.object(d.p,"sync"),patch.object(d.p,"ingest_features") as ingest,\
                 patch.object(d,"assert_fresh_cache"),patch.object(torch.cuda,"reset_peak_memory_stats"),\
                 patch.object(torch.cuda,"max_memory_allocated",return_value=1024**3),contextlib.redirect_stdout(io.StringIO()):
                row=d.run_question(model,analyzer,fixture_video(),0,"recent_4",root,meta,"config",{},args)
                self.assertEqual(row["correct"],1)
                self.assertEqual(row["sampled_indices"],[0,1,2,3])
                self.assertEqual(ingest.call_count,1)
                resumed=d.run_question(model,analyzer,fixture_video(),0,"recent_4",root,meta,"config",{},args)
                self.assertEqual(row,resumed)
                self.assertEqual(len(calls),1)
                skipped=d.run_question(model,analyzer,fixture_video(),0,"native_prefix",root,meta,"config",{},args)
                self.assertEqual(skipped["skip_reason"],"configured_native_cap")
                self.assertIsNone(skipped["correct"])
                self.assertEqual(skipped["sampled_indices"],[0,1,2,3])  # never truncated


if __name__ == "__main__":
    unittest.main()
