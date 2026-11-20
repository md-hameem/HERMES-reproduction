"""CPU checks for AutoDL selection, HTTP ZIP safety and native causal scoring.

Run with the pinned Transformers environment for numerical tests. Download and
selection tests also run in a standard-library environment.
"""

import copy
import contextlib
import io
import json
import pathlib
import tempfile
import time
import types
import unittest
import zipfile

import autodl_failure_pilot as pilot


class Response:
    def __init__(self, data, start, end, ignore=False):
        self.status_code = 200 if ignore else 206
        self.headers = {"Content-Range": f"bytes {start}-{end}/{len(data)}"}
        self.data, self.start, self.end, self.ignore = data, start, end, ignore

    @property
    def content(self):
        if self.ignore:
            raise AssertionError("A server ignoring Range must never have its body read")
        return self.data[self.start:self.end + 1]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class Session:
    def __init__(self, data, ignore=False):
        self.data, self.ignore = data, ignore

    def get(self, url, headers, **kwargs):
        start, end = map(int, headers["Range"].split("=")[1].split("-"))
        return Response(self.data, start, end, self.ignore)


class DataSafetyTests(unittest.TestCase):
    def test_heartbeat_records_liveness_and_stops_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            self.assertTrue(root.is_relative_to(pathlib.Path(tempfile.gettempdir()).resolve()))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                with pilot.ProgressMonitor(root, interval=0.01) as monitor:
                    pilot.report_progress("waiting_for_fixture", requested_bytes=100)
                    time.sleep(0.04)
                    value = json.loads((root / "progress.json").read_text(encoding="utf-8"))
                    self.assertEqual(value["stage"], "waiting_for_fixture")
                    self.assertEqual(value["requested_bytes"], 100)
                    self.assertGreaterEqual(value["seconds_since_stage_update"], 0)
            self.assertIn("HEARTBEAT", output.getvalue())
            self.assertFalse(monitor.thread.is_alive())
            self.assertIsNone(pilot._PROGRESS)

    def test_monitor_reports_failure_without_suppressing_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            self.assertTrue(root.is_relative_to(pathlib.Path(tempfile.gettempdir()).resolve()))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "fixture"):
                with pilot.ProgressMonitor(root):
                    raise ValueError("fixture")
            value = json.loads((root / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(value["stage"], "failed")
            self.assertEqual(value["error_type"], "ValueError")
            self.assertIsNone(pilot._PROGRESS)

    def test_remote_zip64_member_extraction(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            with archive.open("nested/sample_202_real.mp4", "w", force_zip64=True) as handle:
                handle.write(b"test-only-not-a-video" * 1000)
        blob = data.getvalue()
        reader = pilot.HTTPRangeFile("fixture", len(blob), Session(blob), block_size=37)
        with zipfile.ZipFile(reader) as archive:
            self.assertEqual(archive.read("nested/sample_202_real.mp4"), b"test-only-not-a-video" * 1000)

    def test_range_ignored_rejected_before_body(self):
        reader = pilot.HTTPRangeFile("fixture", 100, Session(b"x" * 100, ignore=True))
        with self.assertRaises(pilot.RangeUnsupported):
            reader.read(2)

    def test_chunk_boundaries_preserve_question_history(self):
        video = dict(video_path="sample_202_real.mp4", conversations=[dict(end_time=t) for t in (6, 46, 46, 49)])
        self.assertEqual(pilot.chunk_plan(video, 30), [dict(qa_index=0, start=0, stop=3),
                         dict(qa_index=1, start=3, stop=19), dict(qa_index=1, start=19, stop=23),
                         dict(qa_index=3, start=23, stop=25)])
        with self.assertRaises(ValueError):
            pilot.chunk_plan(video, 24)

    def test_checkpoint_rejects_corruption_and_wrong_config(self):
        video = dict(video_id=4, conversations=[dict(question="q", choices=["0", "1"], answer="0")])
        rows = [dict(video_id="4", qa_index=0, question="q", choices=["0", "1"], gold_choice="A",
                     gold_answer="0", correct=1, pred_choice="A")]
        value = dict(config_hash="test", status="complete", method="fifo", predictions=rows,
                     prediction_sha256=pilot.sha_predictions(rows))
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            self.assertTrue(root.is_relative_to(pathlib.Path(tempfile.gettempdir()).resolve()))
            path = root / "checkpoint.json"
            pilot.atomic_json(path, value)
            self.assertTrue(pilot.checkpoint_valid(path, "test", video))
            self.assertFalse(pilot.checkpoint_valid(path, "different", video))
            self.assertFalse(pilot.checkpoint_valid(path, "test", video, "hermes"))
            value["predictions"][0]["question"] = "bad"
            pilot.atomic_json(path, value)
            self.assertFalse(pilot.checkpoint_valid(path, "test", video))

    def test_evidence_package_contains_hash_verified_log_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            self.assertTrue(root.is_relative_to(pathlib.Path(tempfile.gettempdir()).resolve()))
            (root / "logs").mkdir()
            (root / "logs/pilot.log").write_text("fixture output\n", encoding="utf-8")
            pilot.atomic_json(root / "exports/completion.json", dict(status="partial_time_budget"))
            pilot.package(root, "partial_time_budget")
            with zipfile.ZipFile(root / "HERMES_causal_scorer_pilot_evidence.zip") as archive:
                manifest = json.loads(archive.read("evidence_manifest.json"))
                self.assertEqual(manifest["status"], "partial_time_budget")
                self.assertIn("captured_logs/pilot.log", archive.namelist())
                for row in manifest["files"]:
                    import hashlib
                    data = archive.read(row["path"])
                    self.assertEqual(len(data), row["bytes"])
                    self.assertEqual(hashlib.sha256(data).hexdigest(), row["sha256"])


class NativeScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import transformers
        if transformers.__version__ != "4.45.0.dev0":
            raise unittest.SkipTest("Numerical scorer tests require pinned Transformers 4.45.0.dev0")
        cls.torch = torch
        torch.set_num_threads(1)

    def model(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from inference.llavaov_hermes import LlavaOneVision_Hermes as H
        torch.manual_seed(9)
        config = Qwen2Config(vocab_size=100, hidden_size=64, intermediate_size=96,
                             num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=512, attention_dropout=0)
        config._attn_implementation = "sdpa"
        lm = Qwen2ForCausalLM(config).eval()
        obj = types.SimpleNamespace(language_model=lm, num_layers=3, device=torch.device("cpu"),
                                    dtype=torch.float32, _layer_position_ids={}, _hook_handles=[],
                                    _position_ids_cache=[None] * 3, kv_cache=None, total_processed_frames=0)
        for name in ("_build_position_ids", "_get_next_start_pos_per_layer", "_get_cache_seq_len_per_layer",
                     "_set_rotary_required_len", "_append_position_ids", "_append_position_ids_layer"):
            setattr(obj, name, types.MethodType(getattr(H, name), obj))
        obj.get_input_embeddings = lm.get_input_embeddings
        H._register_forward_hooks(obj)
        H._patch_rotary_embeddings(obj)
        ids = torch.tensor([[1, 3, 5, 7, 9]])
        obj.kv_cache = list(lm(input_ids=ids, use_cache=True).past_key_values)
        obj._position_ids_cache = [torch.arange(5) for _ in range(3)]
        return obj

    def test_causal_scores_equal_native_and_do_not_mutate_source(self):
        torch = self.torch
        with torch.inference_mode():
            model = self.model()
            # Different logical query positions, equal physical lengths.
            model._position_ids_cache[1] += 7
            ids = torch.tensor([[4, 6, 8, 10]])
            old = [(k.clone(), v.clone()) for k, v in model.kv_cache]
            a, b = pilot.causal_scores(model, ids), pilot.reference_scores(model, ids)
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)
                self.assertFalse(torch.any(torch.triu(x[..., -4:], diagonal=1)))
            for (ka, va), (kb, vb) in zip(old, model.kv_cache):
                self.assertTrue(torch.equal(ka, kb) and torch.equal(va, vb))
            self.assertFalse(model._layer_position_ids)

    def test_layerwise_scorer_supports_different_cache_lengths(self):
        torch = self.torch
        with torch.inference_mode():
            model = self.model()
            k, v = model.kv_cache[1]
            model.kv_cache[1] = (k[:, :, :3], v[:, :, :3])
            model._position_ids_cache[1] = torch.arange(3)
            result = pilot.causal_scores(model, torch.tensor([[3, 4]]))
            self.assertEqual([x.shape[-1] for x in result], [7, 5, 7])

    def test_native_equivalence_after_actual_summary_pruning(self):
        torch = self.torch
        from inference.llavaov_hermes import LlavaOneVision_Hermes as H
        with torch.inference_mode():
            model = self.model()
            model.streaming, model.long_term_threshold = True, 2
            model.token_activity_cache = [None] * 3
            model.visual_start_idx = 5
            for name in ("_sanitize_keep_indices", "_shrink_positions_and_rerotate_keys", "_prune_activity_cache"):
                setattr(model, name, types.MethodType(getattr(H, name), model))
            pilot.ingest_features(model, torch.randn(1, 32, 64), 16)
            keep = [torch.cat((torch.arange(5), torch.arange(37 - (2 if i == 2 else 3), 37))) for i in range(3)]
            H.apply_kv_cache_pruning_strict(model, keep)
            self.assertEqual(model._get_cache_seq_len_per_layer(), [8, 8, 8])
            ids = torch.tensor([[2, 4, 6]])
            for a, b in zip(pilot.causal_scores(model, ids), pilot.reference_scores(model, ids)):
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    def test_feature_ingestion_matches_original_encoder_wrapper(self):
        torch = self.torch
        from inference.llavaov_hermes import LlavaOneVision_Hermes as H
        with torch.inference_mode():
            a, b = self.model(), self.model()
            features = torch.randn(1, 6, 64)
            a.processor = types.SimpleNamespace(video_processor=lambda *args, **kw:
                                                 types.SimpleNamespace(pixel_values_videos=features))
            a.get_video_features = lambda pixels: pixels
            H.encode_video_chunk(a, torch.zeros(3, 2, 2, 3))
            pilot.ingest_features(b, features, frames=3)
            for (ka, va), (kb, vb) in zip(a.kv_cache, b.kv_cache):
                torch.testing.assert_close(ka, kb, atol=0, rtol=0)
                torch.testing.assert_close(va, vb, atol=0, rtol=0)
            for pa, pb in zip(a._position_ids_cache, b._position_ids_cache):
                self.assertTrue(torch.equal(pa, pb))


if __name__ == "__main__":
    unittest.main()
