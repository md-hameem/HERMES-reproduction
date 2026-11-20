"""Validate the matched intervention and its self-contained notebook."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import autodl_matched_boundary as matched
import make_matched_boundary_notebook as builder
import test_stage_attribution as stage_tests


class MatchedNotebookTests(unittest.TestCase):
    def test_all_cells_and_embedded_runner_import(self):
        saved = json.loads(builder.DEST.read_text(encoding="utf-8"))
        self.assertEqual(saved, builder.notebook())
        ns = {}
        with patch.dict(os.environ, HERMES_MATCHED_VALIDATE_ONLY="1"), contextlib.redirect_stdout(io.StringIO()) as out:
            for cell in saved["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), cell["id"], "exec"), ns)
        self.assertIn("VALIDATION_PASS", out.getvalue())
        self.assertFalse(ns["RUN_LONG_CASE"])
        self.assertEqual(len(json.loads(ns["REFERENCE_JSON"])["stage_rows"]), 200)
        embed = "".join(next(c for c in saved["cells"] if c["id"] == "matched-08")["source"])
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            # fcntl is Linux-specific. Exercise materialization on Windows with an inert lock.
            from types import SimpleNamespace
            lock = SimpleNamespace(flock=lambda *args: None, LOCK_EX=1, LOCK_NB=2)
            embed_ns = dict(ns, WORK=work, VALIDATE_ONLY=False)
            try:
                with patch.dict(sys.modules, fcntl=lock), contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(embed, "embed", "exec"), embed_ns)
                result = subprocess.run([sys.executable, str(work / "stage_runner.py"), "--help"],
                                        cwd=work, capture_output=True, text=True, check=True)
                self.assertIn("--max-hours", result.stdout)
                self.assertEqual((work / "provenance/stage_base.py").read_text(encoding="utf-8"), ns["STAGE_BASE_SOURCE"])
            finally:
                if "session_lock" in embed_ns:
                    embed_ns["session_lock"].close()


class MatchedNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stage_tests.StageNumericalTests.setUpClass()
        cls.torch = stage_tests.StageNumericalTests.torch

    def test_same_group_masks_and_historical_control_states(self):
        torch = self.torch
        model, bank, meta, initial = stage_tests.StageNumericalTests.fixture(self)
        s = matched.s
        with torch.inference_mode(), patch.object(s.p, "reset", side_effect=lambda m: s.restore(m, initial)):
            old, _ = s.prepare_states(model, bank, meta, [0], [1])
            new, detail = matched.prepare_states(model, bank, meta, [0], [1])
            self.assertTrue(detail["original_group_crosses_boundary"])
            self.assertEqual(detail["visible_recent_groups"], detail["hidden_recent_groups"])
            self.assertEqual(detail["visible_construction_mask_trace"][0]["shape"],
                             detail["hidden_construction_mask_trace"][0]["shape"])
            self.assertEqual(detail["visible_construction_mask_trace"][0]["blocked_stop"], 5)
            self.assertEqual(detail["hidden_construction_mask_trace"][0]["blocked_stop"], 201)
            self.assertFalse(torch.equal(new["A"]["kv"][-1][0][:, :, 201:], new["B"]["kv"][-1][0][:, :, 201:]))
            for name in ("B", "compact"):
                s.restore(model, new[name])
                s.assert_state(model, old[name])

    def test_preflight_checks_both_precisions_and_saves_receipt(self):
        torch = self.torch
        model, bank, meta, _ = stage_tests.StageNumericalTests.fixture(self)
        model.dtype = torch.float16
        lm = model.language_model.half()

        def reset(target):
            target._layer_position_ids.clear()
            target.kv_cache = list(lm(input_ids=torch.tensor([[1, 3, 5, 7, 9]]), use_cache=True).past_key_values)
            target._position_ids_cache = [torch.arange(5) for _ in range(model.num_layers)]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            with torch.inference_mode(), patch.object(matched.p, "reset", side_effect=reset):
                matched.numerical_preflight(model, bank, meta, torch.tensor([[2, 4, 6]]), path)
            report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["fp32"]["result"]["status"], "pass")
        self.assertEqual(len(report["fp16"]["logits"]), 9)
        self.assertTrue(report["fp16_masked_readout_exact"])
