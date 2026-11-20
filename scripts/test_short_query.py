"""Verify physical compaction through the real QA wrapper and notebook entry point."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import autodl_short_query as short
import make_short_query_notebook as builder
import test_stage_attribution as fixtures


class ShortQueryTests(unittest.TestCase):
    def test_notebook_cells_and_runner_import(self):
        nb = json.loads(builder.DEST.read_text(encoding="utf-8"))
        self.assertEqual(nb, builder.notebook())
        ns = {}
        with patch.dict(os.environ, HERMES_SHORT_QUERY_VALIDATE_ONLY="1"), contextlib.redirect_stdout(io.StringIO()) as out:
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), cell["id"], "exec"), ns)
        self.assertIn("VALIDATION_PASS", out.getvalue())
        self.assertEqual(ns["REPEATS"], 2)
        self.assertEqual(len(json.loads(ns["REFERENCE_JSON"])["baseline_choices"]), 120)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            materialize_ns = dict(ns, WORK=work, VALIDATE_ONLY=False)
            lock = types.SimpleNamespace(flock=lambda *args: None, LOCK_EX=1, LOCK_NB=2)
            with patch.dict(sys.modules, fcntl=lock), contextlib.redirect_stdout(io.StringIO()):
                exec("".join(nb["cells"][8]["source"]), materialize_ns)
            materialize_ns["session_lock"].close()
            result = subprocess.run([sys.executable, str(work / "stage_runner.py"), "--help"],
                                    cwd=work, check=True, capture_output=True, text=True)
            self.assertIn("--repeats", result.stdout)


class ShortQueryNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.StageNumericalTests.setUpClass()
        cls.torch = fixtures.StageNumericalTests.torch

    def fixture(self):
        return fixtures.StageNumericalTests.fixture(self)

    def test_physical_cache_and_real_prefill_decode(self):
        torch = self.torch
        s = short.s
        model, bank, meta, initial, video, analyzer = fixtures.StageNumericalTests.answer_fixture(self)
        with (torch.inference_mode(), patch.object(s.p, "reset", side_effect=lambda m: s.restore(m, initial)),
              patch.object(s.p, "sync"), contextlib.redirect_stdout(io.StringIO())):
            states, detail = short.m.prepare_states(model, bank, meta, [0], [1])
            full = states["A"]
            state = short.make_short(full, 5, 201)
            self.assertEqual(state["positions"][0].tolist(), list(range(5)) + list(range(201, 397)))
            for old, new in zip(full["kv"], state["kv"]):
                for original, shortened in zip(old, new):
                    torch.testing.assert_close(shortened, torch.cat((original[:, :, :5], original[:, :, 201:]), dim=2), atol=0, rtol=0)
            records, orders = short.run_block(model, analyzer, video, 0, states, detail, state, "test", {}, 2, warmup=True)
        self.assertEqual(len(orders), 2)
        masked, reduced = records[1:3]
        self.assertEqual((masked["physical_tokens"], reduced["physical_tokens"]), (397, 201))
        self.assertEqual(masked["query_logical_start"], reduced["query_logical_start"])
        for a, b in zip(masked["traces"][0], reduced["traces"][0], strict=True):
            self.assertEqual(a["logical_start"], b["logical_start"])
            self.assertEqual(a["past_tokens"] - b["past_tokens"], 196)
        self.assertEqual(len(reduced["traces"]), 2)
        pairs = short.compare_rows(records)
        self.assertEqual(pairs[0]["median_kv_bytes_saved"], short.cache_bytes(full) - short.cache_bytes(state))
        self.assertEqual(pairs[1]["median_kv_bytes_saved"], 0)

    def test_numeric_preflight_preserves_the_strict_fp32_reference(self):
        torch = self.torch
        model, bank, meta, _ = self.fixture()
        model.dtype = torch.float16
        lm = model.language_model.half()

        def reset(target):
            target._layer_position_ids.clear()
            target.kv_cache = list(lm(input_ids=torch.tensor([[1, 3, 5, 7, 9]]), use_cache=True).past_key_values)
            target._position_ids_cache = [torch.arange(5) for _ in range(model.num_layers)]

        with tempfile.TemporaryDirectory() as directory, torch.inference_mode(), patch.object(short.p, "reset", side_effect=reset):
            work = Path(directory)
            short.preflight(model, bank, meta, torch.tensor([[2, 4, 6]]), work)
            report = json.loads((work / "exports/short_preflight.json").read_text())
        self.assertEqual(set(report), {"torch.float16", "torch.float32"})
        self.assertTrue(all(r["argmax_equal"] and r["exceeds_active_bound"] == 0 for r in report["torch.float32"]))
