"""Check the frozen media plan, causal sampling, real QA and notebook entry point."""
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

import autodl_short_query_confirmation as runner
import make_short_query_confirmation_notebook as builder
import test_stage_attribution as fixtures


class ConfirmationTests(unittest.TestCase):
    def test_source_and_analysis(self):
        manifest = builder.build()
        saved = json.loads(builder.OUTPUT.read_bytes())
        self.assertEqual(manifest, saved)
        self.assertEqual(len(runner.videos_for_manifest(saved, builder.ROOT)), 30)
        stat = runner.paired_interval({1: [1, 0], 2: [1, 1], 3: [-1, 0]}, draws=200)
        self.assertAlmostEqual(stat["difference_pp"], 100 * 2 / 6)
        self.assertEqual(stat["leave_one_video_min_pp"], 0.0)
        self.assertEqual(stat["leave_one_video_max_pp"], 75.0)

    def test_grid_never_uses_later_frame(self):
        source = [[.1, 1.0], [1.0, 2.2], [2.2, 3.75], [3.75, 5.1], [5.1, 7.0]]
        v = {"conversations": [{"end_time": 6.1}]}
        times = runner.plan_sample_grid(source, v)
        self.assertEqual([r["source_frame"] for r in times], [0, 1, 3, 4])
        self.assertTrue(all(r["start_s"] <= r["sampled_index"] * 2 for r in times))

    def test_headless_notebook_and_cli(self):
        notebook = json.loads(builder.DEST.read_text(encoding="utf-8"))
        self.assertEqual(notebook, builder.notebook())
        cells = notebook["cells"]
        ns = {}
        with patch.dict(os.environ, HERMES_CONFIRMATION_VALIDATE_ONLY="1"), contextlib.redirect_stdout(io.StringIO()) as out:
            for cell in cells:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), cell["id"], "exec"), ns)
        self.assertIn("VALIDATION_PASS", out.getvalue())
        self.assertEqual(json.loads(ns["REFERENCE_JSON"])["questions"], 150)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            lock = types.SimpleNamespace(flock=lambda *args: None, LOCK_EX=1, LOCK_NB=2)
            materialize = dict(ns, VALIDATE_ONLY=False, WORK=work)
            with patch.dict(sys.modules, fcntl=lock), contextlib.redirect_stdout(io.StringIO()):
                exec("".join(cells[8]["source"]), materialize)
            materialize["session_lock"].close()
            result = subprocess.run([sys.executable, str(work / "stage_runner.py"), "--help"],
                                    cwd=work, check=True, capture_output=True, text=True)
            self.assertIn("--max-hours", result.stdout)


class ConfirmationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.StageNumericalTests.setUpClass()
        cls.torch = fixtures.StageNumericalTests.torch

    def fixture(self):
        return fixtures.StageNumericalTests.fixture(self)

    def test_real_answer_paths_share_query_budget(self):
        torch = self.torch
        s = runner.s
        model, bank, meta, initial, video, analyzer = fixtures.StageNumericalTests.answer_fixture(self)
        bank.x = torch.cat((bank.x, .5 * bank.x), dim=1)
        meta["chunks"][0]["stop"] = 4
        with (torch.inference_mode(), patch.object(s.p, "reset", side_effect=lambda m: s.restore(m, initial)),
              patch.object(s.p, "sync"), contextlib.redirect_stdout(io.StringIO())):
            rows, _ = runner.construct_block(model, analyzer, bank, meta, video, 0, [0, 1, 2], [3], warmup=True)
        self.assertEqual([r["method"] for r in rows], list(runner.ARMS))
        self.assertEqual(len({r["query_kv_bytes"] for r in rows}), 1)
        self.assertEqual([r["selected_history_indices"] for r in rows], [[], [1, 2], [0, 1, 2]])
        self.assertEqual([r["position_trace"][0]["logical_start"] for r in rows], [201, 789, 789])
