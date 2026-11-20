"""Numerical and notebook checks for the construction selector."""
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

import autodl_construction_selection as candidate
import make_construction_selection_notebook as builder
import test_stage_attribution as fixtures


class SelectionTests(unittest.TestCase):
    def test_notebook_and_embedded_cli(self):
        nb = json.loads(builder.DEST.read_text(encoding="utf-8"))
        self.assertEqual(nb, builder.notebook())
        ns = {}
        with patch.dict(os.environ, HERMES_SELECTION_VALIDATE_ONLY="1"), contextlib.redirect_stdout(io.StringIO()) as out:
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), cell["id"], "exec"), ns)
        self.assertIn("VALIDATION_PASS", out.getvalue())
        self.assertEqual(len(json.loads(ns["REFERENCE_JSON"])["baseline_choices"]), 80)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            state = dict(ns, WORK=work, VALIDATE_ONLY=False)
            lock = types.SimpleNamespace(flock=lambda *args: None, LOCK_EX=1, LOCK_NB=2)
            with patch.dict(sys.modules, fcntl=lock), contextlib.redirect_stdout(io.StringIO()):
                exec("".join(nb["cells"][8]["source"]), state)
            state["session_lock"].close()
            result = subprocess.run([sys.executable, str(work / "stage_runner.py"), "--help"],
                                    cwd=work, check=True, capture_output=True, text=True)
            self.assertIn("--max-hours", result.stdout)

    def test_grouped_greedy_matches_direct_attention(self):
        import torch

        rng = torch.Generator().manual_seed(19)
        logits = [torch.randn(1, 3, 5, n, generator=rng, dtype=torch.float32) for n in (7, 2, 3, 3)]
        values = [torch.randn(1, 3, n, 4, generator=rng, dtype=torch.float32) for n in (7, 2, 3, 3)]
        groups = [candidate.attention_stats(l, v) for l, v in zip(logits, values, strict=True)]

        def direct(group_ids):
            return torch.cat([logits[i] for i in group_ids], -1).softmax(-1) @ torch.cat([values[i] for i in group_ids], -2)

        full = direct(range(4))
        merged = groups[0]
        for group in groups[1:]:
            merged = candidate.merge_stats(merged, group)
        torch.testing.assert_close(merged[1], full, atol=1e-6, rtol=1e-6)
        expected = []
        for _ in range(2):
            expected.append(min((i for i in range(1, 4) if i not in expected),
                                key=lambda i: (float((direct([0, *expected, i]) - full).square().mean()), -i)))
        selected, mass, _ = candidate.select_groups(groups[0], groups[1:])
        self.assertEqual(selected, [i-1 for i in expected])
        weights = torch.cat(logits, -1).softmax(-1)
        expected_mass = sorted(range(3), key=lambda i: (float(weights.split([7, 2, 3, 3], dim=-1)[i+1].sum(-1).mean()), i),
                               reverse=True)[:2]
        self.assertEqual(mass, expected_mass)


class SelectionNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.StageNumericalTests.setUpClass()
        cls.torch = fixtures.StageNumericalTests.torch

    def fixture(self):
        return fixtures.StageNumericalTests.fixture(self)

    def test_full_reference_and_real_subset_answer_positions(self):
        torch = self.torch
        s = candidate.s
        model, bank, meta, initial, video, analyzer = fixtures.StageNumericalTests.answer_fixture(self)
        bank.x = torch.cat((bank.x, 0.5 * bank.x), dim=1)
        meta["chunks"][0]["stop"] = 4
        history, recent = [0, 1, 2], [3]
        with (torch.inference_mode(), patch.object(s.p, "reset", side_effect=lambda m: s.restore(m, initial)),
              patch.object(s.p, "sync"), contextlib.redirect_stdout(io.StringIO())):
            reference, _ = candidate.q.m.prepare_states(model, bank, meta, history, recent)
            prefix = s.prefix_state(reference["A"], 5 + 3 * 196)
            all_kv, _, _ = candidate.construction(model, bank, meta, prefix, recent, [0, 1, 2], 5 + 3 * 196)
            s.restore(model, all_kv)
            s.assert_state(model, reference["A"])
            rows, detail = candidate.question(model, analyzer, bank, meta, video, 0, history, recent, {})
        self.assertEqual(len(detail["score_trace"]["attention_mass"]), 3)
        self.assertEqual(len(rows[0]["selected_history"]), 2)
        self.assertEqual(len({r["query_kv_bytes"] for r in rows}), 1)
        for row in rows[:-1]:
            self.assertEqual(row["position_trace"][0]["logical_start"], 5 + 4 * 196)
        self.assertEqual(rows[-1]["position_trace"][0]["logical_start"], 5 + 196)
