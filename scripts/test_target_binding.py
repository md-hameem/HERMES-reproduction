"""Check the frozen design, candidate controls, CPU decoder and notebook bootstrap."""
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

import autodl_target_binding as runner
import make_target_binding_notebook as builder
import test_stage_attribution as fixtures


class TargetBindingTests(unittest.TestCase):
    def test_frozen_cohort(self):
        manifest = builder.build()
        self.assertEqual(manifest, json.loads(builder.OUTPUT.read_bytes()))
        videos = runner.frozen_videos(manifest, builder.ROOT)
        self.assertEqual(len(videos), 60)
        self.assertEqual(sum(len(v["conversations"]) for v in videos), 300)
        self.assertEqual(manifest["primary_questions"], 269)

    def test_conjunction_and_mean_use_the_same_inputs(self):
        self.assertEqual(runner.ARM_PROBES["conjunction"], runner.ARM_PROBES["mean"])
        scores = [dict(relevance=9, target=10, binding=-1), dict(relevance=0, target=4, binding=4)]
        chosen = runner.select(scores)
        self.assertEqual(chosen, dict(recent=0, previous=1, global_relevance=0, target_only=0, mean=0, conjunction=1))
        self.assertEqual(set(runner.select([{}]).values()), {0})
        self.assertEqual(runner.select([scores[1], scores[1]])["conjunction"], 0)

    def test_causal_window_balance(self):
        self.assertEqual(runner.candidate_windows(list(range(16)), list(range(16, 20))),
                         [list(range(start, start + 4)) for start in (16, 12, 8, 4, 0)])
        self.assertEqual(runner.candidate_windows([0, 1, 2], [3, 4, 5, 6]), [[3, 4, 5, 6]])

    def test_headless_and_exported_cli(self):
        notebook = json.loads(builder.DEST.read_bytes())
        self.assertEqual(notebook, builder.notebook())
        ns = {}
        with patch.dict(os.environ, HERMES_TARGET_BINDING_VALIDATE_ONLY="1"), contextlib.redirect_stdout(io.StringIO()) as out:
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), cell["id"], "exec"), ns)
        self.assertIn("VALIDATION_PASS", out.getvalue())
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            lock = types.SimpleNamespace(flock=lambda *args: None, LOCK_EX=1, LOCK_NB=2)
            materialize = dict(ns, VALIDATE_ONLY=False, WORK=work)
            with patch.dict(sys.modules, fcntl=lock), contextlib.redirect_stdout(io.StringIO()):
                exec("".join(notebook["cells"][8]["source"]), materialize)
            materialize["session_lock"].close()
            result = subprocess.run([sys.executable, str(work / "stage_runner.py"), "--help"],
                                    cwd=work, check=True, capture_output=True, text=True)
            self.assertIn("--max-hours", result.stdout)


class TargetBindingModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.StageNumericalTests.setUpClass()
        cls.torch = fixtures.StageNumericalTests.torch

    def fixture(self):
        return fixtures.StageNumericalTests.fixture(self)

    def test_original_answer_and_reversed_probes_on_matched_windows(self):
        torch = self.torch
        model, bank, meta, initial, video, analyzer = fixtures.StageNumericalTests.answer_fixture(self)
        bank.x = torch.cat([bank.x * factor for factor in (1, .25, .75, .5)], dim=1)
        meta["chunks"][0]["stop"] = 8
        video.update(shard="test", timestamps=[dict(start_s=2 * i) for i in range(8)])
        video["conversations"][0].update(end_time=16, question="Which setting?",
                                         choices=["secret option", "other option"], answer="secret option")
        seen = []

        class Tokenizer:
            eos_token_id = -1

            def encode(self, letter, add_special_tokens=False):
                return [7 if letter == "A" else 8]

            def __call__(self, text, return_tensors=None):
                if "Verification:" in text:
                    seen.append(text)
                ids = [2, 4 if "(A) Yes" in text else 5]
                if text.endswith(("\nA", "\nB")):
                    ids += self.encode(text[-1])
                return types.SimpleNamespace(input_ids=torch.tensor([ids]) if return_tensors else ids)

            def decode(self, ids, **kwargs):
                return "A"

        model.processor.tokenizer = Tokenizer()
        model.get_prompt = lambda query: query + "\n"
        with (torch.inference_mode(), patch.object(runner.p, "reset", side_effect=lambda m: runner.s.restore(m, initial)),
              patch.object(runner.p, "sync"), contextlib.redirect_stdout(io.StringIO())):
            block = runner.evaluate_question(model, analyzer, bank, meta, video, 0)
        self.assertEqual([c["indices"] for c in block["windows"]], [list(range(4, 8)), list(range(4))])
        self.assertEqual(len({r["query_kv_bytes"] for r in block["rows"]}), 1)
        self.assertTrue(all(c["position_trace"][0]["logical_start"] == 789 for c in block["windows"]))
        self.assertTrue(all(r["correct"] for r in block["rows"]))
        self.assertEqual(len(seen), 36)
        self.assertTrue(all("secret option" not in text for text in seen))
        self.assertTrue(all(len(c["order_margins"][key]) == 2 for c in block["windows"] for key in runner.PROBES))
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            runner.export([block], work, complete=True)
            result = json.loads((work / "exports/paired.json").read_bytes())
            self.assertEqual(result["primary"]["questions"], 1)
            self.assertEqual(result["primary"]["difference_pp"], 0)


if __name__ == "__main__":
    unittest.main()
