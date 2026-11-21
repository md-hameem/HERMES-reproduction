"""Check frozen frame identities, review rendering and notebook execution."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import autodl_confirmation_visual_review as reviewer
import make_confirmation_visual_review_notebook as builder


class ReviewTests(unittest.TestCase):
    def test_frozen_cases_and_headless_cells(self):
        nb = json.loads(builder.DEST.read_bytes())
        self.assertEqual(nb, builder.notebook())
        ns = {}
        with patch.dict(os.environ, HERMES_VISUAL_REVIEW_VALIDATE_ONLY="1"), contextlib.redirect_stdout(io.StringIO()) as output:
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), cell["id"], "exec"), ns)
        self.assertIn("VISUAL_REVIEW_VALIDATION_PASS", output.getvalue())
        self.assertEqual(len(ns["cases"]), 12)
        for case in ns["cases"].values():
            self.assertFalse({"gold_choice", "correct_arm", "pred_choice"}.intersection(case))
            self.assertTrue(reviewer.validated_frames(case))

    def test_post_query_frames_rejected(self):
        case = dict(case_id="sample_1_q0", query_time_s=2,
                    frames=[dict(sampled_index=1, source_frame=30, start_s=2.01, region="recent")])
        with self.assertRaisesRegex(ValueError, "post-query"):
            reviewer.validated_frames(case)

    def test_render_notes_and_export(self):
        import numpy as np
        from PIL import Image

        class Reader:
            def __len__(self):
                return 5

            def get_frame_timestamp(self, idx):
                return np.asarray([[float(i), i+1.] for i in range(5)])[idx]

            def __getitem__(self, idx):
                return types.SimpleNamespace(asnumpy=lambda: np.full((32, 48, 3), 20*idx, dtype=np.uint8))

        cases = [
            dict(case_id=f"sample_1_q{i}", sample_id=1, video_sha256="test", query_time_s=4.2,
                 question="Which is <visible>?", choices=["One", "Two"], frames=[
                     dict(sampled_index=j, source_frame=j, start_s=float(j), region="older" if j == 0 else "recent")
                     for j in range(5)
                 ])
            for i in range(12)
        ]
        outcomes = [dict(case_id=c["case_id"], gold_choice="B", correct_arm="compact_recent4") for c in cases]
        reference = dict(archive_sha256="synthetic", cases=cases, outcomes=outcomes)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(reviewer, "reader_for", return_value=Reader()) as opened, contextlib.redirect_stdout(io.StringIO()):
                reviewer.generate(reference, root, root)
                notes = root/"review_notes.csv"
                with notes.open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                rows[0]["notes"] = "Keep this judgement"
                reviewer.write_csv(notes, reviewer.NOTES_FIELDS, rows)
                reviewer.generate(reference, root, root)
                self.assertEqual(opened.call_count, 2)
                reviewer.inspect_frame(reference, root, root, "sample_1_q0", 4)
                reviewer.package(root, root/"review.zip")
            self.assertIn("Keep this judgement", notes.read_text(encoding="utf-8"))
            report = (root/"index.html").read_text(encoding="utf-8")
            self.assertIn("&lt;visible&gt;", report)
            self.assertNotIn("correct_arm", report)
            with Image.open(root/"originals/sample_1_q0_sample_0004.png") as original:
                self.assertEqual(original.size, (48, 32))
            with ZipFile(root/"review.zip") as zipped:
                self.assertIsNone(zipped.testzip())
                self.assertIn("after_review/outcome_key.csv", zipped.namelist())
                self.assertEqual(sum(n.endswith(".jpg") for n in zipped.namelist()), 24)
