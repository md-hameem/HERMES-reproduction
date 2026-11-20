"""Reject misleading saved input-diagnostic records without executing archived code."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import validate_input_diagnostic_evidence as audit


class SavedDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive = audit.ROOT/"evidence/archives/HERMES_input_diagnostic_evidence.zip"
        cls.notebook = audit.ROOT/"notebooks/executed/2026-11-20_hermes-input-diagnostic.ipynb"
        with ZipFile(cls.archive) as z:
            reference = json.loads(z.read("reference.json"))
            cls.video = next(v for v in reference["videos"] if v["sample_id"] == 204)
            cls.row = json.loads(z.read("checkpoints/recent_4/sample_204_q0.json"))["row"]

    def test_valid_record_and_wrong_correctness_label(self):
        audit.validate_row(self.row,self.video,0,"recent_4",self.row["feature_fingerprint"])
        row = copy.deepcopy(self.row)
        row["correct"] = 0
        with self.assertRaises(AssertionError):
            audit.validate_row(row,self.video,0,"recent_4",row["feature_fingerprint"])

    def test_future_frame_and_wrong_feature_identity_are_rejected(self):
        row = copy.deepcopy(self.row)
        row["sampled_indices"][-1] = 8
        with self.assertRaises(AssertionError):
            audit.validate_row(row,self.video,0,"recent_4",row["feature_fingerprint"])
        with self.assertRaises(AssertionError):
            audit.validate_row(self.row,self.video,0,"recent_4","wrong-features")

    def test_skipped_record_cannot_be_counted_as_an_answer(self):
        row = copy.deepcopy(self.row)
        row.update(method="support_oracle",sampled_indices=[],source_frame_indices=[],sampled_start_seconds=[],
                   visual_tokens=0,status="skipped",skip_reason="not_requested",pred_answer="",pred_choice="",correct=0)
        with self.assertRaises(AssertionError):
            audit.validate_row(row,self.video,0,"support_oracle",row["feature_fingerprint"])
        row["correct"] = None
        audit.validate_row(row,self.video,0,"support_oracle",row["feature_fingerprint"])

    def test_fresh_cache_lengths_must_match_exact_selected_input(self):
        row = copy.deepcopy(self.row)
        row["cache_lengths_before_answer"][0] += 196
        with self.assertRaises(AssertionError):
            audit.validate_row(row,self.video,0,"recent_4",row["feature_fingerprint"])

    def test_complete_archive_recomputes_subsets_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = audit.audit(self.archive,self.notebook,root/"audit.json",root/"tables")
            self.assertEqual(result["answered"],134)
            self.assertEqual(result["checkpoints"],200)
            self.assertEqual([r["correct"] for r in result["methods"]],[14,25,21,10,0])
            self.assertEqual(result["native_historical_pairs"][1]["historical_correct"],10)


if __name__ == "__main__":
    unittest.main()
