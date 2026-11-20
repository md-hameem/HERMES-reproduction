"""Saved context-map, label, skip and audit coverage rejection checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import validate_context_diagnostic_evidence as a


class ContextAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive=a.ROOT/"evidence/archives/HERMES_context_diagnostic_evidence.zip"
        cls.notebook=a.ROOT/"notebooks/executed/2026-11-20_hermes-context-diagnostic.ipynb"
        with ZipFile(cls.archive) as z:
            reference=json.loads(z.read("reference.json"))
            cls.video=next(v for v in reference["videos"] if v["sample_id"]==204)
            cls.row=json.loads(z.read("checkpoints/full_prefix_opening_fill/sample_204_q0.json"))["row"]

    def test_valid_fixed_position_map_and_wrong_tail(self):
        a.validate_row(self.row,self.video,0,"full_prefix_opening_fill",self.video["feature_meta"])
        row=copy.deepcopy(self.row);row["plan"]["sampled_indices"][-1]=6
        with self.assertRaises(AssertionError): a.validate_row(row,self.video,0,"full_prefix_opening_fill",self.video["feature_meta"])

    def test_future_source_or_changed_feature_identity_rejected(self):
        row=copy.deepcopy(self.row);row["source_start_seconds"][-1]=18
        with self.assertRaises(AssertionError): a.validate_row(row,self.video,0,"full_prefix_opening_fill",self.video["feature_meta"])
        row=copy.deepcopy(self.row);row["input_identity"]="different"
        with self.assertRaises(AssertionError): a.validate_row(row,self.video,0,"full_prefix_opening_fill",self.video["feature_meta"])

    def test_wrong_protected_position_or_chunk_order_rejected(self):
        row=copy.deepcopy(self.row);row["protected_tail_token_starts"][0]+=196
        with self.assertRaises(AssertionError): a.validate_row(row,self.video,0,"full_prefix_opening_fill",self.video["feature_meta"])
        row=copy.deepcopy(self.row);row["ingestion_groups"][0]=list(reversed(row["ingestion_groups"][0]))
        # This first group is all filler zero; corrupt its membership explicitly.
        row["ingestion_groups"][0][0]=1
        with self.assertRaises(AssertionError): a.validate_row(row,self.video,0,"full_prefix_opening_fill",self.video["feature_meta"])

    def test_wrong_correctness_or_answered_skip_rejected(self):
        row=copy.deepcopy(self.row);row["correct"]=0
        with self.assertRaises(AssertionError): a.validate_row(row,self.video,0,"full_prefix_opening_fill",self.video["feature_meta"])
        row=copy.deepcopy(self.row);row["status"]="skipped"
        with self.assertRaises(AssertionError): a.validate_row(row,self.video,0,"full_prefix_opening_fill",self.video["feature_meta"])

    def test_full_audit_recomputes_replay_and_control_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            result=a.audit(self.archive,self.notebook,root/"audit.json",root/"tables")
            self.assertEqual((result["checkpoints"],result["answered"],result["comparable_replays"]),(84,74,26))
            self.assertEqual(result["replay_choice_mismatches"],0)
            self.assertEqual(sum(r["matched_questions"] for r in result["pair_summary"][:3]),19)
            self.assertEqual(sum(r["choice_changes"] for r in result["pair_summary"][:3]),0)
            self.assertEqual(result["pair_summary"][5]["b_only_correct"],3)


if __name__=="__main__": unittest.main()
