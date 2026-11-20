"""Reject damaged/ambiguous evidence and text/gold misalignment."""

import copy
import hashlib
import io
import json
import unittest
import warnings
from zipfile import ZipFile

from validate_causal_pilot_evidence import check_manifest, check_prediction


class EvidenceAuditTests(unittest.TestCase):
    def archive(self, *, payload=b"saved result", declared=None, duplicate=False, name="exports/result.txt"):
        content = payload if declared is None else declared
        manifest = {"status": "complete", "files": [{"path": name, "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest()}]}
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as z:
            z.writestr(name, payload)
            if duplicate:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    z.writestr(name, payload)
            z.writestr("evidence_manifest.json", json.dumps(manifest))
        buffer.seek(0)
        return ZipFile(buffer)

    def test_good_manifest(self):
        with self.archive() as z:
            manifest, payload = check_manifest(z)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(payload["exports/result.txt"], b"saved result")

    def test_modified_payload_rejected(self):
        with self.archive(declared=b"different result") as z, self.assertRaises(AssertionError):
            check_manifest(z)

    def test_duplicate_member_rejected(self):
        with self.archive(duplicate=True) as z, self.assertRaises(AssertionError):
            check_manifest(z)

    def test_unsafe_archive_path_rejected(self):
        for name in ("../outside.txt", "/absolute.txt", "folder\\outside.txt"):
            with self.subTest(name=name), self.archive(name=name) as z, self.assertRaises(AssertionError):
                check_manifest(z)

    def test_numeric_gold_and_prediction_identity(self):
        video = dict(video_id=7, conversations=[dict(question="How many?", choices=["1", "2"],
                     answer="1", task="Counting", end_time=2)])
        row = dict(video_id="7", qa_index=0, question="How many?", choices=["1", "2"],
                   gold_answer="1", gold_choice="A", task="Counting", query_time_s=2,
                   pred_answer="A) 1", pred_choice="A", correct=1, answer_s=.1)
        check_prediction(row, video, 0)
        for changes in ({"question": "Different?"}, {"gold_answer": 1}, {"pred_choice": "B"}, {"correct": 0}):
            with self.subTest(changes=changes), self.assertRaises(AssertionError):
                check_prediction({**row, **changes}, video, 0)


if __name__ == "__main__":
    unittest.main()
