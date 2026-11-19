"""Identity and selection safeguards for the research diagnostic builder."""

import csv
import io
import unittest

from build_failure_pilot import load_predictions, select_cases


class PredictionIdentityTests(unittest.TestCase):
    def setUp(self):
        self.annotations = [dict(video_id=9, conversations=[dict(
            question="Which?", choices=["same", "same", "other"],
            answer="same", task="test")])]
        self.row = dict(video_id="9", qa_index="0", question="Which?",
                        choices="['same', 'same', 'other']", answer="same",
                        correct_choice="A", pred_answer="A) same", pred_choice="A",
                        qa_acc="100.0", correct="1", task="test")

    def data(self, rows):
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(self.row))
        writer.writeheader()
        writer.writerows(rows)
        return stream.getvalue().encode()

    def test_first_matching_duplicate_gold_is_preserved(self):
        loaded = load_predictions(self.data([self.row]), self.annotations)
        self.assertEqual(loaded[("9", 0)]["correct"], 1)
        with self.assertRaises(ValueError):
            load_predictions(self.data([{**self.row, "correct_choice": "B"}]), self.annotations)

    def test_missing_duplicate_and_misaligned_rows_rejected(self):
        for rows in ([], [self.row, self.row], [{**self.row, "qa_index": "1"}],
                     [{**self.row, "question": "Different?"}],
                     [{**self.row, "qa_acc": "0"}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                load_predictions(self.data(rows), self.annotations)


class SelectionTests(unittest.TestCase):
    def rows(self):
        rows = []
        for outcome in ("hermes_only", "fifo_only", "both_wrong", "both_correct"):
            for i in range(12):
                rows.append(dict(case_id=f"{outcome}_{i}", video_id=str(i),
                                 task=f"task{i % 3}", outcome=outcome,
                                 duplicate_gold_options=i == 0))
        return rows

    def test_selection_is_order_independent_and_video_distinct(self):
        quotas = {o: 2 for o in ("hermes_only", "fifo_only", "both_wrong", "both_correct")}
        a = select_cases(self.rows(), quotas=quotas)
        b = select_cases(list(reversed(self.rows())), quotas=quotas)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 8)
        self.assertEqual(len({r["video_id"] for r in a}), 8)
        self.assertFalse(any(r["duplicate_gold_options"] for r in a))

    def test_shortage_is_reported_instead_of_reusing_videos(self):
        with self.assertRaisesRegex(ValueError, "Insufficient distinct videos"):
            select_cases(self.rows(), quotas={"hermes_only": 12})


if __name__ == "__main__":
    unittest.main()
