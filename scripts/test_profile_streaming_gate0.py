"""CPU-only checks for the Gate 0 input and artifact contracts."""
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import profile_streaming_gate0 as gate


class Gate0Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clips = gate.inputs()

    def test_archived_inputs(self):
        self.assertEqual([(c["sample_id"], c["video_id"]) for c in self.clips], list(gate.CLIPS))
        self.assertEqual(sum(len(c["conversations"]) for c in self.clips), 30)
        for method in ("fifo", "hermes"):
            self.assertEqual(sum(len(c["expected"][method]) for c in self.clips), 30)

    def test_round_trip_all_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            for clip in self.clips:
                for arm, (method, budget) in gate.ARMS.items():
                    with self.subTest(sample=clip["sample_id"], arm=arm):
                        data = self.make_output(clip, arm, method, budget)
                        target = Path(tmp) / str(clip["sample_id"]) / arm
                        gate.write_outputs(target, data)
                        run = gate.validate_outputs(target, clip, arm)
                        self.assertEqual(run["peak_allocated_mib"], 100)
                        if budget == 4000:
                            parity = target / "parity.csv"
                            parity.write_text(parity.read_text().replace(",True", ",False", 1))
                            with self.assertRaisesRegex(ValueError, "archived answer differs"):
                                gate.validate_outputs(target, clip, arm)

    def test_staging_lifetime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clips = [dict(c, asset=dict(bytes=1, sha256="fixture")) for c in self.clips[:2]]
            for clip in clips:
                (root / Path(clip["video_path"]).name).write_bytes(b"x")
            args = SimpleNamespace(media_dir=root, scratch_dir=root / "scratch",
                                   output_dir=root / "result", model_path=root / "model")
            visited = []
            scratch = []

            def stage(media, target):
                if scratch:
                    self.assertFalse(scratch[-1].exists())
                scratch.append(target)
                (target / "frames.npy").write_bytes(b"fixture")

            def worker(argv, check):
                frames = Path(argv[argv.index("--frames") + 1])
                self.assertTrue(check and frames.is_file())
                sid, arm = argv[argv.index("--sample-id") + 1], argv[argv.index("--arm") + 1]
                visited.append((int(sid), arm))
                (args.output_dir / sid).mkdir(exist_ok=True)

            with patch.object(gate, "sha256", return_value="fixture"), patch.object(
                    gate, "stage_clip", side_effect=stage), patch.object(gate.subprocess, "run", side_effect=worker):
                gate.run_all(args, clips)
            self.assertEqual(visited, [(c["sample_id"], arm) for c in clips for arm in gate.ARMS])
            self.assertTrue(all(not p.exists() for p in scratch))

    def test_first_token_hook(self):
        class MockModel:
            def question_answering(self):
                output_ids = []
                for step, token in enumerate([9, 10]):
                    output_ids.append(token)
                return output_ids

        model = MockModel()
        first = []
        gate.instrument_answer(model, first.append)
        self.assertEqual(model.question_answering(), [9, 10])
        self.assertEqual(first, [9])

    @staticmethod
    def make_output(clip, arm, method, budget):
        identity = dict(video_id=clip["video_id"], arm=arm)
        data = {k: [] for k in (*gate.CSV_FIELDS, *gate.JSONL_FIELDS)}
        threshold = 31744

        def stage(name, q, chunk=-1, frames="", tokens=""):
            data["stages"].append(dict(identity, stage=name, qa_index=q, chunk_index=chunk, calls=1,
                                       seconds=.1, entry_allocated_mib=80, peak_allocated_mib=100,
                                       peak_reserved_mib=110, frames=frames, visual_tokens=tokens))

        cursor = chunk = 0
        stage("other", -1)
        for qi, q in enumerate(clip["conversations"]):
            stop = math.ceil(q["end_time"] * .5)
            while cursor < stop:
                end = min(stop, cursor + 16)
                frames = end - cursor
                pre = 13 + min(budget, 196 * cursor) + frames * 196
                post = 13 + min(budget, 196 * end)
                compacted = 18 <= chunk < 18 + gate.ARCHIVED_RESETS[clip["video_id"]]
                base = dict(identity, qa_index=qi, chunk_index=chunk, sampled_start=cursor,
                            sampled_stop=end, timestamp_s=(end - 1) * 2)
                max_pos = threshold + 1 if compacted else pre-1
                data["cache"].append(dict(base, pre_slots=[pre], post_slots=[post],
                                          max_pre_position=max_pos, compacted=compacted))
                if compacted:
                    data["rope"].append(dict(base, pre_positions=[list(range(max_pos-pre+1, max_pos+1))],
                                             post_positions=[list(range(post))]))
                for name in ("preprocessing", "vision_tower", "chunk_prefill", "compression"):
                    stage(name, qi, chunk, frames, 196 * frames)
                cursor, chunk = end, chunk + 1
            stage("qa_prefill", qi)
            stage("qa_decode", qi)
            expected = clip["expected"][method][qi]
            data["parity"].append(dict(identity, qa_index=qi, **expected,
                                      archived_choice=expected["choice"] if budget == 4000 else "",
                                      choice_matches=True if budget == 4000 else ""))
            data["latency"].append(dict(identity, qa_index=qi, ttft_s=.1, answer_s=.2, first_token_id=10))
        data["run"] = dict(identity, gpu="mock", pytorch="mock", cuda="mock", dtype="fp16", model="mock",
                           archive_sha256=gate.ARCHIVE_SHA256, media_sha256=clip["asset"]["sha256"],
                           rope_threshold=threshold, measurement_mode="mock",
                           peak_allocated_mib=100, peak_reserved_mib=110)
        return data


if __name__ == "__main__":
    unittest.main()
