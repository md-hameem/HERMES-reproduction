"""Render archived TwinPast answer frames and annotated reference frames on CPU."""
import argparse
import json
from pathlib import Path

from decord import VideoReader
import numpy as np
from PIL import Image, ImageDraw

from autodl_twinpast_diagnostic import digest as media_digest
from validate_twinpast_refinement_evidence import ARCHIVE, MANIFEST, ROOT, digest, verify_inventory

LOCAL = ROOT / "evidence/local"
LABELS = ("Annotated reference", "Coarse uniform", "Local uniform", "RGB change", "Shared recent")


def render(path, receipt, trace, intervals, question, destination):
    if path.stat().st_size != receipt["bytes"] or media_digest(path) != receipt["sha256"]:
        raise ValueError(f"Clip does not match the run manifest: {path}")
    reader = VideoReader(str(path), num_threads=1)
    times = np.asarray(reader.get_frame_timestamp(list(range(len(reader)))), dtype=np.float64)
    times -= times[0, 0]
    coarse, fine = trace["coarse"], trace["refined"]
    picks = trace["refined_picks"]
    method_slots = [(coarse, trace["coarse_uniform_slots"]), (fine, picks["local_uniform"]),
                    (fine, picks["state_change"]), (coarse, list(range(len(coarse["indices"])-4, len(coarse["indices"]))))]
    rows = [[data["indices"][i] for i in slots] for data, slots in method_slots]
    in_interval = [i for i, start in enumerate(times[:, 0])
                   if any(lo-1e-6 <= start <= hi+1e-6 for lo, hi in intervals)]
    oracle = [in_interval[i] for i in np.linspace(0, len(in_interval)-1, min(len(in_interval), 6), dtype=int)]
    indices = sorted(set(oracle).union(*rows))
    frames = dict(zip(indices, reader.get_batch(indices).asnumpy(), strict=True))
    for data, slots in method_slots:
        for i in slots:
            frame = frames[data["indices"][i]]
            if digest(frame.tobytes()) != data["hashes"][i]:
                raise ValueError(f"Archived RGB hash differs for {path} frame {data['indices'][i]}")

    width, height, pitch = 218, 123, 165
    board = Image.new("RGB", (width*6, 72+pitch*len(LABELS)), "white")
    draw = ImageDraw.Draw(board)
    draw.text((8, 7), question, fill="black")
    draw.text((8, 31), "Selected windows: " + str(trace["selected_windows"]) +
              "   Annotated intervals: " + (str(intervals) if intervals else "unavailable"), fill="black")
    for row, (label, frames_in_row) in enumerate(zip(LABELS, [oracle]+rows, strict=True)):
        top = 65+row*pitch
        draw.text((8, top), label, fill="black")
        for col, index in enumerate(frames_in_row):
            rgb = Image.fromarray(frames[index])
            rgb.thumbnail((width-4, height))
            left = col*width
            board.paste(rgb, (left+2, top+18))
            timestamp = times[index, 0]
            hit = any(lo-1e-6 <= timestamp <= hi+1e-6 for lo, hi in intervals)
            draw.text((left+4, top+height+19), f"{timestamp:.3f} s" + ("  *" if hit else ""), fill="black")
    board.save(destination, quality=90)
    temporal_support = None
    if intervals:
        def intersects(frame_indices):
            return any(lo-1e-6 <= times[i, 0] <= hi+1e-6
                       for i in frame_indices for lo, hi in intervals)

        windows = [candidate["slots"] for candidate in trace["candidates"]]
        candidate_grid = [start + step/4 for start, end in windows for step in range(4*(end-start)+1)]
        candidate_indices = np.searchsorted(times[:, 0], candidate_grid, side="right") - 1
        assert min(candidate_indices) >= 0 and max(candidate_grid) <= times[-1, 1] + 1e-6
        temporal_support = dict(all_candidate_windows=intersects(candidate_indices),
                                selected_windows=intersects(fine["indices"]),
                                coarse_uniform=intersects(rows[0]), local_uniform=intersects(rows[1]),
                                state_change=intersects(rows[2]))
    return dict(frame_indices_rendered=len(indices), timestamp_overlap=temporal_support)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", type=Path, default=LOCAL / "twinpast-refinement-media",
                        help="Verified clips arranged as pair_id/A.mp4 and pair_id/B.mp4")
    parser.add_argument("--output", type=Path, default=LOCAL / "twinpast-refinement-review")
    args = parser.parse_args()
    saved = verify_inventory(ARCHIVE)
    manifest = json.loads(MANIFEST.read_bytes())
    assert json.loads(saved["provenance/refinement_cases.json"])["manifest_sha256"] == digest(MANIFEST.read_bytes())
    annotations = {row["pair_id"]: row for row in manifest["candidates"]}
    media = {plan["pair_id"]: entry["decoded"] for plan, entry, _ in manifest["media_plan"]}
    args.output.mkdir(parents=True, exist_ok=True)
    review = []
    for pair in json.loads(saved["exports/run_config.json"])["accepted"]:
        row = annotations[pair]
        block = json.loads(saved[f"checkpoints/{pair}.json"])["block"]
        for side in "ab":
            branch = side.upper()
            reviewed = render(args.media / pair / f"{branch}.mp4", media[pair][side], block["branches"][side],
                              (row["evidence_intervals"] or {}).get(branch) or [], row["question"],
                              args.output / f"{pair}_{branch}.jpg")
            review.append(dict(pair_id=pair, branch=branch, **reviewed))
            print(f"{pair} {branch}: {reviewed['frame_indices_rendered']} distinct frame indices", flush=True)
    (args.output / "review_receipt.json").write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
