"""Freeze a real-history DROID paired control and package only its sampled frames."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from decord import VideoReader
import numpy as np
from PIL import Image, ImageDraw

from prepare_droid_natural_history import WORK
from prepare_droid_state_control import WORK as CONTROL
from validate_twinpast_diagnostic_evidence import ROOT, digest

DEST = ROOT / "data/twinpast/droid_natural_history_v1.json"
FRAME_ZIP = "DROID_natural_history_frames.zip"
DONOR = "CLVR+236539bc+2023-06-06-01h-09m-11s"
DONOR_FRAMES = (15, 75, 135, 195)
# Observed initial and terminal intervals, inclusive, in open then close episodes.
REVIEW = (
    ("8b01107c59", "ext2", "wooden drawer to the left of the stove", (0, 55), (230, 256), (0, 65), (164, 175)),
    ("6d74030dc0", "ext1", "wooden cabinet door just right of the range hood", (0, 85), (178, 280), (0, 38), (138, 171)),
    ("87abc314ec", "ext2", "wide white drawer below the countertop", (0, 180), (259, 320), (0, 38), (110, 151)),
    ("29137e19c7", "ext1", "drawer under the white counter beside the black sink", (0, 58), (166, 205), (0, 44), (130, 135)),
    ("c00374385e", "ext2", "wooden drawer just left of the white refrigerator", (0, 108), (285, 330), (0, 60), (180, 223)),
    ("19bf4a67ee", "ext1", "glass door of the blue countertop oven", (0, 110), (253, 313), (0, 69), (151, 184)),
    ("10bd743a70", "ext2", "glass-front cabinet door below the fruit basket", (0, 140), (344, 387), (0, 85), (255, 277)),
)


def reader_for(root, episode, camera):
    path = root / episode["episode_id"] / f"{camera}.mp4"
    receipt = episode["cameras"][camera]
    assert path.stat().st_size == receipt["bytes"] and digest(path.read_bytes()) == receipt["sha256"]
    reader = VideoReader(str(path), num_threads=1)
    assert len(reader) == receipt["frame_count"]
    return reader


def choose_history(starts, initial, terminal):
    assert 0 == initial[0] <= initial[1] < terminal[0] < terminal[1] < len(starts)
    last = terminal[1]
    ticks = np.linspace(starts[0], starts[last], 6)
    uniform = (np.searchsorted(starts, ticks, side="right") - 1).tolist()
    assert uniform[0] == 0 and uniform[-1] == last and len(set(uniform)) == 6
    already = sum(terminal[0] <= i <= terminal[1] for i in uniform)
    need = max(0, 3-already)
    available = [i for i in range(terminal[0], terminal[1]+1) if i not in uniform]
    replacements = [available[(j+1)*len(available)//(need+1)] for j in range(need)]
    removed = [i for i in uniform if 0 < i < terminal[0]][::-1][:need]
    assert len(removed) == need and len(set(replacements)) == need
    annotated = sorted((set(uniform)-set(removed)) | set(replacements))
    assert len(annotated) == 6 and annotated[0] == 0 and annotated[-1] == last
    return dict(uniform=uniform, annotated=annotated)


def encode(reader, indices, prefix, assets):
    frames = reader.get_batch(indices).asnumpy()
    names = {}
    for index, rgb in zip(indices, frames, strict=True):
        image = Image.fromarray(rgb).resize((640, 360), Image.Resampling.BILINEAR)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        name = f"{prefix}_{index:05}.png"
        assets[name] = buffer.getvalue(), digest(image.tobytes())
        names[index] = name
    return names


def review_sheet(case, assets, out):
    sheet = Image.new("RGB", (1200, 475), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 6), case["question"], fill="black")
    # A and B are counterbalanced. No gold or selection arm is shown here.
    for r, side in enumerate(("a", "b", "recent")):
        names = case["recent_frames"] if side == "recent" else case["branches"][side]["frames"]["uniform"]
        top = r*149+25
        draw.text((5, top), "Shared recent" if side == "recent" else f"Branch {side.upper()}, chronological", fill="black")
        for col, name in enumerate(names):
            with Image.open(BytesIO(assets[name][0])) as image:
                thumbnail = image.resize((196, 110))
                sheet.paste(thumbnail, (col*200, top+20))
    sheet.save(out, quality=90)


def freeze(work, output):
    intake = {r["pair_id"]: r for r in json.loads((work / "intake.json").read_bytes())}
    old = json.loads((CONTROL / "intake.json").read_bytes())
    old_scenes = {(r["site"], r["scene_id"]) for r in old}
    old_episodes = {r["episode_id"] for r in old}
    donor = next(row for row in old if row["episode_id"] == DONOR)
    assets = {}
    donor_reader = reader_for(CONTROL, donor, "ext2")
    recent = list(encode(donor_reader, list(DONOR_FRAMES), "droid_shared_recent", assets).values())
    assert len(recent) == 4 and len({assets[name][1] for name in recent}) == 4
    cases = []
    review = work / "frozen_review"
    review.mkdir(exist_ok=True)
    for i, (suffix, camera, target, early_open, late_open, early_close, late_close) in enumerate(REVIEW):
        identifier = "droid_natural_" + suffix
        row = intake[identifier]
        assert (row["site"], row["scene_id"]) not in old_scenes and camera in row["matched_cameras"]
        episodes = [(row["open_episode"], early_open, late_open, "A"),
                    (row["close_episode"], early_close, late_close, "B")]
        if i % 2:
            episodes.reverse()
        branches = {}
        for side, (episode, initial, terminal, gold) in zip("ab", episodes, strict=True):
            assert episode["episode_id"] not in old_episodes and episode["scene_id"] == row["scene_id"]
            reader = reader_for(work, episode, camera)
            stamps = np.asarray(reader.get_frame_timestamp(list(range(len(reader)))), dtype=np.float64)
            starts = stamps[:, 0] - stamps[0, 0]
            picks = choose_history(starts, initial, terminal)
            unique = sorted(set(picks["uniform"]) | set(picks["annotated"]))
            names = encode(reader, unique, f"{identifier}_{side}", assets)
            branches[side] = dict(source_episode=episode["episode_id"], video_source=episode["cameras"][camera],
                                  metadata_source=episode["metadata_source"], initial_interval=list(initial),
                                  terminal_interval=list(terminal), gold=gold, source_indices=picks,
                                  source_timestamps_s={m: starts[idx].tolist() for m, idx in picks.items()},
                                  frames={m: [names[k] for k in idx] for m, idx in picks.items()})
        case = dict(pair_id=identifier, source_site=row["site"], scene_id=row["scene_id"], camera=camera,
                    question=f"In the last view of the {target} before the scene changed, was it open or closed?",
                    options=["Open", "Closed"], branches=branches, recent_frames=recent)
        review_sheet(case, assets, review / f"{identifier}.jpg")
        cases.append(case)
    assert len(cases) == len({(c["source_site"], c["scene_id"]) for c in cases}) == 7
    assert len({b["source_episode"] for c in cases for b in c["branches"].values()}) == 14
    assert len({c["source_site"] for c in cases}) == 4
    frame_file = work / FRAME_ZIP
    with ZipFile(frame_file, "w", compression=ZIP_STORED) as archive:
        for name, (raw, _) in sorted(assets.items()):
            archive.writestr(ZipInfo(name), raw)
    manifest = dict(protocol="DROID natural state history capability v1", dataset="DROID raw 1.0.1",
                    dataset_reference="https://droid-dataset.github.io/", dataset_license="CC BY 4.0",
                    frame_zip_sha256=digest(frame_file.read_bytes()),
                    recent_donor=dict(source_episode=DONOR, source_site=donor["site"], camera="ext2",
                                      source_frames=DONOR_FRAMES, video_source=donor["cameras"]["ext2"]),
                    frame_receipts={name: dict(png_sha256=digest(raw), rgb_sha256=rgb_sha)
                                    for name, (raw, rgb_sha) in sorted(assets.items())}, cases=cases)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Frozen {len(cases)} scene pairs with {len(assets)} PNG frames in {frame_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEST)
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    freeze(args.work, args.out)
