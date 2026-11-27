"""Freeze a source-separated, controlled open/closed state pilot from DROID."""
import argparse
import json
from pathlib import Path

from decord import VideoReader
from PIL import Image, ImageDraw

from prepare_droid_state_control import INDEX, WORK
from validate_twinpast_diagnostic_evidence import ROOT, digest

DEST = ROOT / "data/twinpast/droid_state_control_v1.json"
# Reviewed from both external camera previews without running any model.
REVIEWED = (
    ("GuptaLab+553d1bd5+2023-06-18-16h-54m-33s", "ext1", "top drawer of the silver cabinet",
     (6, 14), "open", (88, 96), "closed"),
    ("IPRL+7790ec0a+2023-06-27-21h-01m-12s", "ext2", "white door with the dark handle",
     (9, 17), "closed", (178, 190), "open"),
    ("IRIS+7dfa2da3+2023-05-23-11h-37m-06s", "ext1", "white wall cabinet door",
     (24, 32), "closed", (345, 355), "open"),
    ("PennPAL+acda9df3+2023-06-29-20h-29m-07s", "ext2", "wooden drawer just below the countertop",
     (26, 35), "closed", (296, 310), "open"),
    ("REAL+de601749+2023-05-20-15h-07m-52s", "ext2", "top white kitchen drawer",
     (11, 20), "open", (127, 137), "closed"),
    ("TRI+52ca9b6a+2023-10-19-15h-04m-59s", "ext2", "white closet door in front of the robot",
     (16, 26), "open", (174, 188), "closed"),
)

TIMELINE = {
    "history_grid_s": [0, 3, 6, 9, 12, 15],
    "recent_grid_s": [16, 17, 18, 19],
    "query_s": 20,
    "neutral_rgb": [112, 112, 112],
    "aligned": {"cue_start_s": 8.75, "cue_end_s": 9.25, "oracle_grid_s": [0, 3, 6, 9, 12, 15]},
    "off_grid": {"cue_start_s": 9.25, "cue_end_s": 9.75, "oracle_grid_s": [0, 3, 6, 9.5, 12, 15]},
    "cue_representation": "One RGB video frame held for the entire half-open cue interval.",
}


def freeze(work):
    intake = {row["episode_id"]: row for row in json.loads((work / "intake.json").read_bytes())}
    out = work / "frozen_v1"
    out.mkdir(exist_ok=True)
    cases = []
    for episode_id, camera, target, early, state_early, late, state_late in REVIEWED:
        episode = intake[episode_id]
        source = episode["cameras"][camera]
        path = work / episode_id / f"{camera}.mp4"
        assert path.stat().st_size == source["bytes"] and digest(path.read_bytes()) == source["sha256"]
        reader = VideoReader(str(path), num_threads=1)
        assert 0 <= early[0] <= early[1] < late[0] <= late[1] < len(reader)
        assert len(reader) == source["frame_count"] and {state_early, state_late} == {"open", "closed"}
        pair_id = f"droid_{episode['site'].lower()}"
        options = ["Open", "Closed"]
        branches = {}
        sheet = Image.new("RGB", (960, 292), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((6, 5), episode_id + " | " + target, fill="black")
        for row, (side, bounds, state) in enumerate((("a", early, state_early), ("b", late, state_late))):
            mid = (bounds[0]+bounds[1])//2
            frame_indices = [bounds[0], mid, bounds[1]]
            frames = reader.get_batch(frame_indices).asnumpy()
            timestamp = float(reader.get_frame_timestamp([mid])[0][0])
            branches[side] = dict(state=state, gold="AB"[options.index(state.title())],
                                  source_interval_inclusive=list(bounds), source_frame=mid,
                                  source_frame_timestamp_s=timestamp, decoded_rgb_sha256=digest(frames[1].tobytes()))
            for column, (index, frame) in enumerate(zip(frame_indices, frames, strict=True)):
                image = Image.fromarray(frame)
                image.thumbnail((316, 108))
                x, y = column*320, 23+row*133
                sheet.paste(image, (x, y))
                draw.text((x+5, y+109), f"{side.upper()} | {state} | source frame {index}", fill="black")
        sheet.save(out / f"{pair_id}.jpg", quality=90)
        cases.append(dict(pair_id=pair_id, source_group=episode["site"], source_episode=episode_id,
                          scene_id=episode["scene_id"], camera=camera, video_source=source,
                          metadata_source=episode["metadata_source"], object=target, options=options,
                          question=f"In the earlier view, was the {target} open or closed?", branches=branches))
    assert len(cases) == len({c["source_group"] for c in cases}) == len({c["scene_id"] for c in cases}) == 6
    assert [c["branches"]["a"]["gold"] for c in cases].count("A") == 3
    for name in ("aligned", "off_grid"):
        spec = TIMELINE[name]
        within = lambda slots: sum(spec["cue_start_s"] <= t < spec["cue_end_s"] for t in slots)
        assert within(spec["oracle_grid_s"]) == 1
        assert within(TIMELINE["history_grid_s"]) == int(name == "aligned")
        assert len(spec["oracle_grid_s"]) == len(TIMELINE["history_grid_s"]) == 6
        assert max(spec["oracle_grid_s"]) < min(TIMELINE["recent_grid_s"])
    return dict(protocol="DROID open/closed historical-state capability, v1", dataset="DROID raw 1.0.1",
                dataset_license="CC BY 4.0", dataset_reference="https://droid-dataset.github.io/",
                source_index_sha256=INDEX, scope="Controlled virtual timelines with real RGB evidence and neutral filler.",
                timeline=TIMELINE, cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--out", type=Path, default=DEST)
    args = parser.parse_args()
    pilot = freeze(args.work)
    raw = json.dumps(pilot, indent=2) + "\n"
    args.out.write_text(raw, encoding="utf-8", newline="\n")
    print(f"Frozen {len(pilot['cases'])} site-separated pairs: {args.out}")


if __name__ == "__main__":
    main()
