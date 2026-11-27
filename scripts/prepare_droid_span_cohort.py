"""Acquire source-disjoint DROID candidates and render ten-slot review sheets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path

import requests

from prepare_droid_natural_history import WORK as NATURAL, metadata
from prepare_droid_state_control import WORK as CONTROL, acquire, candidates
from validate_twinpast_diagnostic_evidence import ROOT, digest

WORK = ROOT / "evidence/local/droid-span-cohort"
SOURCE_ERRORS = (requests.RequestException, OSError, KeyError, ValueError, RuntimeError)


def exclusions():
    episodes, scenes = set(), set()
    for root in (CONTROL, NATURAL):
        for path in sorted(root.glob("*/metadata.json")):
            row = json.loads(path.read_bytes())
            episodes.add(row["uuid"])
            scenes.add((row["lab"], row["scene_id"]))
        for row in json.loads((root / "intake.json").read_bytes()):
            scenes.add((row["site"], row["scene_id"]))
            if root == CONTROL:
                episodes.add(row["episode_id"])
            else:
                episodes.update(row[k]["episode_id"] for k in ("open_episode", "close_episode"))
    return episodes, scenes


def review_pair(work, pair):
    from decord import VideoReader
    from PIL import Image, ImageDraw

    grids = {}
    # Alternate the displayed ordering. The contact sheets omit source instructions.
    episodes = (pair["open_episode"], pair["close_episode"])
    if int(pair["pair_id"][-1], 16) % 2:
        episodes = episodes[::-1]
    for camera in pair["matched_cameras"]:
        view = Image.new("RGB", (1200, 610), "white")
        draw = ImageDraw.Draw(view)
        slots = {}
        for branch, (side, episode) in enumerate(zip("AB", episodes, strict=True)):
            reader = VideoReader(str(work / episode["episode_id"] / f"{camera}.mp4"), num_threads=1)
            if len(reader) < 10:
                raise ValueError("Episode has fewer than ten frames")
            indices = [i*(len(reader)-1)//9 for i in range(10)]
            for i, frame in enumerate(reader.get_batch(indices).asnumpy()):
                x, y = (i % 5)*240, branch*304+(i//5)*152
                frame = Image.fromarray(frame)
                frame.thumbnail((236, 133))
                view.paste(frame, (x, y))
                draw.text((x + 4, y + 134), f"{side}  slot {i}  frame {indices[i]}", fill="black")
            slots[side.lower()] = {"source_episode": episode["episode_id"], "source_frames": indices}
        sheet = work / f"{pair['pair_id']}_{camera}.jpg"
        view.save(sheet, quality=88)
        grids[camera] = slots
    return grids


def prepare(work, per_site):
    work.mkdir(parents=True, exist_ok=True)
    seen_episodes, seen_scenes = exclusions()
    prior_scene_count = len(seen_scenes)
    pools = {}
    for row in candidates(CONTROL, None):
        if row["episode_id"] in seen_episodes:
            continue
        collector = row["episode_id"].split("+")[1]
        key = (row["site"], collector)
        pools.setdefault(key, {"open": [], "close": []})[row["action"]].append(row)
    episodes = {r["episode_id"]: r for pool in pools.values() if pool["open"] and pool["close"]
                for rows in pool.values() for r in rows}
    values, unavailable = {}, []
    print(f"DROID_SPAN_SOURCE_SCREEN metadata_requests={len(episodes)} excluded_scenes={len(seen_scenes)}", flush=True)
    def cached_metadata(row):
        path = work / "metadata" / (row["episode_id"] + ".json")
        if path.is_file():
            return json.loads(path.read_bytes())
        value = metadata(row)
        path.write_text(json.dumps(value), encoding="utf-8")
        return value

    (work / "metadata").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        jobs = {key: pool.submit(cached_metadata, row) for key, row in episodes.items()}
        for key, job in jobs.items():
            try:
                values[key] = job.result()
            except SOURCE_ERRORS as exc:
                unavailable.append({"episode": key, "error": type(exc).__name__})
    groups = {}
    for episode, value in values.items():
        row = episodes[episode]
        _, collector, _ = episode.split("+")
        scene = (row["site"], value["scene_id"])
        if value["success"] and scene not in seen_scenes:
            key = (*scene, collector)
            groups.setdefault(key, {"open": [], "close": []})[row["action"]].append(row)

    def start(row):
        return datetime.strptime(row["episode_id"].split("+")[2], "%Y-%m-%d-%Hh-%Mm-%Ss")

    combos = []
    for key, group in groups.items():
        for a in group["open"]:
            for b in group["close"]:
                x, y = (values[r["episode_id"]] for r in (a, b))
                cameras = [cam for cam in ("ext1", "ext2") if x[f"{cam}_cam_serial"] == y[f"{cam}_cam_serial"]]
                if cameras:
                    combos.append((a["target"] != b["target"], abs((start(a)-start(b)).total_seconds()),
                                   a["episode_id"], b["episode_id"], key[:2], cameras))
    accepted = []
    for site in sorted({combo[4][0] for combo in combos}):
        total = 0
        for _, _, a, b, scene, cameras in sorted(combos):
            if scene[0] != site or scene in seen_scenes:
                continue
            accepted.append((episodes[a], episodes[b], scene, cameras))
            seen_scenes.add(scene)
            total += 1
            if total == per_site:
                break
    choices = {r["episode_id"]: r for a, b, _, _ in accepted for r in (a, b)}
    intake = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {key: pool.submit(acquire, row, work) for key, row in choices.items()}
        for a, b, scene, cameras in accepted:
            ids = [a["episode_id"], b["episode_id"]]
            try:
                first, second = (jobs[episode].result() for episode in ids)
                if first["scene_id"] != scene[1] or second["scene_id"] != scene[1]:
                    raise ValueError("Acquired scene differs from screened metadata")
                pair_id = "droid_span_" + digest("|".join(ids).encode())[:10]
                row = dict(pair_id=pair_id, site=scene[0], scene_id=scene[1], matched_cameras=cameras,
                           open_episode=first, close_episode=second)
                row["grids"] = review_pair(work, row)
                intake.append(row)
                print("DROID_SPAN_REVIEW_READY", pair_id, scene, flush=True)
            except SOURCE_ERRORS as exc:
                unavailable.append({"pair": ids, "error": type(exc).__name__})
    return dict(intake=intake, unavailable=unavailable, screened_candidate_pairs=len(combos),
                excluded_prior_episodes=len(seen_episodes), excluded_prior_scenes=prior_scene_count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--per-site", type=int, default=4)
    args = parser.parse_args()
    if args.per_site < 1:
        parser.error("--per-site must be positive")
    result = prepare(args.work, args.per_site)
    (args.work / "intake.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["intake"]:
        raise RuntimeError("No fresh matched video candidates acquired. See intake.json.")


if __name__ == "__main__":
    main()
