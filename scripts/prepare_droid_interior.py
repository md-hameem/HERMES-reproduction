"""Acquire fresh DROID candidates for a CPU-only interior-evidence review."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re

from prepare_droid_natural_history import metadata
from prepare_droid_span_cohort import SOURCE_ERRORS, WORK as SPAN, exclusions
from prepare_droid_state_control import INDEX, PREFIX, WORK as CONTROL, acquire
from validate_twinpast_diagnostic_evidence import ROOT, digest

WORK = ROOT / "evidence/local/droid-interior"
ACTION = re.compile(r"\b(open|close|shut)\b", re.I)
TARGET = re.compile(r"\b(drawer|cabinet|cupboard|door|lid|oven|microwave|kettle|toaster|fryer|flask)\b", re.I)
ANCHOR = re.compile(r"\b(put|place|take|remove|pick|retrieve|insert|move|drop)\b", re.I)


def source_candidates():
    payloads = {}
    for name, expected in INDEX.items():
        raw = (CONTROL / name).read_bytes()
        if digest(raw) != expected:
            raise ValueError(f"Unexpected DROID index bytes for {name}")
        payloads[name] = json.loads(raw)
    paths = defaultdict(list)
    for row in payloads["manifest.json"]:
        parts = row["path"].split("r2d2_full_raw/", 1)[-1].split("/")
        if len(parts) == 5 and parts[1] == "success":
            stamp = re.search(r"(\d{1,2}):(\d{2}):(\d{2})", parts[-2])
            if stamp:
                h, m, s = stamp.groups()
                key = (parts[0], f"{parts[2]}-{int(h):02}h-{m}m-{s}s")
                paths[key].append(PREFIX + "/".join(parts[:-1]) + "/")

    for uid, annotation in payloads["aggregated-annotations-030724.json"].items():
        site, _, stamp = uid.split("+")
        prefixes = paths[(site, stamp)]
        if len(prefixes) != 1:
            continue
        for instruction in filter(None, annotation.values()):
            text = instruction.strip()
            if not TARGET.search(text):
                continue
            actions = list(ACTION.finditer(text))
            # Ignore adjectival mentions such as "close the open drawer".
            if not re.match(r"^(?:fully |completely )?(?:open|close|shut)\b", text, re.I):
                continue
            matches = [m for m in actions[1:] if m.group().lower() != actions[0].group().lower()
                       and not re.search(r"\b(?:the|a|an)\s+$", text[:m.start()], re.I)]
            if not matches:
                continue
            first, second = actions[0], matches[0]
            between = text[first.end():second.start()]
            anchor = bool(ANCHOR.search(between))
            if not anchor and not re.search(r"\b(and|then|before|after|again)\b", between, re.I):
                continue
            yield dict(episode_id=uid, site=site, prefix=prefixes[0], instruction=text,
                       between_action=anchor, starts_open=first.group().lower() != "open")
            break


def prior_sources():
    episodes, scenes = exclusions()
    for path in SPAN.glob("*/metadata.json"):
        row = json.loads(path.read_bytes())
        episodes.add(row["uuid"])
        scenes.add((row["lab"], row["scene_id"]))
    frozen = json.loads((ROOT / "data/twinpast/droid_span_cohort_v1.json").read_bytes())
    episodes.update(b["source_episode"] for c in frozen["cases"] for b in c["branches"].values())
    scenes.update((c["source_site"], c["scene_id"]) for c in frozen["cases"])
    return episodes, scenes


def render(work, episode):
    from decord import VideoReader
    from PIL import Image, ImageDraw

    uid = episode["episode_id"]
    grids = {}
    for camera in ("ext1", "ext2"):
        reader = VideoReader(str(work / uid / f"{camera}.mp4"), num_threads=1)
        if len(reader) < 10:
            raise ValueError("Fewer than ten source frames")
        slots = [i*(len(reader)-1)//9 for i in range(10)]
        frames = reader.get_batch(slots).asnumpy()
        view = Image.new("RGB", (1250, 325), "white")
        draw = ImageDraw.Draw(view)
        for slot, (frame, index) in enumerate(zip(frames, slots, strict=True)):
            image = Image.fromarray(frame)
            image.thumbnail((248, 139))
            x, y = slot % 5 * 250, slot // 5 * 159
            view.paste(image, (x, y))
            draw.text((x + 4, y + 141), f"slot {slot}  frame {index}", fill="black")
        view.save(work / uid / f"grid_{camera}.jpg", quality=89)
        grids[camera] = slots
    return grids


def prepare(work, per_site):
    work.mkdir(parents=True, exist_ok=True)
    seen_episodes, seen_scenes = prior_sources()
    old_scenes = seen_scenes.copy()
    pools = defaultdict(list)
    for row in source_candidates():
        if row["episode_id"] not in seen_episodes:
            pools[row["site"]].append(row)
    shortlist = [r for site in sorted(pools) for r in sorted(
        pools[site], key=lambda r: (not r["starts_open"], not r["between_action"], digest(r["episode_id"].encode()))
    )[:max(12, 4 * per_site)]]
    screened, failed = [], []
    print(f"DROID_INTERIOR_SCREEN metadata={len(shortlist)} prior_scenes={len(old_scenes)}", flush=True)
    cache = SPAN / "metadata"

    def lookup(row):
        cached = cache / (row["episode_id"] + ".json")
        return json.loads(cached.read_bytes()) if cached.is_file() else metadata(row)

    taken = defaultdict(int)
    with ThreadPoolExecutor(max_workers=14) as pool:
        jobs = [pool.submit(lookup, row) for row in shortlist]
        for row, job in zip(shortlist, jobs, strict=True):
            site = row["site"]
            try:
                data = job.result()
                scene = (site, data["scene_id"])
                if data["success"] and scene not in seen_scenes and taken[site] < per_site:
                    screened.append((row, scene))
                    seen_scenes.add(scene)
                    taken[site] += 1
            except SOURCE_ERRORS as exc:
                failed.append(dict(episode=row["episode_id"], stage="metadata", error=type(exc).__name__))

    intake = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = [pool.submit(acquire, row, work) for row, _ in screened]
        for (row, scene), job in zip(screened, jobs, strict=True):
            try:
                episode = job.result()
                if episode["scene_id"] != scene[1]:
                    raise ValueError("Scene differs from screened metadata")
                episode["grids"] = render(work, episode)
                intake.append(episode)
                print("DROID_INTERIOR_REVIEW_READY", episode["episode_id"], episode["scene_id"], flush=True)
            except SOURCE_ERRORS as exc:
                failed.append(dict(episode=row["episode_id"], stage="acquire", error=type(exc).__name__))
    result = dict(index_candidates=sum(map(len, pools.values())), metadata_requests=len(shortlist),
                  excluded_episodes=sorted(seen_episodes), excluded_scenes=sorted(old_scenes),
                  unavailable=failed, intake=intake)
    (work / "intake.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not intake:
        raise RuntimeError("No new DROID clips were acquired. See intake.json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--per-site", type=int, default=3)
    args = parser.parse_args()
    if args.per_site < 1:
        parser.error("--per-site must be positive")
    prepare(args.work, args.per_site)
