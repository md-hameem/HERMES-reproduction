"""Acquire and preview source-verifiable DROID episodes before any model evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from urllib.parse import quote

import requests

from validate_twinpast_diagnostic_evidence import ROOT, digest

PREFIX = "robotics/droid_raw/1.0.1/"
BUCKET = "https://storage.googleapis.com/gresearch/"
API = "https://storage.googleapis.com/storage/v1/b/gresearch/o"
WORK = ROOT / "evidence/local/droid-state-intake"
INDEX = {
    "manifest.json": "f540f0a86e791d11588670f82e8045b54ae79959a8104aa6781087486f65a5d2",
    "aggregated-annotations-030724.json": "42ed672ee974bfcb15752dc0768dd66a47e01ea3eebfdb74d8d5e468b37b2b6d",
}
ACTION = re.compile(r"^(open|close) (?:the |a )?(drawer|cabinet|door)\b", re.I)


def get(url, **kwargs):
    response = requests.get(url, timeout=(10, 45), **kwargs)
    response.raise_for_status()
    return response


def candidates(work, per_site):
    for name, sha in INDEX.items():
        path = work / name
        if not path.exists():
            path.write_bytes(get(BUCKET + PREFIX + name).content)
        if digest(path.read_bytes()) != sha:
            raise ValueError(f"DROID index digest differs: {name}")
    annotations = json.loads((work / "aggregated-annotations-030724.json").read_bytes())
    manifest = json.loads((work / "manifest.json").read_bytes())
    paths = {}
    for row in manifest:
        tail = row["path"].split("r2d2_full_raw/", 1)[-1].split("/")
        match = re.search(r"(\d{1,2}):(\d{2}):(\d{2})", tail[-2])
        if match and len(tail) == 5 and tail[1] == "success":
            h, m, s = match.groups()
            paths.setdefault((tail[0], f"{tail[2]}-{int(h):02}h-{m}m-{s}s"), []).append("/".join(tail[:-1]) + "/")
    grouped = {}
    for uuid, row in annotations.items():
        lab, _, start = uuid.split("+")
        instruction = row.get("language_instruction1", "").strip()
        match = ACTION.match(instruction)
        found = paths.get((lab, start), [])
        if match and len(found) == 1:
            action, target = map(str.lower, match.groups())
            grouped.setdefault(lab, []).append(dict(episode_id=uuid, site=lab, action=action, target=target,
                                                     instruction=instruction, prefix=PREFIX+found[0]))
    for lab, episodes in sorted(grouped.items()):
        # Prefer simple target descriptions, then use a deterministic order.
        ranked = sorted(episodes, key=lambda row: (len(row["instruction"]), digest(row["episode_id"].encode())))
        yield from ranked[:per_site]


def download(object_spec, dest):
    url = BUCKET + quote(object_spec["name"], safe="/")
    size = int(object_spec["size"])
    if size > 35*1024**2:
        raise ValueError(f"Source object too large for this intake: {size}")
    marker = dest.with_suffix(dest.suffix + ".generation")
    if not (dest.is_file() and dest.stat().st_size == size and marker.is_file()
            and marker.read_text() == object_spec["generation"]):
        temp = dest.with_suffix(dest.suffix + ".part")
        with get(url, params={"generation": object_spec["generation"]}, stream=True) as response, temp.open("wb") as output:
            count = 0
            for chunk in response.iter_content(256*1024):
                count += len(chunk)
                if count > size:
                    raise ValueError("DROID object exceeds listed size")
                output.write(chunk)
        if count != size:
            raise ValueError("DROID object is incomplete")
        temp.replace(dest)
        marker.write_text(object_spec["generation"])
    return dict(url=url, generation=object_spec["generation"], bytes=size,
                sha256=digest(dest.read_bytes()))


def preview(path, out):
    from decord import VideoReader
    from PIL import Image, ImageDraw

    reader = VideoReader(str(path), num_threads=1)
    indices = [int((len(reader)-1)*f) for f in (0, .08, .18, .36, .63, .80, .90, .99)]
    frames = reader.get_batch(indices).asnumpy()
    sheet = Image.new("RGB", (960, 328), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (index, frame) in enumerate(zip(indices, frames, strict=True)):
        image = Image.fromarray(frame)
        image.thumbnail((238, 140))
        x, y = (i % 4)*240, (i//4)*162
        sheet.paste(image, (x, y))
        draw.text((x+4, y+140), str(index), fill="black")
    sheet.save(out, quality=89)
    return len(reader)


def acquire(row, work):
    directory = work / row["episode_id"]
    directory.mkdir(exist_ok=True)
    objects = get(API, params={"prefix": row["prefix"], "maxResults": 40}).json().get("items", [])
    stored = {obj["name"]: obj for obj in objects}
    meta_object = row["prefix"] + f"metadata_{row['episode_id']}.json"
    meta_path = directory / "metadata.json"
    metadata_source = download(stored[meta_object], meta_path)
    metadata = json.loads(meta_path.read_bytes())
    if metadata["uuid"] != row["episode_id"] or metadata["lab"] != row["site"] or not metadata["success"]:
        raise ValueError(f"Source identity differs: {row['episode_id']}")
    cameras = {}
    for side in ("ext1", "ext2"):
        spec = stored[PREFIX + row["site"] + "/" + metadata[f"{side}_mp4_path"]]
        path = directory / f"{side}.mp4"
        cameras[side] = download(spec, path)
        cameras[side]["frame_count"] = preview(path, directory / f"{side}.jpg")
    return {**row, "scene_id": metadata["scene_id"], "metadata_source": metadata_source, "cameras": cameras}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--per-site", type=int, default=2)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    chosen = list(candidates(args.work, args.per_site))
    rows = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        jobs = [(row, executor.submit(acquire, row, args.work)) for row in chosen]
        for row, job in jobs:
            try:
                rows.append(job.result())
                print("DROID_PREVIEW_READY", row["episode_id"], flush=True)
            except (requests.RequestException, KeyError, ValueError, RuntimeError) as exc:
                print("DROID_PREVIEW_FAILED", row["episode_id"], type(exc).__name__, flush=True)
    (args.work / "intake.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8", newline="\n")
    if not rows:
        raise RuntimeError("No DROID episode previews were acquired")


if __name__ == "__main__":
    main()
