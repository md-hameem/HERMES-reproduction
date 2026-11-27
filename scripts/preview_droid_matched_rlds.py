"""Recover bounded original RLDS records for matched-event candidate review."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import re
import struct

import requests

from prepare_droid_matched_events import WORK, nominate, prior_ids
from validate_twinpast_diagnostic_evidence import digest

MAP_URL = ("https://huggingface.co/datasets/EpicPinkPenguin/droid_dataset_segmentation_mask/"
           "resolve/main/lerobot_to_rlds_episode_mapping.parquet")
MAP_SHA = "7548f287fb4f2d62c642e4e89a8b7a5ac50d278d18b1d2ff0b042e4c7309eb5a"
BUCKET = "https://storage.googleapis.com/gresearch/robotics/droid/1.0.1/"
FIELDS = ("rlds_tfrecord_path", "rlds_byte_offset", "rlds_recording_folderpath", "rlds_file_path", "rlds_length")


def byte_range(url, start, length):
    end = start + length - 1
    with requests.get(url, params={"preview_range": f"{start}-{end}"},
                      headers={"Range": f"bytes={start}-{end}"}, timeout=(8, 30), stream=True) as response:
        response.raise_for_status()
        if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
            raise ValueError("Server returned an unexpected range")
        data = response.raw.read(length + 1)
    if len(data) != length:
        raise ValueError("Incomplete TFRecord range")
    return data


def review_record(work, row, entry, example_type, max_bytes):
    from PIL import Image, ImageDraw

    shard = entry["rlds_tfrecord_path"]
    if not re.fullmatch(r"droid_101-train\.tfrecord-\d{5}-of-02048", shard):
        raise ValueError("Unexpected shard path")
    offset = int(entry["rlds_byte_offset"])
    url = BUCKET + shard
    size, _ = struct.unpack("<QI", byte_range(url, offset, 12))
    if not 0 < size <= max_bytes:
        raise ValueError(f"Record has {size} bytes, above preview limit {max_bytes}")
    record = byte_range(url, offset + 12, size)
    features = example_type.FromString(record).features.feature
    for key in ("file_path", "recording_folderpath"):
        values = features[f"episode_metadata/{key}"].bytes_list.value
        if list(values) != [entry[f"rlds_{key}"].encode()]:
            raise ValueError(f"RLDS identity differs for {key}")
    dest = work / row["episode_id"]
    dest.mkdir(exist_ok=True)
    length = int(entry["rlds_length"])
    if length < 10:
        raise ValueError("Episode has fewer than ten steps")
    slots = [i*(length-1)//9 for i in range(10)]
    for camera in (1, 2):
        images = features[f"steps/observation/exterior_image_{camera}_left"].bytes_list.value
        if len(images) != length:
            raise ValueError(f"Unexpected RLDS camera length {len(images)} != {length}")
        sheet = Image.new("RGB", (1250, 325), "white")
        draw = ImageDraw.Draw(sheet)
        for slot, frame in enumerate(slots):
            with Image.open(BytesIO(images[frame])) as im:
                rgb = im.convert("RGB")
                rgb.thumbnail((248, 139))
            x, y = slot % 5 * 250, slot // 5 * 159
            sheet.paste(rgb, (x, y))
            draw.text((x + 4, y + 141), f"slot {slot}  frame {frame}", fill="black")
        sheet.save(dest / f"rlds_ext{camera}.jpg", quality=90)
    return dict(url=url, byte_offset=offset, bytes=size, sha256=digest(record), sampled_indices=slots,
                source_recording=entry["rlds_recording_folderpath"])


def main(work, limit, max_mib):
    import pyarrow.parquet as pq
    from tensorflow.core.example.example_pb2 import Example

    path = work / "lerobot_to_rlds_episode_mapping.parquet"
    if not path.exists():
        response = requests.get(MAP_URL, timeout=(10, 45))
        response.raise_for_status()
        if digest(response.content) != MAP_SHA:
            raise ValueError("Mapping bytes differ from pinned digest")
        path.write_bytes(response.content)
    if digest(path.read_bytes()) != MAP_SHA:
        raise ValueError("Cached mapping bytes differ from pinned digest")
    entries = {}
    for row in pq.read_table(path, columns=list(FIELDS)).to_pylist():
        key = "/".join(row["rlds_recording_folderpath"].split("/")[-6:])
        entries.setdefault(key, []).append(row)
    ranked = nominate(prior_ids()[0], 60, True)[:limit]
    episodes = {r["episode_id"]: r for _, _, _, a, b in ranked for r in (a, b)}
    print(f"DROID_RLDS_PREVIEW pairs={len(ranked)} episodes={len(episodes)}", flush=True)
    output_path = work / "rlds_intake.json"
    previous = json.loads(output_path.read_bytes()) if output_path.is_file() else {}
    cached = previous.get("receipts", {}) if previous.get("mapping_sha256") == MAP_SHA else {}
    receipts, failures = {}, []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for uid, row in episodes.items():
            key = "/".join(row["prefix"].rstrip("/").split("/")[-4:]) + "/recordings/MP4"
            mapped = entries.get(key, [])
            if len(mapped) != 1:
                failures.append(dict(episode=uid, error="ambiguous_or_missing_mapping"))
                continue
            prior = cached.get(uid)
            entry = mapped[0]
            if (prior and prior["byte_offset"] == entry["rlds_byte_offset"]
                    and prior["source_recording"] == entry["rlds_recording_folderpath"]
                    and all((work / uid / f"rlds_ext{i}.jpg").is_file() for i in (1, 2))):
                receipts[uid] = prior
                continue
            futures[uid] = pool.submit(review_record, work, row, entry, Example, max_mib * 1024**2)
        for uid, future in futures.items():
            try:
                receipts[uid] = future.result()
                print("DROID_RLDS_PREVIEW_READY", uid, flush=True)
            except (requests.RequestException, OSError, ValueError, KeyError) as exc:
                failures.append(dict(episode=uid, error=type(exc).__name__, detail=str(exc)))
    pairs = [dict(nomination=key, episodes=[a, b]) for _, _, key, a, b in ranked]
    output = dict(mapping_url=MAP_URL, mapping_sha256=MAP_SHA, pairs=pairs, receipts=receipts, failures=failures)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--limit", type=int, default=14)
    parser.add_argument("--max-mib", type=int, default=160)
    args = parser.parse_args()
    if min(args.limit, args.max_mib) < 1:
        parser.error("--limit and --max-mib must be positive")
    args.work.mkdir(parents=True, exist_ok=True)
    main(args.work, args.limit, args.max_mib)
