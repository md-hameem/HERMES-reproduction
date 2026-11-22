"""Check causal sampling and shared recent frames for the frozen TwinPast sample."""
import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from decord import VideoReader
from PIL import Image, ImageDraw

from prepare_twinpast_development import ANNOTATION_SHA, BASE_URL, OUT, ROOT, freeze

CACHE = ROOT / "evidence/local/twinpast-development"


def download(relative, dest):
    if dest.exists():
        return
    temp = dest.with_suffix(".part")
    with requests.get(f"{BASE_URL}/{relative}", stream=True, timeout=(10, 40)) as response:
        response.raise_for_status()
        size = 0
        with temp.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                size += len(chunk)
                if size > 128 * 1024**2:
                    raise ValueError(f"Unexpected clip size: {relative}")
                handle.write(chunk)
    os.replace(temp, dest)


def check_pair(plan, annotation, work):
    pair = plan["pair_id"]
    work = work / pair
    work.mkdir(exist_ok=True)
    recent = plan["latest_four_grid_s"]
    earlier = [t for t in plan["common_earlier_grid_s"] if t < recent[0]][:2]
    if len(earlier) < 2:
        return dict(pair_id=pair, task=plan["task"], status="insufficient_disjoint_earlier_slots",
                    earlier_grid_s=earlier, recent_grid_s=recent)
    slots = earlier + recent
    assert slots == sorted(set(slots)) and slots[-1] < plan["query_time_s"]
    small, previews, decoded = {}, {}, {}
    for branch in ("a", "b"):
        dest = work / f"{branch.upper()}.mp4"
        download(plan[f"video_{branch}"], dest)
        reader = VideoReader(str(dest), num_threads=1)
        timestamps = np.asarray(reader.get_frame_timestamp(list(range(len(reader)))), dtype=np.float64)
        timestamps -= timestamps[0, 0]
        indices = np.searchsorted(timestamps[:, 0], slots, side="right") - 1
        if indices[0] < 0 or np.any(np.diff(indices) <= 0) or slots[-1] > timestamps[-1, 1] + 1e-6:
            raise ValueError(f"Video cannot represent requested sample grid: {pair}/{branch}")
        seconds = timestamps[indices, 0].tolist()
        intervals = annotation["evidence_intervals"][branch.upper()]
        inside = [any(start - 1e-6 <= t <= stop + 1e-6 for start, stop in intervals) for t in seconds[:2]]
        images = [Image.fromarray(rgb) for rgb in reader.get_batch(indices.tolist()).asnumpy()]
        frame_hashes = [hashlib.sha256(im.tobytes()).hexdigest() for im in images]
        small[branch] = [np.asarray(im.resize((192, 108)), dtype=np.int16) for im in images]
        for im in images:
            im.thumbnail((320, 180))
        previews[branch] = images
        with dest.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        decoded[branch] = dict(bytes=dest.stat().st_size, sha256=digest, source_frames=indices.tolist(),
                               frame_start_s=seconds, early_samples_inside=inside, decoded_rgb_sha256=frame_hashes)
    canvas = Image.new("RGB", (320 * len(slots), 410), "white")
    draw = ImageDraw.Draw(canvas)
    for row, branch in enumerate(("a", "b")):
        for i, im in enumerate(previews[branch]):
            x, y = i * 320, row * 205
            canvas.paste(im, (x, y))
            draw.text((x + 4, y + 183), f"{branch.upper()} | {slots[i]} s", fill="black")
    canvas.save(work / "review.jpg", quality=91)
    mae = [float(np.abs(a - b).mean() / 255) for a, b in zip(small["a"], small["b"], strict=True)]
    print(f"{pair} recent_matches={sum(value == 0 for value in mae[2:])}/4 "
          f"inside={all(all(x['early_samples_inside']) for x in decoded.values())}", flush=True)
    return dict(pair_id=pair, task=plan["task"], status="checked",
                grid_s=slots, normalized_rgb_difference=mae, decoded=decoded)


def main():
    response = requests.get(f"{BASE_URL}/data.jsonl", timeout=(10, 25))
    response.raise_for_status()
    assert hashlib.sha256(response.content).hexdigest() == ANNOTATION_SHA
    plan = json.loads(OUT.read_bytes())
    assert freeze(response.content) == plan
    annotations = {r["pair_id"]: r for r in (json.loads(line) for line in response.content.splitlines() if line.strip())}
    CACHE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(check_pair, row, annotations[row["pair_id"]], CACHE) for row in plan["selected_pairs"]]
        results = [future.result() for future in futures]
    (CACHE / "media_check.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    with (CACHE / "visual_questions.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("pair_id", "question", "options"))
        for row in plan["selected_pairs"]:
            item = annotations[row["pair_id"]]
            writer.writerow((row["pair_id"], item["question"], json.dumps(item["options"], ensure_ascii=False)))


if __name__ == "__main__":
    main()
