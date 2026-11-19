"""Resumable AutoDL scorer screen. Bootstrap is embedded in the companion notebook.

Only experimental instances are patched. Official inference files remain intact.
Heavy imports are deferred so selection, checkpoint and download logic can be
verified on a CPU host without installing the GPU environment.
"""

import argparse
import ast
import copy
import csv
import hashlib
import html
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import types
from pathlib import Path
from zipfile import ZipFile


MODEL_ID = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
MODEL_SHA = "74dd0bf867a4cda7950c17663794267c60cf4b40"
SHARD_NAME = "Real-Time Visual Understanding_201-250.zip"
SHARD_BYTES = 2230277873
SHARD_SHA = "b7946e8b678fbf704b9816fa708e6febb77f5c37a7ad884084adf858f5d20dbf"
METHODS = ("fifo", "hermes", "causal")
FPS = 0.5
SEED = 2024


def stable(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(temp, path)


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot publish an empty result table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_id(video):
    return int(re.search(r"sample_(\d+)", video["video_path"])[1])


def choose_screen(repo, per_outcome=2):
    from build_failure_pilot import select_cases
    rows = read_csv(repo / "results/failure_pilot/paired_inventory.csv")
    pool = []
    for row in rows:
        if 201 <= int(row["sample_id"]) <= 250:
            pool.append({**row, "duplicate_gold_options": row["duplicate_gold_options"] == "True"})
    quotas = {k: per_outcome for k in ("hermes_only", "fifo_only", "both_wrong", "both_correct")}
    selected = select_cases(pool, seed=SEED, quotas=quotas)
    annotations = json.loads((repo / "data/streamingbench/streamingbench_realtime.json").read_text(encoding="utf-8"))
    ids = {r["video_id"] for r in selected}
    by_id = {str(v["video_id"]): v for v in annotations if str(v["video_id"]) in ids}
    # Calibrate on one HERMES-only and one FIFO-only target before other cases.
    ordered = [selected[group * per_outcome + index]["video_id"]
               for index in range(per_outcome) for group in range(4)]
    videos = [by_id[vid] for vid in ordered]
    if len(videos) != per_outcome * 4:
        raise ValueError("Screen selection and annotation IDs differ")
    return selected, videos


class RangeUnsupported(RuntimeError):
    pass


class HTTPRangeFile(io.RawIOBase):
    """Seekable bounded HTTP reader for ZIP/ZIP64 central directories and members.

    A server returning 200 to a Range request is rejected BEFORE its body is read.
    ZipFile verifies each extracted member's CRC. Whole-archive SHA is only
    verified when the full-download fallback is used.
    """
    def __init__(self, url, size, session, block_size=8 * 1024**2):
        self.url, self.size, self.session = url, size, session
        self.pos, self.block_size = 0, block_size
        self.buffer_start, self.buffer = -1, b""
        self.bytes_received = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if pos < 0:
            raise ValueError("Negative seek")
        self.pos = pos
        return pos

    def read(self, count=-1):
        count = max(0, self.size - self.pos) if count is None or count < 0 else min(count, max(0, self.size - self.pos))
        chunks = []
        while count:
            if not self.buffer_start <= self.pos < self.buffer_start + len(self.buffer):
                start = self.pos
                end = min(self.size - 1, start + max(self.block_size, min(count, self.block_size)) - 1)
                for attempt in range(3):
                    try:
                        with self.session.get(self.url, headers={"Range": f"bytes={start}-{end}",
                                                               "Accept-Encoding": "identity"},
                                              stream=True, timeout=(30, 120)) as response:
                            expected = f"bytes {start}-{end}/{self.size}"
                            if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                                raise RangeUnsupported(f"Range unavailable or size mismatch, HTTP {response.status_code}")
                            data = response.content
                            if len(data) != end - start + 1:
                                raise OSError("Truncated HTTP range")
                            self.bytes_received += len(data)
                            self.buffer_start, self.buffer = start, data
                        break
                    except RangeUnsupported:
                        raise
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(2 * (attempt + 1))
            offset = self.pos - self.buffer_start
            take = min(count, len(self.buffer) - offset)
            chunks.append(self.buffer[offset:offset + take])
            self.pos += take
            count -= take
        return b"".join(chunks)


def extract_members(archive, videos, directory):
    directory.mkdir(parents=True, exist_ok=True)
    members = {}
    for item in archive.infolist():
        if item.is_dir() or Path(item.filename).name.startswith("._"):
            continue
        match = re.search(r"sample_(\d+)", item.filename)
        if match and item.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            sid = int(match[1])
            if sid in members:
                raise ValueError(f"Ambiguous source members for sample {sid}")
            members[sid] = item
    metadata = []
    for video in videos:
        sid = sample_id(video)
        item = members[sid]
        # The archive name is never used as an extraction path.
        dest = directory / f"sample_{sid}_real{Path(item.filename).suffix.lower()}"
        temp = dest.with_suffix(dest.suffix + ".tmp")
        with archive.open(item) as source, temp.open("wb") as target:
            while True:
                block = source.read(4 * 1024**2)
                if not block:
                    break
                target.write(block)
        if temp.stat().st_size != item.file_size:
            raise ValueError(f"Truncated extracted video {sid}")
        os.replace(temp, dest)
        metadata.append(dict(sample_id=sid, path=str(dest.resolve()), bytes=item.file_size,
                             member=item.filename, zip_crc32=item.CRC, sha256=digest(dest)))
        print(f"VIDEO_READY sample={sid} MiB={item.file_size/1024**2:.1f}", flush=True)
    return metadata


def download_full(urls, path):
    import requests
    if path.is_file() and path.stat().st_size == SHARD_BYTES and digest(path) == SHARD_SHA:
        return
    for url in urls:
        for attempt in range(3):
            try:
                start = path.stat().st_size if path.exists() else 0
                if start >= SHARD_BYTES:
                    path.unlink()
                    start = 0
                with requests.get(url, headers={"Range": f"bytes={start}-", "Accept-Encoding": "identity"},
                                  stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    if response.status_code == 206:
                        if not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                            raise ValueError("Wrong resume offset")
                        mode = "ab"
                    else:
                        mode = "wb"
                    last = time.monotonic()
                    with path.open(mode) as handle:
                        for block in response.iter_content(4 * 1024**2):
                            handle.write(block)
                            if time.monotonic() - last > 30:
                                print(f"Archive {path.stat().st_size/1024**3:.2f}/2.08 GiB", flush=True)
                                last = time.monotonic()
                if path.stat().st_size == SHARD_BYTES and digest(path) == SHARD_SHA:
                    return
                raise ValueError("Full archive size/hash mismatch")
            except Exception as exc:
                print(f"Download retry {attempt+1}: {type(exc).__name__}", flush=True)
                time.sleep(3 * (attempt + 1))
    raise RuntimeError("Dataset download failed on mirror and official endpoint; rerun to resume")


def prepare_assets(videos, work):
    import requests
    directory = work / "videos"
    marker = work / "asset_manifest.json"
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if {x["sample_id"] for x in saved["videos"]} == {sample_id(v) for v in videos}:
            if all(Path(x["path"]).is_file() and Path(x["path"]).stat().st_size == x["bytes"]
                   and digest(x["path"]) == x["sha256"] for x in saved["videos"]):
                print("Reusing SHA-verified selected videos", flush=True)
                return saved
    from urllib.parse import quote
    urls = [f"https://{host}/datasets/mjuicem/StreamingBench/resolve/main/{quote(SHARD_NAME)}?download=true"
            for host in ("hf-mirror.com", "huggingface.co")]
    for url in urls:
        try:
            with requests.Session() as session:
                remote = HTTPRangeFile(url, SHARD_BYTES, session)
                with ZipFile(remote) as archive:
                    metadata = extract_members(archive, videos, directory)
            saved = dict(mode="range_selected_members", videos=metadata, transferred_bytes=remote.bytes_received,
                         archive_expected_sha256=SHARD_SHA, whole_archive_sha256_verified=False,
                         integrity="ZIP member CRC verified; observed video SHA256 saved for resumption",
                         source=url)
            atomic_json(marker, saved)
            return saved
        except Exception as exc:
            print(f"Selective ZIP unavailable ({type(exc).__name__}); trying next endpoint", flush=True)
    print("Falling back to the single 2.08 GiB archive; extracting only selected videos", flush=True)
    archive_path = work / "downloads" / SHARD_NAME
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    download_full(urls, archive_path)
    with ZipFile(archive_path) as archive:
        metadata = extract_members(archive, videos, directory)
    saved = dict(mode="full_archive", videos=metadata, archive_expected_sha256=SHARD_SHA,
                 whole_archive_sha256_verified=True, source=urls)
    atomic_json(marker, saved)
    return saved


def chunk_plan(video, sampled_frames):
    cursor, chunks = 0, []
    for i, qa in enumerate(video["conversations"]):
        end = math.ceil(float(qa["end_time"]) * FPS)
        if end < cursor or end > sampled_frames:
            raise ValueError(f"Invalid prefix bound for sample {sample_id(video)}, question {i}")
        while cursor < end:
            stop = min(cursor + 16, end)
            chunks.append(dict(qa_index=i, start=cursor, stop=stop))
            cursor = stop
    return chunks


def causal_scores(model, input_ids, past_key_values=None):
    """Fresh layerwise causal propagation through actual Qwen2 decoder layers.

    The temporary DynamicCache has separate lists; Qwen2 appends via torch.cat
    and never writes original K/V tensors. The source cache remains unchanged.
    Explicit masks allow different physical cache lengths per layer.
    """
    import torch
    from transformers import DynamicCache
    past = model.kv_cache if past_key_values is None else past_key_values
    cache = DynamicCache.from_legacy_cache(tuple((k, v) for k, v in past))
    hidden = model.get_input_embeddings()(input_ids)
    batch, q_len = input_ids.shape
    starts = model._get_next_start_pos_per_layer()
    model._set_rotary_required_len(max(starts) + q_len)
    original_hooks = dict(model._layer_position_ids)
    weights = []
    try:
        for i, layer in enumerate(model.language_model.model.layers):
            pos = model._build_position_ids(starts[i], q_len, batch)
            model._layer_position_ids[i] = pos
            past_len = past[i][0].shape[2]
            mask = torch.zeros((batch, 1, q_len, past_len + q_len), device=hidden.device, dtype=hidden.dtype)
            future = torch.ones((q_len, q_len), device=hidden.device, dtype=torch.bool).triu(1)
            mask[:, :, :, past_len:].masked_fill_(future, torch.finfo(hidden.dtype).min)
            result = layer(hidden, attention_mask=mask, position_ids=pos, past_key_value=cache,
                           output_attentions=True, use_cache=False)
            hidden = result[0]
            weights.append(result[1])
            if result[1] is None or result[1].shape[-1] != past_len + q_len:
                raise ValueError("Native decoder did not return expected attention")
    finally:
        model._layer_position_ids.clear()
        model._layer_position_ids.update(original_hooks)
    return weights


def reference_scores(model, input_ids):
    import torch
    from transformers import DynamicCache
    starts = model._get_next_start_pos_per_layer()
    q_len, batch = input_ids.shape[1], input_ids.shape[0]
    previous = dict(model._layer_position_ids)
    try:
        for i in range(model.num_layers):
            model._layer_position_ids[i] = model._build_position_ids(starts[i], q_len, batch)
        model._set_rotary_required_len(max(starts) + q_len)
        cache = DynamicCache.from_legacy_cache(tuple((k.clone(), v.clone()) for k, v in model.kv_cache))
        output = model.language_model.model(input_ids=input_ids, past_key_values=cache,
                                            position_ids=model._layer_position_ids[0],
                                            use_cache=False, output_attentions=True,
                                            attention_mask=None, return_dict=True)
        return list(output.attentions)
    finally:
        model._layer_position_ids.clear()
        model._layer_position_ids.update(previous)


def ingest_features(model, features, frames):
    import torch
    from inference.reindex_1d import contiguous_kv
    features = features.to(device=model.device, dtype=model.dtype)
    starts = model._get_next_start_pos_per_layer()
    batch, length = features.shape[:2]
    model._set_rotary_required_len(max(starts) + length)
    try:
        for i in range(model.num_layers):
            model._layer_position_ids[i] = model._build_position_ids(starts[i], length, batch)
        output = model.language_model(inputs_embeds=features, past_key_values=model.kv_cache,
                                      use_cache=True, return_dict=True,
                                      position_ids=model._layer_position_ids[0],
                                      cache_position=None, attention_mask=None)
        model.kv_cache = contiguous_kv(output.past_key_values)
        model._append_position_ids(starts, length)
        model.last_encoded_frames = frames
        model.total_processed_frames += frames
    finally:
        model._layer_position_ids.clear()


def fifo_compress(model):
    import torch
    from inference.reindex_1d import compute_cos_sin_for_positions, rotary_delta, apply_rotary_delta_to_keys_only
    lengths = model._get_cache_seq_len_per_layer()
    if max(lengths) <= model.kv_size:
        return
    threshold = model.language_model.config.max_position_embeddings - 1024
    compact = max(int(p[-1]) for p in model._position_ids_cache) > threshold
    new = []
    for i, (keys, values) in enumerate(model.kv_cache):
        keep = torch.cat((torch.arange(model.visual_start_idx, device=model.device),
                          torch.arange(max(model.visual_start_idx, lengths[i] - model.kv_size), lengths[i], device=model.device)))
        old = model._position_ids_cache[i][keep]
        pos = torch.arange(len(keep), device=model.device) if compact else old
        k = keys.index_select(2, keep)
        if compact:
            co, si = compute_cos_sin_for_positions(model.language_model, len(old), old, keys.dtype, model.device)
            cn, sn = compute_cos_sin_for_positions(model.language_model, len(pos), pos, keys.dtype, model.device)
            cd, sd = rotary_delta(co, si, cn, sn)
            k = apply_rotary_delta_to_keys_only(k, cd, sd)
        new.append((k.contiguous(), values.index_select(2, keep).contiguous()))
        model._position_ids_cache[i] = pos.clone()
    model.kv_cache = new
    model.token_activity_cache = [None] * model.num_layers


def reset(model):
    model.clear_cache()
    model._layer_position_ids.clear()
    model._position_ids_cache = [None] * model.num_layers
    model.token_activity_cache = [None] * model.num_layers
    model.total_processed_frames = 0
    model.encode_init_prompt()


def sync():
    import torch
    torch.cuda.synchronize()


def prepare_features(model, video, work, identity):
    import numpy as np
    import torch
    from decord import VideoReader
    from PIL import Image, ImageDraw
    directory = work / "features" / f"sample_{sample_id(video)}"
    marker = directory / "complete.json"
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved["identity"] == identity and all((directory / x["file"]).is_file()
                and digest(directory / x["file"]) == x["sha256"] for x in saved["chunks"]):
            print(f"FEATURE_CACHE reuse sample={sample_id(video)}", flush=True)
            return saved
    directory.mkdir(parents=True, exist_ok=True)
    reader = VideoReader(video["video_path"], num_threads=1)
    fps = round(reader.get_avg_fps())
    step = int(fps / FPS)
    if step < 1:
        raise ValueError("Invalid source fps")
    indices = list(range(0, len(reader), step))
    plan = chunk_plan(video, len(indices))
    timestamps = reader.get_frame_timestamp(indices).tolist()
    frame_rows = [dict(sampled_index=i, source_frame=frame, start_s=float(ts[0]), end_s=float(ts[1]))
                  for i, (frame, ts) in enumerate(zip(indices, timestamps))]
    write_csv(directory / "sampled_timestamps.csv", frame_rows)
    chunks, total_encoder_s = [], 0.0
    for index, item in enumerate(plan):
        sync()
        start = time.perf_counter()
        frames = torch.from_numpy(reader.get_batch(indices[item["start"]:item["stop"]]).asnumpy())
        pixels = model.processor.video_processor(frames.permute(0, 3, 1, 2), return_tensors="pt").pixel_values_videos
        features = model.get_video_features(pixels.to(model.device, model.dtype)).detach().cpu().contiguous()
        sync()
        total_encoder_s += time.perf_counter() - start
        dest = directory / f"chunk_{index:04d}.pt"
        temp = dest.with_suffix(".tmp")
        torch.save(features, temp)
        os.replace(temp, dest)
        chunks.append(dict(**item, file=dest.name, sha256=digest(dest), tokens=features.shape[1]))
    # Contact sheets are sparse review aids, not evidence annotations.
    review_dir = work / "review"
    review_dir.mkdir(exist_ok=True)
    for qi, qa in enumerate(video["conversations"]):
        end = math.ceil(float(qa["end_time"]) * FPS)
        chosen = sorted(set(np.linspace(0, end - 1, min(12, end), dtype=int).tolist()))
        thumbnails = reader.get_batch([indices[i] for i in chosen]).asnumpy()
        canvas = Image.new("RGB", (4 * 256, 3 * 164), "white")
        draw = ImageDraw.Draw(canvas)
        for n, (frame, si) in enumerate(zip(thumbnails, chosen)):
            im = Image.fromarray(frame)
            im.thumbnail((256, 142))
            x, y = (n % 4) * 256, (n // 4) * 164
            canvas.paste(im, (x, y))
            draw.text((x + 3, y + 144), f"sampled {si} | {timestamps[si][0]:.2f}s", fill="black")
        canvas.save(review_dir / f"sample_{sample_id(video)}_q{qi}.jpg", quality=80)
    saved = dict(identity=identity, sample_id=sample_id(video), source_video_sha256=digest(video["video_path"]),
                 source_fps_rounded=fps, sampling_step=step, chunks=chunks,
                 preparation_s=total_encoder_s, sampled_frames_available=len(indices),
                 note="Preparation includes chunk decode/preprocess/vision/transfer; excludes contact sheets and serialization")
    atomic_json(marker, saved)
    return saved


def checkpoint_valid(path, expected_hash, video, expected_method=None):
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("config_hash") != expected_hash or value.get("status") != "complete":
            return False
        if expected_method is not None and value.get("method") != expected_method:
            return False
        rows = value["predictions"]
        if len(rows) != len(video["conversations"]):
            return False
        for i, (row, qa) in enumerate(zip(rows, video["conversations"])):
            gold = "ABCDEFGH"[qa["choices"].index(qa["answer"])]
            if not (row["qa_index"] == i and row["video_id"] == str(video["video_id"])
                    and row["question"] == qa["question"] and row["choices"] == qa["choices"]
                    and row["gold_choice"] == gold and row["gold_answer"] == qa["answer"]
                    and row["correct"] == int(row["pred_choice"] == gold)):
                return False
        return sha_predictions(rows) == value["prediction_sha256"]
    except (OSError, ValueError, KeyError, TypeError):
        return False


def sha_predictions(rows):
    return hashlib.sha256(stable(rows).encode()).hexdigest()


def run_method(model, analyzer, video, method, feature_meta, work, config_hash):
    import torch
    output = work / "checkpoints" / method / f"sample_{sample_id(video)}.json"
    if checkpoint_valid(output, config_hash, video, method):
        print(f"REUSE method={method} sample={sample_id(video)}", flush=True)
        return json.loads(output.read_text(encoding="utf-8"))
    random.seed(SEED)
    torch.manual_seed(SEED)
    reset(model)
    original = model._compute_attention_scores_manually
    if method == "causal":
        model._compute_attention_scores_manually = types.MethodType(causal_scores, model)
    feature_dir = work / "features" / f"sample_{sample_id(video)}"
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    events, rows = [], []
    try:
        for i, qa in enumerate(video["conversations"]):
            for item in feature_meta["chunks"]:
                if item["qa_index"] != i:
                    continue
                features = torch.load(feature_dir / item["file"], map_location="cpu", weights_only=True)
                sync()
                before = time.perf_counter()
                ingest_features(model, features, item["stop"] - item["start"])
                sync()
                ingest_s = time.perf_counter() - before
                before = time.perf_counter()
                if method == "fifo":
                    fifo_compress(model)
                else:
                    model.predict_and_compress()
                sync()
                events.append(dict(qa_index=i, sampled_start=item["start"], sampled_stop=item["stop"],
                                   ingest_s=ingest_s, compression_s=time.perf_counter() - before,
                                   cache_lengths=model._get_cache_seq_len_per_layer()))
            gold = "ABCDEFGH"[qa["choices"].index(qa["answer"])]
            sync()
            before = time.perf_counter()
            answer = analyzer.video_close_qa(qa["question"], qa["choices"], gold)
            sync()
            rows.append(dict(video_id=str(video["video_id"]), sample_id=sample_id(video), qa_index=i,
                             query_time_s=qa["end_time"], task=qa["task"], question=qa["question"],
                             choices=qa["choices"], gold_answer=qa["answer"], gold_choice=gold,
                             pred_answer=answer["pred_answer"], pred_choice=answer["pred_choice"],
                             correct=int(answer["acc"]), answer_s=time.perf_counter() - before))
        sync()
        value = dict(status="complete", config_hash=config_hash, method=method,
                     predictions=rows, prediction_sha256=sha_predictions(rows), events=events,
                     replay_wall_s=time.perf_counter() - start,
                     peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                     peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)
        atomic_json(output, value)
        if not checkpoint_valid(output, config_hash, video, method):
            raise ValueError("Newly written checkpoint failed independent identity check")
        print(f"METHOD_DONE {method} sample={sample_id(video)} correct={sum(r['correct'] for r in rows)}/{len(rows)} replay_s={value['replay_wall_s']:.1f}", flush=True)
        return value
    finally:
        model._compute_attention_scores_manually = original
        model.clear_cache()


def preflight(model, video, work):
    """Prove feature ingestion and fresh scores agree with original/native paths."""
    import torch
    from decord import VideoReader
    reader = VideoReader(video["video_path"], num_threads=1)
    step = int(round(reader.get_avg_fps()) / FPS)
    indices = list(range(0, len(reader), step))[:16]
    if len(indices) != 16:
        raise ValueError("Preflight needs 16 sampled frames")
    frames = torch.from_numpy(reader.get_batch(indices).asnumpy())
    reset(model)
    original = model.get_video_features
    captured = []
    def capture(pixels):
        features = original(pixels)
        captured.append(features.detach().cpu())
        return features
    model.get_video_features = capture
    try:
        model.encode_video_chunk(frames)
    finally:
        model.get_video_features = original
    cache = tuple((k.clone(), v.clone()) for k, v in model.kv_cache)
    positions = [p.clone() for p in model._position_ids_cache]
    reset(model)
    ingest_features(model, captured[0], len(frames))
    for (ka, va), (kb, vb), pa, pb in zip(cache, model.kv_cache, positions, model._position_ids_cache):
        torch.testing.assert_close(ka, kb, atol=0, rtol=0)
        torch.testing.assert_close(va, vb, atol=0, rtol=0)
        torch.testing.assert_close(pa, pb, atol=0, rtol=0)
    del cache
    comparisons = []
    for stage in ("before_pruning", "after_position_sensitive_pruning"):
        if stage != "before_pruning":
            # Constrain selection to force real pruning and summary positions.
            keep = []
            for i, length in enumerate(model._get_cache_seq_len_per_layer()):
                n = 255 if i >= model.long_term_threshold else 256
                keep.append(torch.cat((torch.arange(model.visual_start_idx, device=model.device),
                                       torch.arange(length - n, length, device=model.device))))
            model.apply_kv_cache_pruning_strict(keep)
        ids = model.processor.tokenizer("Describe objects and earlier events in the scene.", return_tensors="pt").input_ids.to(model.device)
        previous = tuple((k.clone(), v.clone()) for k, v in model.kv_cache)
        before_positions = [p.clone() for p in model._position_ids_cache]
        before_activity = [p.clone() if p is not None else None for p in model.token_activity_cache]
        actual = causal_scores(model, ids)
        expected = reference_scores(model, ids)
        max_error = 0.0
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=2e-3, rtol=2e-3)
            max_error = max(max_error, float((a - b).abs().max()))
            # Explicit causality must give future probe tokens no mass.
            q = ids.shape[1]
            assert not torch.any(torch.triu(a[..., -q:], diagonal=1))
        for (ka, va), (kb, vb), pa, pb in zip(previous, model.kv_cache, before_positions, model._position_ids_cache):
            torch.testing.assert_close(ka, kb, atol=0, rtol=0)
            torch.testing.assert_close(va, vb, atol=0, rtol=0)
            torch.testing.assert_close(pa, pb, atol=0, rtol=0)
        assert all(a is None and b is None or a is not None and torch.equal(a, b)
                   for a, b in zip(before_activity, model.token_activity_cache))
        comparisons.append(dict(stage=stage, max_attention_error=max_error,
                                cache_lengths=model._get_cache_seq_len_per_layer()))
        del actual, expected, previous
    result = dict(status="pass", feature_ingestion="bitwise_equal_KV_and_positions",
                  causal_scorer_comparisons=comparisons, source_cache_unchanged=True)
    atomic_json(work / "preflight.json", result)
    model.clear_cache()
    print("PREFLIGHT_PASS: feature reuse, native attention, causality and cache immutability", flush=True)
    return result


def export_results(results, selected, work, config):
    targets = {r["case_id"] for r in selected}
    flat = [dict(method=x["method"], **r) for x in results for r in x["predictions"]]
    write_csv(work / "exports/predictions.csv", flat)
    summary, pairs = [], []
    for method in METHODS:
        runs = [x for x in results if x["method"] == method]
        rows = [r for x in runs for r in x["predictions"]]
        focus = [r for r in rows if f"sample_{r['sample_id']:03d}_q{r['qa_index']}" in targets]
        summary.append(dict(method=method, videos=len(runs), questions=len(rows),
                            correct=sum(r["correct"] for r in rows),
                            accuracy_pct=100 * sum(r["correct"] for r in rows) / len(rows),
                            selected_targets=len(focus), target_correct=sum(r["correct"] for r in focus),
                            total_replay_s=sum(x["replay_wall_s"] for x in runs),
                            peak_allocated_gib=max(x["peak_allocated_gib"] for x in runs)))
    indexed = {(r["method"], r["video_id"], r["qa_index"]): r for r in flat}
    for row in flat:
        if row["method"] != "hermes":
            continue
        f, c = [indexed[(m, row["video_id"], row["qa_index"])] for m in ("fifo", "causal")]
        pairs.append(dict(sample_id=row["sample_id"], video_id=row["video_id"], qa_index=row["qa_index"],
                          task=row["task"], hermes_correct=row["correct"], fifo_correct=f["correct"],
                          causal_correct=c["correct"], hermes_choice=row["pred_choice"],
                          fifo_choice=f["pred_choice"], causal_choice=c["pred_choice"],
                          is_selected_target=f"sample_{row['sample_id']:03d}_q{row['qa_index']}" in targets))
    write_csv(work / "exports/summary.csv", summary)
    write_csv(work / "exports/paired_questions.csv", pairs)
    atomic_json(work / "exports/run_config.json", config)
    lines = ["# AutoDL causal-scorer screen", "", "Exploratory outcome-selected development reuse. No confirmatory accuracy claim.", "",
             "Shared decode/vision feature preparation is outside method replay timings. These are not full-system streaming latencies.", "",
             "| Method | Correct / QA | Target correct | Replay seconds |", "|---|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['method']} | {r['correct']}/{r['questions']} | {r['target_correct']}/{r['selected_targets']} | {r['total_replay_s']:.1f} |")
    lines += ["", "Evidence intervals, token survival and visual causes remain unannotated. Contact sheets are sparse review aids.",
              "A fresh causal scorer is a reproduction diagnostic. This run does not establish an original algorithm."]
    (work / "exports/report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return summary


def package(work, status):
    import shutil
    from zipfile import ZIP_DEFLATED
    # The notebook writes the live subprocess log concurrently. Freeze copies
    # before hashing, otherwise archive entries can race their manifest hashes.
    for source in (work / "logs").rglob("*"):
        if source.is_file():
            target = work / "captured_logs" / source.relative_to(work / "logs")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    files = []
    for area in ("exports", "checkpoints", "review", "captured_logs"):
        files.extend(p for p in (work / area).rglob("*") if p.is_file())
    files.extend(p for name in ("preflight.json", "asset_manifest.json", "selection.json", "pilot_runner.py", "launch.json")
                 if (p := work / name).is_file())
    files.extend(p for p in (work / "features").rglob("*.csv"))
    files.extend(p for p in (work / "features").rglob("complete.json"))
    manifest = [dict(path=p.relative_to(work).as_posix(), bytes=p.stat().st_size, sha256=digest(p))
                for p in sorted(set(files))]
    atomic_json(work / "evidence_manifest.json", dict(status=status, files=manifest))
    output = work / "HERMES_causal_scorer_pilot_evidence.zip"
    temp = output.with_suffix(".tmp")
    with ZipFile(temp, "w", compression=ZIP_DEFLATED) as archive:
        for item in manifest:
            archive.write(work / item["path"], item["path"])
        archive.write(work / "evidence_manifest.json", "evidence_manifest.json")
    os.replace(temp, output)
    print(f"EVIDENCE_ZIP={output}\nSHA256={digest(output)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--per-outcome", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--max-hours", type=float, default=2.0)
    args = parser.parse_args()
    if args.model.name != MODEL_SHA:
        raise ValueError("Use the pinned model snapshot directory named by MODEL_SHA")
    work, repo = args.work.resolve(), args.repo.resolve()
    sys.path.insert(0, str(repo / "scripts"))
    sys.path.insert(0, str(repo))
    selected, videos = choose_screen(repo, args.per_outcome)
    work.mkdir(parents=True, exist_ok=True)
    review_sheet = work / "review/evidence_review.csv"
    if review_sheet.exists() and [r["case_id"] for r in read_csv(review_sheet)] != [r["case_id"] for r in selected]:
        raise ValueError("Existing review sheet has different cases; use a new WORK directory for a changed screen size")
    previous_failure = work / "exports/failure.json"
    if previous_failure.exists():
        history = work / "logs" / f"previous_failure_{time.time_ns()}.json"
        history.parent.mkdir(parents=True, exist_ok=True)
        os.replace(previous_failure, history)
    atomic_json(work / "selection.json", dict(cases=selected, source="201-250 inspected development shard",
                                             per_outcome=args.per_outcome, seed=SEED))
    assets = prepare_assets(videos, work)
    asset_by_id = {x["sample_id"]: x for x in assets["videos"]}
    for v in videos:
        v["video_path"] = asset_by_id[sample_id(v)]["path"]
    atomic_json(work / "exports/replay_annotations.json", videos)
    if not review_sheet.exists():
        write_csv(review_sheet, [dict(case_id=r["case_id"], review_status="pending",
                                    support_intervals_seconds_json="", support_visible_at_original_sampling="",
                                    failure_cause="unclassified", notes="") for r in selected])
    # Pin inference sources and assets in every checkpoint identity.
    import numpy as np
    import torch
    import transformers
    from inference.llavaov_hermes import load_model
    from video_qa.hermes_vqa import HermesVQA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    source_files = ["inference/llavaov_hermes.py", "inference/abstract_hermes.py", "inference/reindex_1d.py",
                    "video_qa/base.py", "video_qa/hermes_vqa.py", "scripts/build_failure_pilot.py"]
    config = dict(model_id=MODEL_ID, model_revision=MODEL_SHA, kv_size=4000, sample_fps=FPS, seed=SEED,
                  source_sha256={p: digest(repo / p) for p in source_files},
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                  torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0),
                  methods=METHODS, shared_features=True, feature_cache_dtype="float16",
                  assets={str(x["sample_id"]): x["sha256"] for x in assets["videos"]},
                  conversations=[dict(video_id=v["video_id"], conversations=v["conversations"]) for v in videos],
                  selected_case_ids=[r["case_id"] for r in selected],
                  note="Outcome-selected development screen; full-history replay; no population inference")
    config_hash = hashlib.sha256(stable(config).encode()).hexdigest()
    config["config_hash"] = config_hash
    config["soft_budget_hours_this_invocation"] = args.max_hours
    atomic_json(work / "exports/run_config.json", config)
    model, processor = load_model(model_path=str(args.model), kv_size=4000, streaming=True, sample_fps=FPS)
    if model.language_model.config._attn_implementation != "sdpa":
        raise RuntimeError("Current HERMES control must use its original SDPA/manual-score path")
    analyzer = HermesVQA(anno=[], save_dir=str(work), sample_fps=FPS, qa_model=model, qa_processor=processor)
    results, completed_videos = [], 0
    started = time.perf_counter()
    summary = []
    try:
        with torch.inference_mode():
            preflight(model, videos[0], work)
            for index, video in enumerate(videos):
                # Stop only between complete three-method video blocks.
                if index >= 2 and args.max_hours > 0 and time.perf_counter() - started >= args.max_hours * 3600:
                    print("TIME_BUDGET_STOP: valid partial screen; rerun to resume", flush=True)
                    break
                feature_identity = hashlib.sha256(stable(dict(
                    model_revision=MODEL_SHA, source=config["source_sha256"],
                    source_video=asset_by_id[sample_id(video)]["sha256"],
                    conversations=video["conversations"], sample_fps=FPS,
                    runner=config["runner_sha256"], torch=torch.__version__)).encode()).hexdigest()
                meta = prepare_features(model, video, work, feature_identity)
                order = list(METHODS)
                random.Random(SEED + index).shuffle(order)
                for method in order:
                    results.append(run_method(model, analyzer, video, method, meta, work, config_hash))
                completed_videos += 1
                summary = export_results(results, selected, work, config)
                elapsed = time.perf_counter() - started
                print(f"BLOCK_DONE {completed_videos}/{len(videos)} elapsed_min={elapsed/60:.1f} remaining_estimate_min={(len(videos)-completed_videos)*elapsed/completed_videos/60:.1f}", flush=True)
        status = "complete" if completed_videos == len(videos) else "partial_time_budget"
        atomic_json(work / "exports/completion.json", dict(status=status, videos=completed_videos,
                    planned_videos=len(videos), methods=list(METHODS), summary=summary,
                    config_hash=config_hash, elapsed_s=time.perf_counter() - started))
        package(work, status)
        print("PILOT_COMPLETE" if status == "complete" else "PILOT_PARTIAL", flush=True)
    except Exception as exc:
        atomic_json(work / "exports/failure.json", dict(status="failed", type=type(exc).__name__,
                    detail=str(exc)[-1500:], completed_video_blocks=completed_videos))
        atomic_json(work / "exports/completion.json", dict(status="failed", config_hash=config_hash,
                    completed_video_blocks=completed_videos, planned_videos=len(videos)))
        package(work, "failed_partial")
        raise


if __name__ == "__main__":
    main()
