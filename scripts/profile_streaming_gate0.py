"""Gate 0 profiler for the six frozen StreamingBench clips."""
import argparse
import ast
import csv
import hashlib
import inspect
import json
import math
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from functools import wraps
from pathlib import Path
from types import MethodType, SimpleNamespace
from zipfile import ZipFile

from build_failure_pilot import ROOT, load_predictions

ARCHIVE = ROOT / "evidence/archives/HERMES_AutoDL_five_stage_evidence.zip"
ARCHIVE_SHA256 = "b8bc4cf8a3b28e6854b2e76b8c0239b12d3b5598ee1a07dafc03d0326a900518"
CLIPS = ((290, 438), (278, 69), (480, 471), (200, 27), (453, 81), (449, 132))
ARCHIVED_RESETS = {438: 1, 69: 1, 471: 2, 27: 2, 81: 2, 132: 3}
ARMS = {"fifo4k": ("fifo", 4000), "hermes4k": ("hermes", 4000),
        "fifo784": ("fifo", 784), "hermes784": ("hermes", 784)}
STAGES = {"preprocessing", "vision_tower", "chunk_prefill", "compression", "qa_prefill", "qa_decode"}
CSV_FIELDS = {
    "parity": "video_id arm qa_index choice correct archived_choice choice_matches",
    "stages": "video_id arm qa_index chunk_index stage calls seconds entry_allocated_mib peak_allocated_mib peak_reserved_mib frames visual_tokens",
    "latency": "video_id arm qa_index ttft_s answer_s first_token_id",
}
JSONL_FIELDS = {
    "cache": "video_id arm qa_index chunk_index sampled_start sampled_stop timestamp_s pre_slots post_slots max_pre_position compacted",
    "rope": "video_id arm qa_index chunk_index sampled_start sampled_stop timestamp_s pre_positions post_positions",
}


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def sample_id(video):
    return int(re.search(r"sample_(\d+)", video["video_path"])[1])


def inputs():
    annotations = json.loads((ROOT / "data/streamingbench/streamingbench_realtime.json").read_text(encoding="utf-8"))
    by_sample = {sample_id(v): v for v in annotations}
    if sha256(ARCHIVE) != ARCHIVE_SHA256:
        raise ValueError("The canonical benchmark archive hash differs")
    with ZipFile(ARCHIVE) as archive:
        archived = {m: load_predictions(archive.read(f"exports/full_{m}_4000_predictions.csv"), annotations)
                    for m in ("fifo", "hermes")}
        clips = []
        for sid, vid in CLIPS:
            video = by_sample[sid]
            if video["video_id"] != vid or len(video["conversations"]) != 5:
                raise ValueError(f"Unexpected frozen clip identity: {sid}")
            start = ((sid - 1) // 50) * 50 + 1
            shard = f"{start}_{start+49}"
            records = json.loads(archive.read(f"data/streamingbench_{shard}.json"))
            assets = json.loads(archive.read(f"data/streamingbench_{shard}_assets.json"))["videos"]
            index = [sample_id(v) for v in records].index(sid)
            if len(assets) != len(records) or records[index]["conversations"] != video["conversations"]:
                raise ValueError(f"Archived assets and annotations differ for {sid}")
            expected = {m: [{"choice": archived[m][str(vid), i]["pred_choice"],
                             "correct": archived[m][str(vid), i]["correct"]}
                            for i in range(5)] for m in archived}
            clips.append(dict(video, sample_id=sid, shard=shard, asset=assets[index], expected=expected))
    return clips


def write_outputs(folder, data):
    folder.mkdir(parents=True, exist_ok=False)
    for name, fields in CSV_FIELDS.items():
        with (folder / f"{name}.csv").open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fields.split())
            writer.writeheader()
            writer.writerows(data[name])
    for name, fields in JSONL_FIELDS.items():
        with (folder / f"{name}.jsonl").open("w", encoding="utf-8") as out:
            for row in data[name]:
                if set(row) != set(fields.split()):
                    raise ValueError(f"Invalid {name} row")
                out.write(json.dumps(row, allow_nan=False) + "\n")
    (folder / "run.json").write_text(json.dumps(data["run"], indent=2, allow_nan=False), encoding="utf-8")


def validate_outputs(folder, clip, arm):
    method, budget = ARMS[arm]
    loaded = {}
    for name, fields in CSV_FIELDS.items():
        with (folder / f"{name}.csv").open(encoding="utf-8", newline="") as src:
            reader = csv.DictReader(src)
            if reader.fieldnames != fields.split():
                raise ValueError(f"Invalid {name} header")
            loaded[name] = list(reader)
    for name, fields in JSONL_FIELDS.items():
        loaded[name] = [json.loads(line) for line in (folder / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
        if any(set(row) != set(fields.split()) for row in loaded[name]):
            raise ValueError(f"Invalid {name} fields")

    def check(condition, message):
        if not condition:
            raise ValueError(f"{arm}, video {clip['video_id']}: {message}")

    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    check(all(run.get(k) for k in ("gpu", "pytorch", "cuda", "dtype", "model", "archive_sha256",
                                  "media_sha256", "measurement_mode")), "missing run provenance")
    check(run["archive_sha256"] == ARCHIVE_SHA256, "archive provenance differs")
    check(run["media_sha256"] == clip["asset"]["sha256"], "media provenance differs")
    checks = loaded["parity"]
    check([int(r["qa_index"]) for r in checks] == list(range(5)), "parity coverage differs")
    for i, row in enumerate(checks):
        q = clip["conversations"][i]
        gold = "ABCDEFGH"[q["choices"].index(q["answer"])]
        check(int(row["correct"]) == int(row["choice"] == gold), "incorrect scoring")
        if budget == 4000:
            ref = clip["expected"][method][i]
            check(row["choice"] == row["archived_choice"] == ref["choice"]
                  and int(row["correct"]) == ref["correct"] and row["choice_matches"] == "True",
                  f"archived answer differs at q{i}")

    latency = loaded["latency"]
    check([int(r["qa_index"]) for r in latency] == list(range(5)), "latency coverage differs")
    for row in latency:
        check(0 <= float(row["ttft_s"]) <= float(row["answer_s"]) < math.inf
              and int(row["first_token_id"]) >= 0, "invalid question latency")

    chunks = loaded["cache"]
    cursor = 0
    expected = []
    for i, q in enumerate(clip["conversations"]):
        stop = math.ceil(q["end_time"] * .5)
        for start in range(cursor, stop, 16):
            expected.append((i, start, min(start + 16, stop)))
        cursor = stop
    check([(r["qa_index"], r["sampled_start"], r["sampled_stop"]) for r in chunks] == expected,
          "chunk schedule differs")
    for i, row in enumerate(chunks):
        start, stop = row["sampled_start"], row["sampled_stop"]
        num_layers = len(row["pre_slots"])
        check(row["chunk_index"] == i and num_layers > 0 and
              row["pre_slots"] == [13 + min(budget, 196 * start) + 196 * (stop - start)] * num_layers and
              row["post_slots"] == [13 + min(budget, 196 * stop)] * num_layers, "physical KV lengths differ")
        check(row["compacted"] == (max(row["pre_slots"]) > budget and
                                  row["max_pre_position"] > run["rope_threshold"]) and
              0 <= row["timestamp_s"] < math.inf, "invalid compression trace")
    for rows in loaded.values():
        check(all(int(r["video_id"]) == clip["video_id"] and r["arm"] == arm for r in rows),
              "output identity differs")
    resets = loaded["rope"]
    check(bool(resets), "long fixture produced no compaction traces")
    check([r["chunk_index"] for r in resets] == [r["chunk_index"] for r in chunks if r["compacted"]],
          "reset coverage differs")
    if budget == 4000:
        check(len(resets) == ARCHIVED_RESETS[clip["video_id"]], "archived reset count differs")
    for row in resets:
        chunk = chunks[row["chunk_index"]]
        check(all(row[k] == chunk[k] for k in ("qa_index", "sampled_start", "sampled_stop", "timestamp_s")),
              "compaction and chunk identity differ")
        check([len(p) for p in row["pre_positions"]] == chunk["pre_slots"] and
              [len(p) for p in row["post_positions"]] == chunk["post_slots"] and
              max(max(p) for p in row["pre_positions"]) == chunk["max_pre_position"] and
              all(p == list(range(len(p))) for p in row["post_positions"]), "invalid RoPE reset")

    stages = loaded["stages"]
    check({r["stage"] for r in stages} == STAGES | {"other"}, "missing stage coverage")
    for row in stages:
        calls = int(row["calls"])
        valid_calls = calls > 0 or (calls == 0 and row["stage"] == "qa_decode" and
                                    all(float(row[k]) == 0 for k in ("seconds", "peak_allocated_mib",
                                                                     "peak_reserved_mib")))
        check(0 <= float(row["seconds"]) < math.inf and valid_calls and
              0 <= float(row["entry_allocated_mib"]) <= float(row["peak_allocated_mib"]) <=
              float(row["peak_reserved_mib"]) < math.inf, "invalid stage timings or peaks")
    for name in ("preprocessing", "vision_tower", "chunk_prefill", "compression"):
        check([int(r["chunk_index"]) for r in stages if r["stage"] == name] == list(range(len(chunks))),
              f"incomplete {name} coverage")
    for name in ("qa_prefill", "qa_decode"):
        check([int(r["qa_index"]) for r in stages if r["stage"] == name] == list(range(5)),
              f"incomplete {name} coverage")
    for row in stages:
        if row["stage"] in {"vision_tower", "chunk_prefill"}:
            chunk = chunks[int(row["chunk_index"])]
            check(int(row["frames"]) == chunk["sampled_stop"] - chunk["sampled_start"] and
                  int(row["visual_tokens"]) == 196 * int(row["frames"]), "incorrect visual tensor shape")
    for metric in ("peak_allocated_mib", "peak_reserved_mib"):
        check(run[metric] == max(float(r[metric]) for r in stages), "whole-stream peak does not reconcile")
    return run


class Profiler:
    def __init__(self, cuda, device, identity):
        self.cuda, self.device, self.identity = cuda, device, identity
        self.rows = []
        self.context = dict(qa_index=-1, chunk_index=-1, frames="", visual_tokens="")
        self.enabled = False

    def sync(self):
        self.cuda.synchronize(self.device)

    def start(self, stage):
        self.sync()
        self.cuda.reset_peak_memory_stats(self.device)
        self.current = (stage, self.context.copy(), time.perf_counter(), self.cuda.memory_allocated(self.device))

    def end(self):
        self.sync()
        stage, ctx, start, allocated = self.current
        self.rows.append(dict(self.identity, **ctx, stage=stage, calls=1,
                              seconds=time.perf_counter() - start, entry_allocated_mib=allocated / 2**20,
                              peak_allocated_mib=self.cuda.max_memory_allocated(self.device) / 2**20,
                              peak_reserved_mib=self.cuda.max_memory_reserved(self.device) / 2**20))

    def begin(self):
        self.enabled = True
        self.start("other")

    def finish(self):
        self.end()
        self.enabled = False

    def call(self, stage, operation, *args, **kwargs):
        if not self.enabled:
            return operation(*args, **kwargs)
        self.end()
        self.start(stage)
        try:
            return operation(*args, **kwargs)
        finally:
            self.end()
            self.start("other")

    def consolidated(self):
        # Merge decode forwards and unattributed gaps within the same question/chunk.
        results = {}
        for row in self.rows:
            key = row["stage"], row["qa_index"], row["chunk_index"]
            if key not in results:
                results[key] = row.copy()
                continue
            acc = results[key]
            for field in ("peak_allocated_mib", "peak_reserved_mib"):
                acc[field] = max(acc[field], row[field])
            acc["calls"] += row["calls"]
            acc["seconds"] += row["seconds"]
        return list(results.values())


def install_fifo(model, torch):
    with ZipFile(ARCHIVE) as z:
        tree = ast.parse(z.read("scripts/experiment_runner.py"))
    names = {"fifo_keep_indices", "predict_and_compress_fifo"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {f.name for f in functions} != names:
        raise ValueError("Archived FIFO policy unavailable")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<archived FIFO policy>", "exec"), namespace)
    model.predict_and_compress = MethodType(namespace["predict_and_compress_fifo"], model)


def instrument_answer(model, first_token):
    # Keep the audited decoder intact except for a callback at the first selected token.
    original = inspect.unwrap(model.question_answering)
    source = textwrap.dedent(inspect.getsource(original))
    source = source[source.index("def question_answering("):]
    anchor = "        output_ids.append(token)\n"
    if source.count(anchor) != 1:
        raise ValueError("Audited first-token boundary changed")
    source = source.replace(anchor, anchor + "        if step == 0:\n            gate0_first_token(token)\n")
    namespace = dict(original.__globals__, gate0_first_token=first_token)
    exec(compile(source, "<Gate 0 decoder probe>", "exec"), namespace)
    model.question_answering = MethodType(namespace["question_answering"], model)


def run_arm(args, clip, arm):
    import numpy as np
    import torch

    with ZipFile(ARCHIVE) as archive:
        config = json.loads(archive.read("results/generalization_251_300/fifo_kv4000/config.json"))
    for name, expected in config["source_sha256"].items():
        source = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError(f"Audited inference source differs: {name}")

    sys.path.insert(0, str(ROOT))
    from inference.llavaov_hermes import load_model
    from video_qa.hermes_vqa import HermesVQA

    method, budget = ARMS[arm]
    model, processor = load_model(model_path=str(args.model_path), kv_size=budget, streaming=True, sample_fps=.5)
    if model.device != torch.device("cuda:0") or {p.device for p in model.parameters()} != {model.device}:
        raise ValueError("Gate 0 requires a single visible GPU with the entire model on cuda:0")
    if model.dtype != torch.float16:
        raise ValueError("Gate 0 requires the archived FP16 model dtype")
    if method == "fifo":
        install_fifo(model, torch)
    analyzer = HermesVQA([clip], str(args.output_dir), .5, model, processor)
    frames = np.load(args.frames, mmap_mode="r", allow_pickle=False)
    stamps = json.loads(args.timestamps.read_text(encoding="utf-8"))
    if len(frames) != len(stamps) or len(frames) < 16:
        raise ValueError("Invalid staged frames or timestamps")
    identity = dict(video_id=clip["video_id"], arm=arm)
    profiler = Profiler(torch.cuda, model.device, identity)
    phase = None
    forwards = 0
    first = None
    n_frames = 0
    started = 0.0

    def first_token(token):
        nonlocal first
        if phase == "latency":
            profiler.sync()
            first = (time.perf_counter() - started, token)

    instrument_answer(model, first_token)
    video_processor = processor.video_processor
    vision_forward = model.vision_tower.forward
    forward = model.language_model.forward

    @wraps(forward)
    def forward_probe(*a, **kw):
        nonlocal forwards
        if phase == "encode":
            tokens = kw["inputs_embeds"].shape[1]
            if tokens != n_frames * 196:
                raise ValueError(f"Unexpected visual input width: {tokens}")
            profiler.context["visual_tokens"] = tokens
            if not profiler.rows or profiler.rows[-1]["stage"] != "vision_tower":
                raise ValueError("Visual tokens could not be linked to the vision stage")
            profiler.rows[-1]["visual_tokens"] = tokens
            return profiler.call("chunk_prefill", forward, *a, **kw)
        if phase == "answer":
            name = "qa_prefill" if forwards == 0 else "qa_decode"
            forwards += 1
            return profiler.call(name, forward, *a, **kw)
        return forward(*a, **kw)

    def preprocess(*a, **kw):
        def prepare():
            pixels = video_processor(*a, **kw).pixel_values_videos.to(model.device, model.dtype)
            return SimpleNamespace(pixel_values_videos=pixels)
        return profiler.call("preprocessing", prepare)

    processor.video_processor = preprocess
    model.vision_tower.forward = wraps(vision_forward)(
        lambda *a, **kw: profiler.call("vision_tower", vision_forward, *a, **kw))
    model.language_model.forward = forward_probe
    threshold = getattr(model.language_model.config, "max_position_embeddings", 32768) - 1024
    data = {key: [] for key in (*CSV_FIELDS, *JSONL_FIELDS)}

    def qa(q):
        choices = q["choices"]
        return analyzer.video_close_qa(q["question"], choices, "ABCDEFGH"[choices.index(q["answer"])])

    def encode(start, stop):
        return model.encode_video_chunk(torch.from_numpy(frames[start:stop].copy()))

    def replay(profile):
        nonlocal phase, forwards, first, n_frames, started
        model.clear_cache()
        if profile:
            profiler.begin()
        model.encode_init_prompt()
        if model.visual_start_idx != 13:
            raise ValueError("Unexpected initial prompt length")
        cursor = chunk = 0
        results = []
        for qi, q in enumerate(clip["conversations"]):
            started = time.perf_counter()
            first = None
            stop = math.ceil(q["end_time"] * .5)
            if stop > len(frames):
                raise ValueError("Staged video ends before the archived question cutoff")
            while cursor < stop:
                end = min(cursor + 16, stop)
                n_frames = end - cursor
                profiler.context = dict(qa_index=qi, chunk_index=chunk, frames=n_frames, visual_tokens="")
                phase = "encode" if profile else None
                encode(cursor, end)
                phase = None
                if profile:
                    pre = [int(k.shape[2]) for k, _ in model.kv_cache]
                    positions = model._position_ids_cache
                    max_position = max(int(p[-1]) for p in positions)
                    reset = max(pre) > budget and max_position > threshold
                    old = [p.tolist() for p in positions] if reset else None
                profiler.call("compression", model.predict_and_compress)
                if profile:
                    base = dict(identity, qa_index=qi, chunk_index=chunk, sampled_start=cursor,
                                sampled_stop=end, timestamp_s=stamps[end - 1])
                    after = [int(k.shape[2]) for k, _ in model.kv_cache]
                    data["cache"].append(dict(base, pre_slots=pre, post_slots=after,
                                              max_pre_position=max_position, compacted=reset))
                    if reset:
                        data["rope"].append(dict(base, pre_positions=old,
                                                 post_positions=[p.tolist() for p in model._position_ids_cache]))
                cursor, chunk = end, chunk + 1
            profiler.context = dict(qa_index=qi, chunk_index=-1, frames="", visual_tokens="")
            forwards = 0
            phase = "answer" if profile else "latency"
            result = qa(q)
            phase = None
            if profile and forwards == 1:
                # First-token EOS has no second forward, so the decode interval has zero calls.
                profiler.rows.append(dict(identity, **profiler.context, stage="qa_decode", calls=0,
                                          seconds=0.0, entry_allocated_mib=0.0,
                                          peak_allocated_mib=0.0, peak_reserved_mib=0.0))
            if not profile:
                profiler.sync()
                answer_s = time.perf_counter() - started
                if first is None:
                    raise ValueError("Missing synchronized first-token timestamp")
                data["latency"].append(dict(identity, qa_index=qi, ttft_s=first[0],
                                             answer_s=answer_s, first_token_id=first[1]))
            results.append(result)
        if profile:
            profiler.finish()
        return results

    with torch.inference_mode():
        # Warm real full-chunk and answer kernels on the staged clip, then reset all stream state.
        for profile in (True, False):
            model.clear_cache()
            model.encode_init_prompt()
            encode(0, 16)
            model.predict_and_compress()
            qa(clip["conversations"][0])
            profiler.sync()
            if profile:
                profiled = replay(True)
                stages = profiler.consolidated()
            else:
                measured = replay(False)
    for i, (a, b) in enumerate(zip(profiled, measured, strict=True)):
        if a["pred_choice"] != b["pred_choice"] or a["acc"] != b["acc"]:
            raise ValueError(f"Profiling and latency replay disagree on q{i}")
        ref = clip["expected"][method][i] if budget == 4000 else None
        data["parity"].append(dict(identity, qa_index=i, choice=b["pred_choice"], correct=int(b["acc"]),
                                   archived_choice=ref["choice"] if ref else "",
                                   choice_matches=b["pred_choice"] == ref["choice"] if ref else ""))
    data["stages"] = stages
    data["run"] = dict(identity, gpu=torch.cuda.get_device_name(0), pytorch=str(torch.__version__),
                       cuda=torch.version.cuda, dtype=str(model.dtype), model=str(args.model_path),
                       model_config_sha256=sha256(args.model_path / "config.json"),
                       source_sha256=config["source_sha256"],
                       archive_sha256=ARCHIVE_SHA256, media_sha256=clip["asset"]["sha256"],
                       measurement_mode="isolated profile and warmed lean replay", rope_threshold=threshold,
                       **{k: max(r[k] for r in stages) for k in ("peak_allocated_mib", "peak_reserved_mib")})
    target = args.output_dir / str(clip["sample_id"]) / arm
    write_outputs(target, data)
    validate_outputs(target, clip, arm)


def stage_clip(media, target):
    import numpy as np
    from decord import VideoReader

    reader = VideoReader(str(media), num_threads=1)
    indexes = list(range(0, len(reader), int(round(reader.get_avg_fps()) / .5)))
    np.save(target / "frames.npy", reader.get_batch(indexes).asnumpy(), allow_pickle=False)
    stamps = reader.get_frame_timestamp(indexes)[:, 0].tolist()
    (target / "timestamps.json").write_text(json.dumps(stamps, allow_nan=False), encoding="utf-8")


def run_all(args, clips):
    media = [args.media_dir / Path(clip["video_path"]).name for clip in clips]
    for path, clip in zip(media, clips, strict=True):
        if not path.is_file() or path.stat().st_size != clip["asset"]["bytes"] or sha256(path) != clip["asset"]["sha256"]:
            raise ValueError(f"Missing or mismatched media: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    for path, clip in zip(media, clips, strict=True):
        with tempfile.TemporaryDirectory(prefix="hermes-gate0-", dir=args.scratch_dir) as scratch:
            start = time.perf_counter()
            stage_clip(path, Path(scratch))
            stage_s = time.perf_counter() - start
            for arm in ARMS:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--arm", arm,
                                "--sample-id", str(clip["sample_id"]), "--model-path", str(args.model_path),
                                "--output-dir", str(args.output_dir), "--frames", str(Path(scratch) / "frames.npy"),
                                "--timestamps", str(Path(scratch) / "timestamps.json")], check=True)
            (args.output_dir / str(clip["sample_id"]) / "staging.json").write_text(
                json.dumps(dict(media=path.name, sha256=clip["asset"]["sha256"], seconds=stage_s),
                           indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("media-dir", "model-path", "output-dir", "frames", "timestamps"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--scratch-dir", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--sample-id", type=int)
    args = parser.parse_args()
    clips = inputs()
    if args.manifest:
        for clip in clips:
            print(f"{Path(clip['video_path']).name}  video_id={clip['video_id']}  shard={clip['shard']}  "
                  f"sha256={clip['asset']['sha256']}")
        return
    required = ("model_path", "output_dir", "frames", "timestamps", "sample_id") if args.arm else (
        "model_path", "output_dir", "media_dir")
    if any(getattr(args, name) is None for name in required):
        parser.error("Required arguments: " + ", ".join("--" + name.replace("_", "-") for name in required))
    if not (args.model_path / "config.json").is_file():
        parser.error("--model-path must point to the local archived model snapshot")
    if args.arm:
        run_arm(args, next(c for c in clips if c["sample_id"] == args.sample_id), args.arm)
    else:
        run_all(args, clips)
        print("Gate 0 PASS: all 24 isolated clip-arm runs validated")


if __name__ == "__main__":
    main()
