# Standalone notebook cell. Uses WORK, ENV, MODEL_ID, MODEL_SHA and logged from setup.
import hashlib
import json
import re
import shutil
from pathlib import Path

SHARDS = (
    ("model-00001-of-00004.safetensors", 4911200360, "ad4cb27d2a3f6e987147619cf2508a5bbb1b6b4c2d5533501fcc4a287a90e5af"),
    ("model-00002-of-00004.safetensors", 4991497664, "cfaed9e4ba187283a936fd6df5d55148a6de5a8416f1279778b364232c05f39a"),
    ("model-00003-of-00004.safetensors", 4932752752, "86cb0bb7f7d57873b27f16fd66b54be71055e582a48ce26e30884bfb209dffd4"),
    ("model-00004-of-00004.safetensors", 1226266240, "a2d1e40e47e9b524efcbbcab68e97fe4e7aadfe9a51bc14b54cc2331c48ebff6"),
)
METADATA = (
    ("added_tokens.json", 122, "33e0fb93dadacb864bd2f2e8441e147daa2baceb67f94d3ef5283b495572cea0"),
    ("chat_template.json", 826, "2466d1704df30f0067f28d8e30e0190a1bf74e5b430942697af974d162a056bd"),
    ("config.json", 2528, "69277c2c9ba8a4f61d2e72f79fbe02e043b8c3af670858a606818527f971a0c2"),
    ("generation_config.json", 126, "89dc53229f50b59570b6852056dafeac8116c458f1a748bff491b6d4d24d3b51"),
    ("merges.txt", 1671853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    ("model.safetensors.index.json", 77952, "e389b969d3b9b7120f136fbf6592cbbc1c07326157cae6f863fb64559261dae9"),
    ("preprocessor_config.json", 1732, "3644c108b9f0fa53e62ff422a9be6639642f0e64dab4a71f961c7911d4386384"),
    ("processor_config.json", 178, "04e9899e93f2a412c94e153cab4081457c9a44defb3b2c0b9df673d42c42cdd0"),
    ("special_tokens_map.json", 367, "f4f79e08d97f4d1c87f8d89264f525c8789da3b73b3bb55d1e12f692f41a7b1b"),
    ("tokenizer.json", 7028579, "3c0ce3213b50ff38d8aa1e91136a2d2cb142a3f569246170872e439cb2a29d15"),
    ("tokenizer_config.json", 1801, "b3151cdc153a44fa4e28abc8fc83567fc663aef7cd9dc041f5d66e617fe10b8e"),
    ("video_preprocessor_config.json", 621, "0ea9b672282b78353960b6069a521ed496a9f07c033e1e3362bff669234caa8d"),
    ("video_processor/preprocessor_config.json", 428, "1e71b2e75b90ddf696692529485b4a75fd54ecd4bbc03e2cc7be4af032875765"),
    ("vocab.json", 2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
)


def file_bytes(path):
    return path.stat().st_size if path.is_file() else 0


def verified(path, size, expected):
    if file_bytes(path) != size:
        return False
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() == expected


def range_matches(headers, offset, end, total):
    replies = re.split(r"(?m)(?=^HTTP/)", headers.strip())
    final = replies[-1]
    return (re.match(r"HTTP/\S+\s+206\b", final) is not None
            and re.search(rf"(?im)^content-range:\s*bytes {offset}-{end}/{total}\s*$", final) is not None)


def fetch_pinned(name, size, expected, destination):
    if verified(destination, size, expected):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".incomplete")
    direct_env = {k: v for k, v in ENV.items() if k.lower() not in {
        "http_proxy", "https_proxy", "all_proxy", "no_proxy"}}
    endpoints = (
        (f"https://modelscope.cn/models/{MODEL_ID}/resolve/master/{name}", direct_env),
        (f"https://hf-mirror.com/{MODEL_ID}/resolve/{MODEL_SHA}/{name}", ENV),
        (f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_SHA}/{name}", ENV),
    )
    offset = file_bytes(partial)
    if offset > size:
        raise RuntimeError(f"Oversized partial file: {partial} ({offset} bytes, expected {size})")
    chunk = partial.with_suffix(".range")
    headers = partial.with_suffix(".headers")
    while offset < size:
        end = min(offset + 64 * 1024**2, size) - 1
        length = end - offset + 1
        for url, env in endpoints:
            host = url.split("/")[2]
            print(f"MODEL_RANGE {name} bytes={offset}-{end}/{size} source={host}", flush=True)
            try:
                logged(["curl", "--fail", "--location", "--range", f"{offset}-{end}", "--connect-timeout", "15",
                        "--speed-limit", "16384", "--speed-time", "90", "--max-time", "600",
                        "--max-filesize", str(length), "--dump-header", headers, "--output", chunk,
                        "--progress-bar", url], "model_download.log", env)
            except RuntimeError:
                print(f"MODEL_ENDPOINT_RETRY {name} source={host}", flush=True)
                continue
            response = headers.read_text(encoding="iso-8859-1")
            # Some mirrors answer a request for an entire small file with a plain HTTP 200.
            whole_file = offset == 0 and length == size and verified(chunk, size, expected)
            if file_bytes(chunk) != length or not (whole_file or range_matches(response, offset, end, size)):
                print(f"MODEL_RANGE_REJECTED {name} source={host}", flush=True)
                continue
            with partial.open("ab") as output, chunk.open("rb") as incoming:
                if output.tell() != offset:
                    raise RuntimeError(f"Another downloader changed {partial}. Stop concurrent downloads.")
                try:
                    shutil.copyfileobj(incoming, output, length=1024**2)
                except BaseException:
                    output.flush()
                    output.truncate(offset)
                    raise
            offset += length
            print(f"MODEL_PROGRESS {name} {offset}/{size} bytes", flush=True)
            break
        else:
            raise RuntimeError(f"All endpoints failed for {name} at byte {offset}. Rerun to resume {partial}.")
    chunk.unlink(missing_ok=True)
    headers.unlink(missing_ok=True)
    if not verified(partial, size, expected):
        raise RuntimeError(f"SHA-256 mismatch in {partial}. The file was kept for inspection.")
    partial.replace(destination)


if not VALIDATE_ONLY:
    assert MODEL_ID == "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    assert MODEL_SHA == "0d50680527681998e456c7b78950205bedd8a068"
    repo = Path(ENV["HF_HOME"]) / "hub" / ("models--" + MODEL_ID.replace("/", "--"))
    blobs = repo / "blobs"
    MODEL_PATH = repo / "snapshots" / MODEL_SHA
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    missing = sum(max(0, size - file_bytes(blobs / (sha + ".incomplete")))
                  for name, size, sha in SHARDS
                  if file_bytes(MODEL_PATH / name) != size and file_bytes(blobs / sha) != size)
    free = shutil.disk_usage(MODEL_PATH).free
    assert free >= missing + 2 * 1024**3, f"Need {(missing + 2 * 1024**3 - free) / 1024**3:.1f} GiB more free"
    print(f"MODEL_CACHE remaining_weight_download={missing / 1024**3:.2f} GiB", flush=True)
    for name, size, sha in SHARDS:
        path = MODEL_PATH / name
        if not verified(path, size, sha):
            blob = blobs / sha
            fetch_pinned(name, size, sha, blob)
            pending_link = path.with_name(path.name + ".verified-link")
            if pending_link.is_symlink():
                pending_link.unlink()
            pending_link.symlink_to(Path("../../blobs") / sha)
            pending_link.replace(path)
        print("MODEL_SHARD_VERIFIED", name, flush=True)
    for name, size, sha in METADATA:
        fetch_pinned(name, size, sha, MODEL_PATH / name)
    index = json.loads((MODEL_PATH / "model.safetensors.index.json").read_bytes())
    assert set(index["weight_map"].values()) == {name for name, _, _ in SHARDS}
    (WORK / "model_path.json").write_text(json.dumps(str(MODEL_PATH)), encoding="utf-8")
    print("MODEL_REVISION_PASS", MODEL_SHA, flush=True)
