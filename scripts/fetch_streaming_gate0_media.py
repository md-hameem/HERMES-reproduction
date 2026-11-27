"""Fetch or verify only the six Gate 0 clips from StreamingBench shard ZIPs."""
import argparse
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote
from zipfile import BadZipFile, ZipFile

import requests

from autodl_failure_pilot import HTTPRangeFile, RangeUnsupported, extract_members
from profile_streaming_gate0 import inputs, sha256

SHARD_BYTES = {
    "151_200": 17984397733,
    "251_300": 18838513104,
    "401_450": 18224908176,
    "451_500": 13468580983,
}
ENDPOINTS = ("https://hf-mirror.com", "https://huggingface.co")


def media_path(directory, clip):
    return directory / Path(clip["video_path"]).name


def verified(directory, clip):
    path = media_path(directory, clip)
    return (path.is_file() and path.stat().st_size == clip["asset"]["bytes"]
            and sha256(path) == clip["asset"]["sha256"])


def fetch(directory, clips):
    shards = defaultdict(list)
    for clip in clips:
        if not verified(directory, clip):
            shards[clip["shard"]].append(clip)
    with requests.Session() as session:
        for shard, pending in shards.items():
            archive = quote(f"Real-Time Visual Understanding_{shard.replace('_', '-')}.zip")
            for endpoint in ENDPOINTS:
                url = f"{endpoint}/datasets/mjuicem/StreamingBench/resolve/main/{archive}?download=true"
                try:
                    with HTTPRangeFile(url, SHARD_BYTES[shard], session) as reader, ZipFile(reader) as source:
                        for clip in pending:
                            if not verified(directory, clip):
                                extract_members(source, [clip], directory)
                                if not verified(directory, clip):
                                    raise ValueError(f"Archived media hash differs for {media_path(directory, clip)}")
                        print(f"Range transfer {shard}: {reader.bytes_received:,} bytes", flush=True)
                    break
                except (OSError, ValueError, KeyError, BadZipFile, RangeUnsupported,
                        requests.RequestException) as exc:
                    if endpoint == ENDPOINTS[-1]:
                        raise RuntimeError(f"Selected-member download failed for shard {shard}") from exc
                    print(f"Retry {shard} on official Hugging Face endpoint: {exc}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Require local files and check their hashes")
    args = parser.parse_args()
    clips = inputs()
    if not args.check:
        fetch(args.media_dir, clips)
    for clip in clips:
        path = media_path(args.media_dir, clip)
        if not verified(args.media_dir, clip):
            raise SystemExit(f"Missing or mismatched media: {path}")
        print(f"{path.name}: SHA-256 OK", flush=True)


if __name__ == "__main__":
    main()
