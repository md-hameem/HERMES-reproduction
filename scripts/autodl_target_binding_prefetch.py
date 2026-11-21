"""Prefetch the existing target-binding run's media with bounded HTTP requests."""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def bounded_reader(pilot, curl, direct):
    """Keep the runner's seek and byte accounting while bounding each transfer."""
    class BoundedRangeFile(pilot.HTTPRangeFile):
        def __init__(self, url, size, session):
            super().__init__(url, size, session)
            location = urlsplit(url)
            alternate = "hf-mirror.com" if location.hostname == "huggingface.co" else "huggingface.co"
            self.endpoints = [url, urlunsplit(location._replace(netloc=alternate))]

        def fetch(self, start, end):
            count = end - start + 1
            expected = f"bytes {start}-{end}/{self.size}"
            proxy = ["--noproxy", "*"] if direct else []
            for endpoint in list(self.endpoints):
                host = urlsplit(endpoint).hostname
                pilot.report_progress("dataset_range_request", endpoint=host, range_start=start,
                                      request_bytes=count, received_mib=round(self.bytes_received / 1024**2, 2))
                with tempfile.TemporaryDirectory(prefix="hermes_range_") as scratch:
                    headers, body = Path(scratch) / "headers", Path(scratch) / "body"
                    command = [curl, "-q", "-fsSL", "--connect-timeout", "8", "--max-time", "50",
                               "--speed-limit", "32768", "--speed-time", "12", "--max-redirs", "8",
                               # Keep a transfer ceiling independent of the requested span. Small
                               # ZIP tail reads can redirect through responses longer than the span.
                               "--max-filesize", str(2 * self.block_size), "--header", "Accept-Encoding: identity",
                               "--range", f"{start}-{end}", "--dump-header", str(headers),
                               "--output", str(body), *proxy, endpoint]
                    try:
                        result = subprocess.run(command, capture_output=True, timeout=55)
                    except subprocess.TimeoutExpired:
                        print(f"RANGE_RETRY host={host} reason=deadline", flush=True)
                        continue
                    if result.returncode != 0:
                        print(f"RANGE_RETRY host={host} curl_exit={result.returncode}", flush=True)
                        continue
                    # The last header block is the final response after redirects.
                    raw = headers.read_bytes().replace(b"\r\n", b"\n").strip().split(b"\n\n")[-1]
                    lines = raw.decode("iso-8859-1").splitlines()
                    fields = {key.lower(): value.strip() for key, _, value in
                              (line.partition(":") for line in lines[1:]) if key}
                    if (not lines or not re.match(r"^HTTP/\S+ 206(?:\s|$)", lines[0])
                            or fields.get("content-range") != expected or not body.is_file()
                            or body.stat().st_size != count):
                        print(f"RANGE_RETRY host={host} reason=range_response_mismatch", flush=True)
                        continue
                    if endpoint != self.endpoints[0]:
                        self.endpoints.reverse()
                    return body.read_bytes()
            raise RuntimeError(f"Both hosts failed the bounded request for bytes {start}-{end}. Completed clips are saved.")

        def read(self, count=-1):
            available = max(0, self.size - self.pos)
            count = available if count is None or count < 0 else min(available, count)
            chunks = []
            while count:
                if not self.buffer_start <= self.pos < self.buffer_start + len(self.buffer):
                    start = self.pos
                    end = min(self.size - 1, start + self.block_size - 1)
                    self.buffer = self.fetch(start, end)
                    self.buffer_start = start
                    self.bytes_received += len(self.buffer)
                    pilot.report_progress("dataset_range_received", received_mib=round(self.bytes_received / 1024**2, 2))
                offset = self.pos - self.buffer_start
                take = min(count, len(self.buffer) - offset)
                chunks.append(self.buffer[offset:offset + take])
                self.pos += take
                count -= take
            return b"".join(chunks)

    return BoundedRangeFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("/root/autodl-tmp/hermes_target_binding"))
    parser.add_argument("--direct", action="store_true", help="Bypass proxy settings for media requests")
    args = parser.parse_args()
    work = args.work.resolve()
    curl = shutil.which("curl")
    if not curl:
        parser.error("curl is required")
    if not (work / "stage_runner.py").is_file() or not (work / "reference.json").is_file():
        parser.error("Run notebook setup first so its runner and frozen manifest are available")

    import fcntl

    sys.path.insert(0, str(work))
    import stage_runner as runner

    with (work / ".notebook.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("The notebook still holds the run lock. Restart its kernel before prefetching.")
        manifest = json.loads((work / "reference.json").read_text(encoding="utf-8"))
        original = runner.p.HTTPRangeFile
        runner.p.HTTPRangeFile = bounded_reader(runner.p, curl, args.direct)
        try:
            with runner.p.ProgressMonitor(work):
                runner.check_media_space(manifest["shards"], work)
                assets = runner.base.acquire_media(manifest["shards"], work)
                runner.p.report_progress("media_prefetch_complete", clips=len(assets))
        finally:
            runner.p.HTTPRangeFile = original
    print(f"MEDIA_PREFETCH_COMPLETE clips={len(assets)}", flush=True)


if __name__ == "__main__":
    main()
