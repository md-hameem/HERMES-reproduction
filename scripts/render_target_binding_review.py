"""Create a CPU-only blind review ZIP from the original target-binding videos."""
import argparse
import csv
import html
import json
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from autodl_confirmation_visual_review import reader_for, render_sheet, verify_timestamps


def review_index(rows):
    by_question = defaultdict(list)
    for row in rows:
        by_question[row["review_id"].split("-")[0]].append(row)
    lines = ['<!doctype html><meta charset="utf-8"><title>HERMES blinded review</title>',
             '<style>body{font:16px system-ui;max-width:960px;margin:2em auto;padding:0 1em}'
             'img{max-width:100%;height:auto}summary{cursor:pointer;padding:.5em}section{margin-bottom:2em}</style>',
             '<h1>HERMES blinded visual-support review</h1>',
             '<p>Use the accompanying CSV to record the supported option, supporting sampled indices, '
             'and whether the evidence applies at the question time. Mark uncertain cases explicitly. '
             'Each reviewer should keep their own worksheet copy.</p>']
    for qid, group in by_question.items():
        question = group[0]
        lines.append(f"<section><h2>{qid} · {float(question['query_time_s']):g} s</h2>"
                     f"<p>{html.escape(question['question'])}</p><ol type='A'>")
        lines.extend(f"<li>{html.escape(option)}</li>" for option in json.loads(question["options"]))
        lines.append("</ol>")
        for row in group:
            rid = row["review_id"]
            lines.append(f'<details><summary>{rid}</summary>'
                         f'<a href="frames/{rid}.jpg"><img loading="lazy" src="frames/{rid}.jpg" alt="{rid}"></a></details>')
        lines.append("</section>")
    return "\n".join(lines)


def build_packet(worksheet, evidence, videos, output):
    with worksheet.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with ZipFile(evidence) as archive:
        assets = {a["sample_id"]: a for a in json.loads(archive.read("asset_manifest.json"))["videos"]}
        times = {sid: json.loads(archive.read(f"provenance/features_{sid}.json"))["timestamps"] for sid in assets}
    grouped = defaultdict(list)
    ids = {row["review_id"] for row in rows}
    if len(ids) != len(rows) or not ids or any(not re.fullmatch(r"Q\d{2}-[A-E]", rid) for rid in ids):
        raise ValueError("Expected unique blind review IDs")
    for row in rows:
        sid = int(row["sample_id"])
        query_time = float(row["query_time_s"])
        indices = json.loads(row["sampled_indices"])
        if len(indices) != 4 or sorted(set(indices)) != indices or indices[0] < 0:
            raise ValueError(f"Expected four ordered frame slots for {row['review_id']}")
        frames = [times[sid][i] for i in indices]
        if (any(float(f["start_s"]) > query_time + 1e-6 for f in frames)
                or [f["source_frame"] for f in frames] != json.loads(row["source_frame_indices"])
                or [f["start_s"] for f in frames] != json.loads(row["frame_start_s"])):
            raise ValueError(f"Worksheet disagrees with archived frame evidence: {row['review_id']}")
        grouped[sid].append((row["review_id"], query_time, frames))
    missing = [sid for sid in grouped if not (videos / f"sample_{sid}_real.mp4").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing original videos for sample IDs: {missing}")

    if output.exists():
        raise FileExistsError(f"Choose a new output filename: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        scratch = Path(tmp)
        dest = scratch / "review.zip"
        with ZipFile(dest, "w", compression=ZIP_DEFLATED) as archive:
            archive.write(worksheet, "visual_support_review.csv")
            archive.writestr("review.html", review_index(rows))
            for sid, windows in sorted(grouped.items()):
                reader = reader_for(dict(sample_id=sid, video_sha256=assets[sid]["sha256"]), videos)
                for rid, query_time, frames in windows:
                    normalized = [dict(f, start_s=float(f["start_s"]), source_frame=int(f["source_frame"]),
                                       sampled_index=int(f["sampled_index"])) for f in frames]
                    verify_timestamps(reader, normalized)
                    sheet = scratch / f"{rid}.jpg"
                    render_sheet(reader, normalized, f"{rid} | Question at {query_time:g} s", 2, 720, 405, sheet)
                    archive.write(sheet, f"frames/{sheet.name}")
                del reader
                print(f"REVIEW_CLIP_READY sample={sid} windows={len(windows)}", flush=True)
        dest.replace(output)
    print(f"Rendered {len(rows)} blinded windows from {len(grouped)} verified clips: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worksheet", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--videos", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build_packet(args.worksheet, args.evidence, args.videos, args.output)
