"""Render the exact saved pre-query frame slots from the verified local clips."""
import argparse
import csv
import hashlib
import html
import json
import re
from collections import defaultdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

NOTES_FIELDS = (
    "case_id", "reviewer", "support_region", "supporting_sample_indices", "reviewer_choice",
    "confidence", "ambiguity_or_motion", "notes",
)
REGIONS = ("recent_only", "older_only", "both_independently", "needs_both", "not_visible", "uncertain")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validated_frames(case):
    frames = case["frames"]
    if not re.fullmatch(r"sample_\d+_q\d+", case["case_id"]):
        raise ValueError("Unexpected case ID")
    if not 1 <= len(frames) <= 20 or [f["sampled_index"] for f in frames] != sorted({f["sampled_index"] for f in frames}):
        raise ValueError(f"Unexpected sample order: {case['case_id']}")
    for i, f in enumerate(frames):
        expected = "recent" if i >= len(frames) - 4 else "older"
        if (f["region"] != expected or f["source_frame"] < 0
                or not 0 <= f["start_s"] <= case["query_time_s"] + 1e-6):
            raise ValueError(f"Invalid or post-query frame in {case['case_id']}")
    return frames


def reader_for(case, media_root):
    from decord import VideoReader, cpu

    path = media_root / f"sample_{case['sample_id']}_real.mp4"
    if not path.is_file():
        raise FileNotFoundError(f"Missing original clip: {path}")
    if sha(path) != case["video_sha256"]:
        raise ValueError(f"Clip checksum differs from the confirmation run: {path.name}")
    return VideoReader(str(path), ctx=cpu(0), num_threads=1)


def verify_timestamps(reader, frames):
    origin = float(reader.get_frame_timestamp(0)[0])
    indices = [f["source_frame"] for f in frames]
    if max(indices) >= len(reader):
        raise ValueError("Saved frame index outside decoded clip")
    times = reader.get_frame_timestamp(indices).tolist()
    if any(abs(t[0] - origin - f["start_s"]) > 0.001 for t, f in zip(times, frames, strict=True)):
        raise ValueError("Decoder timestamps differ from the audited sampling record")


def render_sheet(reader, frames, title, columns, width, height, output):
    from PIL import Image, ImageDraw, ImageFont

    margin, caption = 10, 27
    vertical = (len(frames) + columns - 1) // columns
    canvas = Image.new("RGB", (columns*(width + margin)+margin, 48+vertical*(height+caption+margin)), "#fafafa")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = title_font = ImageFont.load_default()
    draw.text((margin, 12), title, anchor="lt", fill="#182334", font=title_font)
    for i, frame in enumerate(frames):
        x = margin + (i % columns)*(width+margin)
        y = 42 + (i // columns)*(height+caption+margin)
        image = Image.fromarray(reader[frame["source_frame"]].asnumpy())
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        draw.rectangle((x, y, x+width, y+height), fill="#222222")
        canvas.paste(image, (x+(width-image.width)//2, y+(height-image.height)//2))
        text = f"Sample {frame['sampled_index']}    {frame['start_s']:.2f} s"
        draw.text((x, y+height+5), text, anchor="lt", fill="#182334", font=font)
    canvas.save(output, quality=92)


def render(case, reader, output):
    frames = validated_frames(case)
    verify_timestamps(reader, frames)
    folder = output / "sheets" / case["case_id"]
    folder.mkdir(parents=True, exist_ok=True)
    recent = [f for f in frames if f["region"] == "recent"]
    older = [f for f in frames if f["region"] == "older"]
    title = f"{case['case_id']}  |  Question at {case['query_time_s']:.2f} s"
    render_sheet(reader, recent, title + "  |  Recent frames", 2, 560, 315, folder/"recent.jpg")
    for i in range(0, len(older), 8):
        render_sheet(reader, older[i:i+8], title + f"  |  Older frames, page {i//8+1}",
                     4, 295, 166, folder/f"older_{i//8+1}.jpg")


def blind_index(cases, output):
    lines = [
        '<!doctype html><meta charset="utf-8"><title>HERMES visual evidence review</title>',
        "<h1>HERMES visual evidence review</h1>",
        "<p>Review the question and frames before consulting the separate answer key. "
        "Sample numbers refer to the original 0.5 FPS grid.</p>",
    ]
    for case in cases:
        key = case["case_id"]
        folder = output / "sheets" / key
        lines.extend([f'<h2 id="{key}">{key} · {case["query_time_s"]:.2f} s</h2>',
                      f"<p>{html.escape(case['question'])}</p><ol type='A'>"])
        lines.extend(f"<li>{html.escape(choice)}</li>" for choice in case["choices"])
        lines.append("</ol>")
        for file in [folder/"recent.jpg", *sorted(folder.glob("older_*.jpg"))]:
            path = file.relative_to(output).as_posix()
            lines.append(f'<img src="{path}" alt="{file.stem}" style="display:block;max-width:100%;margin:12px 0">')
    (output / "index.html").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(reference, media, output):
    cases = reference["cases"]
    outcome = reference["outcomes"]
    if len(cases) != 12 or {c["case_id"] for c in cases} != {r["case_id"] for r in outcome}:
        raise ValueError("Unexpected discordance list")
    output.mkdir(parents=True, exist_ok=True)
    by_clip = defaultdict(list)
    for case in cases:
        validated_frames(case)
        by_clip[case["sample_id"]].append(case)
    for sid, group in by_clip.items():
        if len({c["video_sha256"] for c in group}) != 1:
            raise ValueError(f"Conflicting media checksums for sample {sid}")
        reader = reader_for(group[0], media)
        for case in group:
            render(case, reader, output)
            print(f"VISUAL_CASE_READY {case['case_id']}", flush=True)
        del reader
    blind_index(cases, output)
    # Save existing judgements across notebook reruns.
    notes = output / "review_notes.csv"
    if not notes.exists():
        write_csv(notes, NOTES_FIELDS, [{"case_id": c["case_id"]} for c in cases])
    else:
        with notes.open(encoding="utf-8-sig", newline="") as handle:
            saved = csv.DictReader(handle)
            if tuple(saved.fieldnames or ()) != NOTES_FIELDS or [r["case_id"] for r in saved] != [c["case_id"] for c in cases]:
                raise ValueError("Existing review notes do not match the frozen cases")
    key = output / "after_review" / "outcome_key.csv"
    key.parent.mkdir(exist_ok=True)
    write_csv(key, tuple(outcome[0]), outcome)
    write_csv(output / "frame_slots.csv", ("case_id", "sampled_index", "source_frame", "start_s", "region"), [
        dict(case_id=c["case_id"], **f) for c in cases for f in c["frames"]
    ])
    (output / "review_reference.json").write_text(json.dumps(dict(archive_sha256=reference["archive_sha256"],
        cases=cases, support_regions=REGIONS), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def inspect_frame(reference, media, output, case_id, sampled_index):
    from PIL import Image

    case = next((c for c in reference["cases"] if c["case_id"] == case_id), None)
    if case is None:
        raise ValueError("Unknown case ID")
    frame = next((f for f in validated_frames(case) if f["sampled_index"] == sampled_index), None)
    if frame is None:
        raise ValueError("Requested sample was outside the model input")
    reader = reader_for(case, media)
    verify_timestamps(reader, [frame])
    folder = output / "originals"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{case_id}_sample_{sampled_index:04d}.png"
    Image.fromarray(reader[frame["source_frame"]].asnumpy()).save(path)
    print(path)


def package(output, dest):
    files = [output/n for n in ("index.html", "review_notes.csv", "frame_slots.csv", "review_reference.json",
                                  "after_review/outcome_key.csv")]
    files += sorted((output / "sheets").glob("*/*.jpg"))
    files += sorted((output / "originals").glob("*.png"))
    temp = dest.with_suffix(".tmp")
    with ZipFile(temp, "w", compression=ZIP_DEFLATED) as zipped:
        for path in files:
            zipped.write(path, path.relative_to(output).as_posix())
    temp.replace(dest)
    print(f"VISUAL_REVIEW_ZIP {dest} SHA256={sha(dest)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ("reference", "media", "output"):
        parser.add_argument("--"+n, type=Path, required=True)
    parser.add_argument("--inspect", nargs=2, metavar=("CASE", "SAMPLE"))
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    if args.inspect:
        inspect_frame(reference, args.media, args.output, args.inspect[0], int(args.inspect[1]))
    elif args.zip:
        package(args.output, args.zip)
    else:
        generate(reference, args.media, args.output)


if __name__ == "__main__":
    main()
