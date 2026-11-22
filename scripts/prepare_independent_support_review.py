"""Prepare a blinded reviewer copy and validate returned visual-support labels."""
import argparse
import csv
import io
import json
import shutil
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "evidence/local/target_binding_2026-11-22"
PACKET = LOCAL / "HERMES_blind_support_review.zip"
HANDOFF = LOCAL / "HERMES_independent_reviewer_packet.zip"
RESPONSE = "independent_review.csv"
FIELDS = ("review_id", "support_label", "supported_option", "supporting_sample_indices", "temporal_validity", "notes")
LABELS = ("visible_support", "wrong_target_setting_or_time", "insufficient", "uncertain")
INSTRUCTIONS = """# Independent HERMES visual-support review

Extract the ZIP and open review.html. Enter your judgments in independent_review.csv.
The original visual_support_review.csv is a frame index. Keep its timestamps and
sample IDs available while reviewing. Each row in the response represents one window.

Use these exact support labels:
- visible_support: the window visibly supports a specific listed option at the question time.
- wrong_target_setting_or_time: a similar cue is visible but refers to another subject,
  setting, or time. For current-state questions, an earlier scene may be stale.
- insufficient: the visible content does not support a listed option.
- uncertain: the stills are ambiguous, too small, motion-dependent, or require audio.

For visible_support, enter the option letter and the printed sampled indices,
separated by spaces (for example, 72 73). Set temporal_validity to yes.
For other labels leave the option blank. Set temporal_validity to no or uncertain
if you can judge it, and leave it blank if it does not apply. Record ambiguous
targets, suspected benchmark wording issues and any need for original footage in notes.

Use only evidence available by the stated question time. If an event happens
later or the stills only hint at a likely future answer, mark it uncertain.
Return independent_review.csv before comparing it with any other review.
"""


def blank_rows(packet):
    with ZipFile(packet) as archive:
        data = archive.read("visual_support_review.csv").decode("utf-8")
    return list(csv.DictReader(io.StringIO(data)))


def prepare(packet, output):
    if output.resolve() == packet.resolve():
        raise ValueError("Reviewer copy must have a different path from the original packet")
    rows = blank_rows(packet)
    response = io.StringIO(newline="")
    writer = csv.DictWriter(response, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows({"review_id": row["review_id"]} for row in rows)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        dest = Path(tmp) / "reviewer.zip"
        shutil.copyfile(packet, dest)
        with ZipFile(dest, "a", compression=ZIP_DEFLATED) as archive:
            archive.writestr(RESPONSE, response.getvalue())
            archive.writestr("START_HERE.md", INSTRUCTIONS)
        dest.replace(output)
    print(f"Prepared {len(rows)} blinded response rows: {output}")


def validate(packet, response):
    expected = {row["review_id"]: row for row in blank_rows(packet)}
    with response.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(expected) or {r.get("review_id") for r in rows} != set(expected):
        raise ValueError("Response must contain each review ID once")
    for row in rows:
        rid = row["review_id"]
        label, option, validity = (row[key].strip() for key in ("support_label", "supported_option", "temporal_validity"))
        indices = [int(s) for s in row["supporting_sample_indices"].split()]
        if len(indices) != len(set(indices)) or not set(indices) <= set(json.loads(expected[rid]["sampled_indices"])):
            raise ValueError(f"{rid}: use the printed sampled indices from this window")
        if label not in LABELS or validity not in {"yes", "no", "uncertain", ""}:
            raise ValueError(f"{rid}: invalid label or temporal-validity entry")
        if label == "visible_support":
            options = json.loads(expected[rid]["options"])
            if option not in list("ABCDEFGH"[:len(options)]) or not indices or validity != "yes":
                raise ValueError(f"{rid}: visible support requires an option, frame indices, and temporal validity")
        elif option:
            raise ValueError(f"{rid}: only visible support receives an option letter")
    print(f"Validated review labels for all {len(rows)} windows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=PACKET)
    parser.add_argument("--output", type=Path, default=HANDOFF)
    parser.add_argument("--validate", type=Path, help="Validate a completed independent review")
    args = parser.parse_args()
    if args.validate:
        validate(args.packet, args.validate)
    else:
        prepare(args.packet, args.output)
