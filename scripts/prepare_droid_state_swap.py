"""Freeze and preview the symmetric DROID early-state frame swap."""
from io import BytesIO
import json
from zipfile import ZipFile

from PIL import Image, ImageDraw

from validate_droid_natural_history_evidence import FRAMES, MANIFEST, verify_frame_pack
from validate_droid_terminal_evidence import ARCHIVE
from validate_twinpast_diagnostic_evidence import ROOT, digest, verify_inventory

DEST = ROOT / "data/twinpast/droid_state_swap_v1.json"
REVIEW = FRAMES.parent / "state_swap_review"
SLOTS = (0, 1)


def prepare():
    manifest = json.loads(MANIFEST.read_bytes())
    assert verify_frame_pack(FRAMES, manifest) == 114
    prior = verify_inventory(ARCHIVE)
    assert prior["provenance/droid_natural_manifest.json"] == MANIFEST.read_bytes()
    REVIEW.mkdir(exist_ok=True)
    cases = []
    with ZipFile(FRAMES) as frames:
        for case in manifest["cases"]:
            pair = case["pair_id"]
            branches = case["branches"]
            assert branches["a"]["gold"] != branches["b"]["gold"]
            results = {}
            sheet = Image.new("RGB", (1260, 578), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((4, 5), case["question"], fill="black")
            for side, source in (("a", "b"), ("b", "a")):
                recipient, donor = branches[side], branches[source]
                original, other = (r["frames"]["annotated"] for r in (recipient, donor))
                assert len(original) == len(set(original)) == len(other) == len(set(other)) == 6
                for branch in (recipient, donor):
                    initial = branch["initial_interval"]
                    indices = branch["source_indices"]["annotated"]
                    assert all(initial[0] <= indices[k] <= initial[1] for k in SLOTS)
                changed = [other[k] if k in SLOTS else name for k, name in enumerate(original)]
                assert [i for i, (a, b) in enumerate(zip(original, changed, strict=True)) if a != b] == list(SLOTS)
                assert set(changed).isdisjoint(case["recent_frames"])
                results[side] = dict(donor_branch=source, original_history=original, swapped_history=changed)
                for arm, history in enumerate((original, changed)):
                    y = 24+(int(side == "b")*2+arm)*137
                    draw.text((4, y), f"{side.upper()} {'swap' if arm else 'original'}", fill="black")
                    for i, name in enumerate(history):
                        with Image.open(BytesIO(frames.read(name))) as image:
                            sheet.paste(image.resize((206, 116)), (i*210, y+16))
            sheet.save(REVIEW / f"{pair}.jpg", quality=89)
            cases.append(dict(pair_id=pair, replaced_history_slots=SLOTS, branches=results,
                              shared_recent_frames=case["recent_frames"]))
    frozen = dict(natural_history_manifest_sha256=digest(MANIFEST.read_bytes()),
                  terminal_context_archive_sha256=digest(ARCHIVE.read_bytes()),
                  frame_zip_sha256=manifest["frame_zip_sha256"], cases=cases)
    DEST.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Froze {len(cases)} paired swaps. Review sheets are in {REVIEW}")


if __name__ == "__main__":
    prepare()
