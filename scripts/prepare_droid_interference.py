"""Freeze equal-area target and non-target masks for both branches of each DROID pair."""
import json
from io import BytesIO
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

from droid_state_masks import MASK_RGB, apply_mask
from validate_droid_natural_history_evidence import ARCHIVE, FRAMES, MANIFEST, verify_frame_pack
from validate_twinpast_diagnostic_evidence import ROOT, digest, verify_inventory

DEST = ROOT / "data/twinpast/droid_state_interference_v1.json"
REVIEW = FRAMES.parent / "interference_mask_review"
SLOTS = (0, 1)
# Pixel rectangle at 640 x 360 and top-left position of its equal-area control.
RECTS = {
    "8b01107c59": ((141, 180, 320, 310), (429, 180)),
    "6d74030dc0": ((220, 12, 416, 286), (432, 12)),
    "87abc314ec": ((138, 130, 514, 260), (150, 0)),
    "29137e19c7": ((220, 139, 360, 252), (465, 139)),
    "c00374385e": ((170, 181, 416, 343), (70, 6)),
    "19bf4a67ee": ((144, 59, 373, 258), (402, 59)),
    "10bd743a70": ((122, 208, 441, 358), (5, 7)),
}


def prepare():
    manifest = json.loads(MANIFEST.read_bytes())
    verified_frames = verify_frame_pack(FRAMES, manifest)
    prior = verify_inventory(ARCHIVE)
    assert prior["provenance/droid_natural_manifest.json"] == MANIFEST.read_bytes()
    REVIEW.mkdir(exist_ok=True)
    cases = []
    with ZipFile(FRAMES) as pack:
        for case in manifest["cases"]:
            pair = case["pair_id"]
            target, (cx, cy) = RECTS[pair.rsplit("_", 1)[1]]
            x0, y0, x1, y1 = target
            control = (cx, cy, cx+x1-x0, cy+y1-y0)
            assert x1 <= cx or control[2] <= x0 or y1 <= cy or control[3] <= y0
            sheet = Image.new("RGB", (960, 832), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((4, 3), case["question"], fill="black")
            for i, label in enumerate(("Original", "Target mask", "Non-target mask")):
                draw.text((i*320+4, 22), label, fill="black")
            modified = {}
            for j, (side, branch) in enumerate(case["branches"].items()):
                modified[side] = {}
                early, terminal = branch["initial_interval"], branch["terminal_interval"]
                for k, slot in enumerate(SLOTS):
                    assert early[0] <= branch["source_indices"]["annotated"][slot] <= early[1] < terminal[0]
                    name = branch["frames"]["annotated"][slot]
                    with Image.open(BytesIO(pack.read(name))) as image:
                        rgb = np.array(image)
                    masked = {key: apply_mask(rgb, rect) for key, rect in
                              (("target", target), ("control", control))}
                    modified[side][name] = {key: digest(frame.tobytes()) for key, frame in masked.items()}
                    for i, frame in enumerate((rgb, masked["target"], masked["control"])):
                        picture = Image.fromarray(frame).resize((320, 180))
                        y = 43+(j*2+k)*195
                        sheet.paste(picture, (i*320, y))
                        draw.text((i*320+4, y+180), f"Branch {side.upper()}, history frame {slot+1}", fill="black")
            sheet.save(REVIEW / f"{pair}.jpg", quality=89)
            cases.append(dict(pair_id=pair, target=target, control=control, edited_slots=SLOTS,
                              edited_rgb_sha256=modified))
    result = dict(natural_history_archive_sha256=digest(ARCHIVE.read_bytes()),
                  natural_history_manifest_sha256=digest(MANIFEST.read_bytes()),
                  frames_verified=verified_frames, fill_rgb=MASK_RGB, cases=cases)
    DEST.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Froze {len(cases)} mask plans, reviewed at {REVIEW}")


if __name__ == "__main__":
    prepare()
