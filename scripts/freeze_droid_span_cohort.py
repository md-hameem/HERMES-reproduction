"""Freeze reviewed DROID span cases and build their companion frame package."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo

from droid_span_selector import WINDOWS
from freeze_droid_natural_history import encode, reader_for
from prepare_droid_span_cohort import ROOT, WORK, digest, exclusions

MANIFEST = ROOT / "data/twinpast/droid_span_cohort_v1.json"
FRAME_ZIP = "DROID_span_cohort_frames.zip"


def freeze(work, review_file, manifest):
    review = json.loads(review_file.read_bytes())
    screen = json.loads((work / "intake.json").read_bytes())
    intake = {r["pair_id"]: r for r in screen["intake"]}
    if {r["pair_id"] for r in review} != set(intake) or len(review) != len(intake):
        raise ValueError("Record exactly one disposition per acquired scene")
    excluded_episodes, excluded_scenes = exclusions()
    assets, cases = {}, []
    seen_episodes, seen_scenes, video_hashes = set(), set(), set()
    for decision in review:
        if not decision.get("camera"):
            if not decision.get("reason"):
                raise ValueError("Every exclusion needs a reason")
            continue
        pair = intake[decision["pair_id"]]
        scene = (pair["site"], pair["scene_id"])
        assert scene not in excluded_scenes | seen_scenes
        seen_scenes.add(scene)
        camera = decision["camera"]
        choices = {pair[k]["episode_id"]: pair[k] for k in ("open_episode", "close_episode")}
        branches = {}
        for side in "ab":
            grid = pair["grids"][camera][side]
            episode = choices[grid["source_episode"]]
            episode_id = episode["episode_id"]
            receipt = episode["cameras"][camera]
            assert episode_id not in excluded_episodes | seen_episodes and receipt["sha256"] not in video_hashes
            seen_episodes.add(episode_id)
            video_hashes.add(receipt["sha256"])
            reader = reader_for(work, episode, camera)
            indices = grid["source_frames"]
            assert indices == [i*(len(reader)-1)//9 for i in range(10)]
            stamps = reader.get_frame_timestamp(indices)[:, 0]
            timestamps = (stamps - stamps[0]).tolist()
            names = encode(reader, indices, f"{pair['pair_id']}_{side}", assets)
            labels = {}
            for task, allowed, anchor in (("first", range(5), 0), ("last", range(5, 10), 9)):
                label = decision["branches"][side][task]
                support = label["support_slots"]
                assert support == sorted(set(support)) and support and set(support).issubset(allowed)
                assert label["state"] in ("open", "closed")
                eligible = [w for w in WINDOWS if set(w).issubset(allowed)]
                oracle = max(eligible, key=lambda w: (len(set(w) & set(support)), -abs(w[1]-anchor)))
                labels[task] = dict(gold="A" if label["state"] == "open" else "B",
                                    support_slots=support, annotation_window=list(oracle))
            assert labels["first"]["gold"] != labels["last"]["gold"]
            branches[side] = dict(source_episode=episode_id, video_source=receipt,
                                  metadata_source=episode["metadata_source"], indices=indices,
                                  timestamps_s=timestamps, frames=list(names.values()), labels=labels)
        assert all(branches["a"]["labels"][t]["gold"] != branches["b"]["labels"][t]["gold"]
                   for t in ("first", "last"))
        cases.append(dict(pair_id=pair["pair_id"], source_site=pair["site"], scene_id=scene[1], camera=camera,
                          questions={t: f"In the {t} view of the {decision['target']}, was it open or closed?"
                                     for t in ("first", "last")}, branches=branches))
    if len(cases) < 6 or len({r["source_site"] for r in cases}) < 3:
        raise ValueError(f"Need six visually eligible scenes from at least three sites, got {len(cases)}")
    path = work / FRAME_ZIP
    with ZipFile(path, "w") as archive:
        for name, (raw, _) in sorted(assets.items()):
            archive.writestr(ZipInfo(name), raw)
    spec = dict(protocol="DROID automatic span selection v1", dataset="DROID raw 1.0.1",
                review_sha256=digest(review_file.read_bytes()), frame_zip_sha256=digest(path.read_bytes()),
                source_screen=dict(metadata_errors=len(screen["unavailable"]),
                                   candidate_combinations=screen["screened_candidate_pairs"],
                                   acquired_scenes=len(intake)),
                excluded_prior_episodes=sorted(excluded_episodes), excluded_prior_scenes=sorted(excluded_scenes),
                frame_receipts={name: dict(png_sha256=digest(raw), rgb_sha256=rgb)
                                for name, (raw, rgb) in sorted(assets.items())}, cases=cases)
    manifest.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Frozen {len(cases)} scenes with {len(assets)} frames in {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    freeze(args.work, args.review, args.manifest)
