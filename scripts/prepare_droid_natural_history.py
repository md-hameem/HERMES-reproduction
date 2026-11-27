"""Find fresh DROID open/close episodes in the same recorded scene."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
from urllib.parse import quote

import requests

from prepare_droid_state_control import BUCKET, WORK as CONTROL, acquire, candidates, get
from validate_twinpast_diagnostic_evidence import ROOT, digest

WORK = ROOT / "evidence/local/droid-natural-history"
TIME_FORMAT = "%Y-%m-%d-%Hh-%Mm-%Ss"


def nearby_opposites(rows):
    sessions = {}
    for row in rows:
        lab, collector, timestamp = row["episode_id"].split("+")
        key = (lab, collector, timestamp[:10], row["target"])
        sessions.setdefault(key, {"open": [], "close": []})[row["action"]].append(row)
    for key, session in sessions.items():
        possibilities = []
        for a in session["open"]:
            start_a = datetime.strptime(a["episode_id"].rsplit("+", 1)[1], TIME_FORMAT)
            for b in session["close"]:
                start_b = datetime.strptime(b["episode_id"].rsplit("+", 1)[1], TIME_FORMAT)
                delta = abs((start_a-start_b).total_seconds())
                if delta <= 1200:
                    possibilities.append((delta, a["episode_id"], b["episode_id"], a, b))
        for _, _, _, a, b in sorted(possibilities)[:3]:
            yield key, a, b


def metadata(row):
    path = f"{row['prefix']}metadata_{row['episode_id']}.json"
    result = get(BUCKET + quote(path, safe="/")).json()
    assert result["uuid"] == row["episode_id"] and result["lab"] == row["site"]
    return result


def discover(limit):
    known = json.loads((CONTROL / "intake.json").read_bytes())
    seen_episodes = {row["episode_id"] for row in known}
    seen_scenes = {(row["site"], row["scene_id"]) for row in known}
    combos = list(nearby_opposites(row for row in candidates(CONTROL, None) if row["episode_id"] not in seen_episodes))
    rows = {row["episode_id"]: row for _, a, b in combos for row in (a, b)}
    print(f"DROID_NATURAL_METADATA episodes={len(rows)} candidate_pairs={len(combos)}", flush=True)
    with ThreadPoolExecutor(max_workers=10) as pool:
        tasks = {key: pool.submit(metadata, row) for key, row in rows.items()}
        values = {}
        for key, job in tasks.items():
            try:
                values[key] = job.result()
            except (requests.RequestException, KeyError, ValueError):
                continue
    selected = []
    for site in sorted({a["site"] for _, a, _ in combos}):
        accepted = 0
        for key, a, b in combos:
            if key[0] != site or a["episode_id"] not in values or b["episode_id"] not in values:
                continue
            aa, bb = (values[row["episode_id"]] for row in (a, b))
            scene = (site, aa["scene_id"])
            views = [cam for cam in ("ext1", "ext2") if aa[f"{cam}_cam_serial"] == bb[f"{cam}_cam_serial"]]
            if (not aa["success"] or not bb["success"] or aa["scene_id"] != bb["scene_id"]
                    or scene in seen_scenes or not views):
                continue
            selected.append((a, b, scene[1], views))
            seen_scenes.add(scene)
            accepted += 1
            if accepted == limit:
                break
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-site", type=int, default=2)
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    candidates_found = discover(args.per_site)
    acquired = []
    episodes = {row["episode_id"]: row for a, b, _, _ in candidates_found for row in (a, b)}
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {key: pool.submit(acquire, row, args.work) for key, row in episodes.items()}
        for a, b, scene, views in candidates_found:
            try:
                first, second = (jobs[row["episode_id"]].result() for row in (a, b))
            except (requests.RequestException, KeyError, ValueError, RuntimeError) as exc:
                print("DROID_NATURAL_PAIR_UNAVAILABLE", a["episode_id"], type(exc).__name__, flush=True)
                continue
            identifier = "droid_natural_" + digest(f"{a['episode_id']}|{b['episode_id']}".encode())[:10]
            acquired.append(dict(pair_id=identifier, site=a["site"], scene_id=scene, matched_cameras=views,
                                 open_episode=first, close_episode=second))
            print("DROID_NATURAL_PAIR_READY", identifier, a["site"], scene, flush=True)
    (args.work / "intake.json").write_text(json.dumps(acquired, indent=2) + "\n", encoding="utf-8", newline="\n")
    if not acquired:
        raise RuntimeError("No source-separated matched pairs downloaded")


if __name__ == "__main__":
    main()
