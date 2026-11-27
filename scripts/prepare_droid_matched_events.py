"""Nominate and preview fresh DROID episodes with a common action in different order."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import itertools
import json
from pathlib import Path
import re

from prepare_droid_interior import WORK as INTERIOR, prior_sources, render
from prepare_droid_natural_history import metadata
from prepare_droid_span_cohort import SOURCE_ERRORS, WORK as SPAN
from prepare_droid_state_control import INDEX, PREFIX, WORK as CONTROL, acquire
from validate_twinpast_diagnostic_evidence import ROOT, digest

WORK = ROOT / "evidence/local/droid-matched-events"
STATE = {"open": "open", "close": "close", "shut": "close"}
EVENT = {"pick": "pickup", "pick up": "pickup", "grab": "pickup", "lift": "pickup", "take": "pickup",
         "remove": "pickup", "retrieve": "pickup", "put": "put", "place": "put", "set": "put", "move": "move"}
VERBS = re.compile(r"\b(pick up|pick|grab|lift|take|remove|retrieve|put|place|set|move|open|close|shut)\b", re.I)
PANEL = re.compile(r"\b(drawer|cabinet|cupboard|door|lid|oven|microwave|fryer|kettle|flask)\b", re.I)
STOP = set("a an the from in into it on and then to of up or this that at by out some one fully back".split())
GENERIC = set("object objects item items thing things cabinet drawer door cupboard oven microwave lid "
              "white black red green blue pink yellow silver brown orange gray grey left right top atop".split())


def indexed():
    inputs = {}
    for name, sha in INDEX.items():
        raw = (CONTROL / name).read_bytes()
        if digest(raw) != sha:
            raise ValueError(f"Unexpected DROID index bytes for {name}")
        inputs[name] = json.loads(raw)
    prefixes = defaultdict(list)
    for row in inputs["manifest.json"]:
        path = row["path"].split("r2d2_full_raw/", 1)[-1].split("/")
        if len(path) != 5 or path[1] != "success":
            continue
        stamp = re.search(r"(\d{1,2}):(\d{2}):(\d{2})", path[-2])
        if stamp:
            h, m, s = stamp.groups()
            prefixes[(path[0], f"{path[2]}-{int(h):02}h-{m}m-{s}s")].append(PREFIX + "/".join(path[:-1]) + "/")
    for uid, annotation in inputs["aggregated-annotations-030724.json"].items():
        site, collector, stamp = uid.split("+")
        found = prefixes[(site, stamp)]
        if len(found) == 1:
            yield uid, site, collector, stamp, found[0], annotation.values()


def parse_instruction(instruction):
    text = instruction.strip().lower()
    if not PANEL.search(text):
        return
    words = [m for m in VERBS.finditer(text) if not re.search(r"\b(?:the|a|an)\s+$", text[:m.start()])]
    state, anchor = [], []
    for i, word in enumerate(words):
        following = text[word.end():words[i+1].start() if i + 1 < len(words) else len(text)]
        verb = word.group()
        if verb in STATE:
            panels = PANEL.findall(following)
            for panel in panels:
                state.append((word.start(), STATE[verb], "cabinet" if panel == "cupboard" else panel))
        else:
            direct = re.split(r"[,;.\n]|\b(?:from|into|inside|onto|on|in|and|then)\b", following, maxsplit=1)[0]
            nouns = sorted(set(re.findall(r"[a-z]+", direct)) - STOP - GENERIC)
            if nouns:
                anchor.append((word.start(), EVENT[verb], nouns))
    for state_pos, verb, panel in state:
        for anchor_pos, action, nouns in anchor:
            for noun in nouns:
                yield (verb, panel, action, noun), state_pos < anchor_pos, tuple(nouns)


def prior_ids():
    episodes, scenes = prior_sources()
    previous = json.loads((INTERIOR / "intake.json").read_bytes())["intake"]
    episodes.update(r["episode_id"] for r in previous)
    scenes.update((r["site"], r["scene_id"]) for r in previous)
    return episodes, scenes


def nominate(excluded_episodes, days, opposing_actions=False):
    groups = defaultdict(lambda: defaultdict(dict))
    for uid, site, collector, stamp, prefix, annotations in indexed():
        if uid in excluded_episodes:
            continue
        for instruction in filter(None, annotations):
            for key, before, nouns in parse_instruction(instruction):
                state_verb, panel, action, noun = key
                row = dict(episode_id=uid, site=site, prefix=prefix, instruction=instruction.strip(),
                           stamp=stamp, event_tokens=list(nouns), state_action=state_verb,
                           state_action_before_event=before)
                if opposing_actions:
                    orientation = (state_verb == "open") == before
                    group = (site, collector, panel, action, noun)
                else:
                    orientation = before
                    group = (site, collector, *key)
                groups[group][uid].setdefault(orientation, row)
    ranked = {}
    for group, entries in groups.items():
        # Drop contradictory alternate annotations of the same episode for this key.
        rows = {order: [v[order] for v in entries.values() if list(v) == [order]] for order in (True, False)}
        for a, b in itertools.product(rows[True], rows[False]):
            a_time, b_time = (datetime.strptime(r["stamp"], "%Y-%m-%d-%Hh-%Mm-%Ss") for r in (a, b))
            gap = abs((a_time-b_time).total_seconds())
            if gap > days * 86400:
                continue
            tokens_a, tokens_b = (set(r["event_tokens"]) for r in (a, b))
            score = len(tokens_a & tokens_b)/len(tokens_a | tokens_b)
            key = (a["episode_id"], b["episode_id"])
            candidate = (score, -gap, group[2:], a, b)
            if key not in ranked or candidate[:2] > ranked[key][:2]:
                ranked[key] = candidate
    return sorted(ranked.values(), key=lambda c: (-c[0], -c[1], c[3]["episode_id"], c[4]["episode_id"]))


def prepare(work, max_pairs, days, opposing_actions):
    work.mkdir(parents=True, exist_ok=True)
    old_episodes, old_scenes = prior_ids()
    ranked = nominate(old_episodes, days, opposing_actions)
    shortlist = ranked[:max(80, 6 * max_pairs)]
    episodes = {r["episode_id"]: r for _, _, _, a, b in shortlist for r in (a, b)}
    print(f"DROID_MATCHED_NOMINATED ranked={len(ranked)} metadata={len(episodes)}", flush=True)
    cache = SPAN / "metadata"

    def lookup(row):
        path = cache / (row["episode_id"] + ".json")
        return json.loads(path.read_bytes()) if path.is_file() else metadata(row)

    meta, failed = {}, []
    with ThreadPoolExecutor(max_workers=14) as pool:
        jobs = {uid: pool.submit(lookup, row) for uid, row in episodes.items()}
        for uid, job in jobs.items():
            try:
                row = job.result()
                if row["uuid"] != uid or row["lab"] != episodes[uid]["site"]:
                    raise ValueError("Source metadata identity mismatch")
                meta[uid] = row
            except SOURCE_ERRORS as exc:
                failed.append(dict(episode=uid, error=type(exc).__name__))
    selected, filter_counts = [], dict(unavailable=0, different_scene=0, prior_scene=0, no_common_camera=0)
    used = set()
    for _, _, key, a, b in shortlist:
        ids = (a["episode_id"], b["episode_id"])
        if any(uid not in meta for uid in ids):
            filter_counts["unavailable"] += 1
            continue
        x, y = (meta[uid] for uid in ids)
        scene = (a["site"], x["scene_id"])
        if x["scene_id"] != y["scene_id"] or not (x["success"] and y["success"]):
            filter_counts["different_scene"] += 1
            continue
        if scene in old_scenes:
            filter_counts["prior_scene"] += 1
            continue
        cameras = [k for k in ("ext1", "ext2") if x[f"{k}_cam_serial"] == y[f"{k}_cam_serial"]]
        if not cameras:
            filter_counts["no_common_camera"] += 1
            continue
        # Keep the first two nominations per scene as alternatives for visual review.
        if sum(item[0] == scene for item in selected) >= 2 or any(uid in used for uid in ids):
            continue
        used.update(ids)
        selected.append((scene, key, cameras, a, b))
        if len(selected) == max_pairs:
            break
    pairs = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {r["episode_id"]: pool.submit(acquire, r, work) for _, _, _, a, b in selected for r in (a, b)}
        for scene, key, cameras, a, b in selected:
            ids = [r["episode_id"] for r in (a, b)]
            pair_id = "matched_event_" + digest("|".join(ids).encode())[:10]
            try:
                results = [jobs[uid].result() for uid in ids]
                assert all((r["site"], r["scene_id"]) == scene for r in results)
                for row in results:
                    row["grids"] = render(work, row)
                pairs.append(dict(id=pair_id, scene=scene, nomination=key, cameras=cameras, episodes=results))
                print(f"DROID_MATCHED_PAIR_READY {pair_id} {scene}", flush=True)
            except SOURCE_ERRORS as exc:
                failed.append(dict(pair=ids, error=type(exc).__name__))
    stats = dict(index_digests=INDEX, max_days_between_episodes=days,
                 nominate_by_implied_state=opposing_actions,
                 excluded_episodes=sorted(old_episodes), excluded_scenes=sorted(old_scenes),
                 ranked_pairs=len(ranked), shortlisted_pairs=len(shortlist), metadata_requests=len(episodes),
                 filter_counts=filter_counts, metadata_matches=len(selected), failures=failed, pairs=pairs)
    (work / "intake.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--max-pairs", type=int, default=12)
    parser.add_argument("--max-days", type=int, default=31)
    parser.add_argument("--opposing-actions", action="store_true")
    args = parser.parse_args()
    if min(args.max_pairs, args.max_days) < 1:
        parser.error("--max-pairs and --max-days must be positive")
    prepare(args.work, args.max_pairs, args.max_days, args.opposing_actions)
