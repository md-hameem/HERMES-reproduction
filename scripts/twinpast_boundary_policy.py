"""Shared annotation-blind TwinPast boundary ranking and sampling policy."""
from itertools import combinations

import numpy as np
from PIL import Image

FPS = 4
THRESHOLD = 0.20
UNIFORM_SLOTS = (3, 9)


def screen_candidates(reader, starts, candidates, first_recent):
    windows = sorted((tuple(row["slots"]), row["relevance"]) for row in candidates)
    grids = {key: np.arange(key[0]*FPS, key[1]*FPS+1) / FPS for key, _ in windows}
    indices = {key: np.searchsorted(starts, grid, side="right") - 1 for key, grid in grids.items()}
    assert all(len(row) == 13 and min(row) >= 0 and np.all(np.diff(row) > 0) for row in indices.values())
    unique = sorted({int(i) for row in indices.values() for i in row})
    assert unique[-1] < first_recent
    tiny = {}
    for offset in range(0, len(unique), 12):
        group = unique[offset:offset+12]
        frames = reader.get_batch(group).asnumpy()
        for i, rgb in zip(group, frames, strict=True):
            tiny[i] = np.asarray(Image.fromarray(rgb).resize((32, 18), Image.Resampling.BILINEAR), dtype=np.float32)
    checked = []
    for key, score in windows:
        images = np.stack([tiny[i] for i in indices[key]])
        delta = np.abs(np.diff(images, axis=0)).mean(axis=(1, 2, 3)) / 255
        cut = int(delta.argmax())
        checked.append(dict(window=key, relevance=score, indices=indices[key].tolist(),
                            peak_delta=float(delta[cut]), pre_cut_slot=cut if delta[cut] >= THRESHOLD else None))
    return checked, len(unique)


def select(candidates):
    groups = [group for group in combinations(candidates, 3)
              if all(left["window"][1] < right["window"][0] for left, right in zip(group, group[1:]))]
    relevance = lambda group: sum(w["relevance"] for w in group)
    original = max(groups, key=relevance)
    with_cut = [group for group in groups if any(w["pre_cut_slot"] is not None for w in group)]
    return original, max(with_cut, key=relevance) if with_cut else original


def pick(windows, preserve):
    qualifying = [w for w in windows if w["pre_cut_slot"] is not None]
    anchor = max(qualifying, key=lambda w: w["peak_delta"]) if preserve and qualifying else None
    chosen = []
    for window in windows:
        slots = UNIFORM_SLOTS
        if window is anchor:
            slot = window["pre_cut_slot"]
            farther = max(UNIFORM_SLOTS, key=lambda ordinary: abs(ordinary-slot))
            slots = sorted((slot, farther))
        chosen.extend(window["indices"][i] for i in slots)
    assert len(chosen) == 6 and chosen == sorted(set(chosen))
    return chosen
