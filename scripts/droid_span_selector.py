"""Select a three-frame window from ten frames using endpoint contrast."""
import re

import numpy as np

RECENCY = (7, 8, 9)
WINDOWS = tuple(tuple(range(i, i + 3)) for i in range(8))


def temporal_endpoint(question):
    """Recognize the two temporal references in the frozen question templates."""
    text = question.lower()
    first = bool(re.search(r"\bfirst view\b", text))
    last = bool(re.search(r"\blast view\b", text))
    if first == last:
        raise ValueError("Question needs a single first-view or last-view reference")
    return 0 if first else 9


def endpoint_window(question):
    return WINDOWS[0] if temporal_endpoint(question) == 0 else RECENCY


def select_window(question, frame_vectors):
    """Score contiguous windows by contrast to the opposite temporal endpoint."""
    features = np.asarray(frame_vectors, dtype=np.float64)
    if features.ndim != 2 or len(features) != 10 or not np.isfinite(features).all():
        raise ValueError("Expected ten finite frame feature vectors")
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    unit = features / np.maximum(norm, 1e-12)
    anchor = temporal_endpoint(question)
    # Endpoints determine the positive and negative anchors from the same video.
    # Search the corresponding half to limit drift toward a different event.
    eligible = WINDOWS[:3] if anchor == 0 else WINDOWS[-3:]
    contrast = unit @ (unit[anchor] - unit[9 - anchor])
    scores = [float(contrast[list(window)].mean()) for window in eligible]
    winner = max(range(len(eligible)), key=lambda i: (round(scores[i], 8), -abs(eligible[i][1] - anchor)))
    return eligible[winner], [{"slots": window, "contrast": score} for window, score in zip(eligible, scores)]
