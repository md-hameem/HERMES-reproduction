"""Apply the frozen competing-state masks without changing other pixels."""
import numpy as np

MASK_RGB = (112, 112, 112)


def apply_mask(rgb, rectangle):
    x0, y0, x1, y1 = rectangle
    assert rgb.shape == (360, 640, 3) and rgb.dtype == np.uint8
    assert 0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 360
    result = rgb.copy()
    result[y0:y1, x0:x1] = MASK_RGB
    return result
