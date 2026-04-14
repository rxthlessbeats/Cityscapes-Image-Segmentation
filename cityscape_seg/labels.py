"""Cityscapes label remapping, class names, and colours."""

from __future__ import annotations

import numpy as np

CLASS_NAMES: list[str] = [
    "road/drivable",   # 0
    "sidewalk",        # 1
    "human",           # 2
    "vehicle",         # 3
    "traffic object",  # 4
    "nature",          # 5  (vegetation, terrain, sky)
    "construction",    # 6  (building, wall, fence, guard rail, bridge, tunnel)
    "background",      # 7  (void classes)
]

CLASS_COLORS: list[tuple[int, int, int]] = [
    (128, 64, 128),   # road – purple
    (244, 35, 232),   # sidewalk – pink
    (220, 20, 60),    # human – crimson
    (0, 0, 142),      # vehicle – dark blue
    (250, 170, 30),   # traffic object – orange
    (107, 142, 35),   # nature – olive green
    (70, 70, 70),     # construction – dark gray
    (0, 0, 0),        # background – black
]

NUM_CLASSES = len(CLASS_NAMES)


def build_label_remap() -> np.ndarray:
    """Return a 256-entry uint8 lookup table mapping Cityscapes labelIds
    to the 8-class scheme.  Default (unmapped) entries go to class 7 (background).
    """
    remap = np.full(256, 7, dtype=np.uint8)

    # Class 0: road / drivable area
    for lid in [7, 9, 10]:
        remap[lid] = 0

    # Class 1: sidewalk
    remap[8] = 1

    # Class 2: human
    for lid in [24, 25]:
        remap[lid] = 2

    # Class 3: vehicle
    for lid in [26, 27, 28, 29, 30, 31, 32, 33]:
        remap[lid] = 3

    # Class 4: traffic object
    for lid in [17, 18, 19, 20]:
        remap[lid] = 4

    # Class 5: nature (vegetation, terrain, sky)
    for lid in [21, 22, 23]:
        remap[lid] = 5

    # Class 6: construction (building, wall, fence, guard rail, bridge, tunnel)
    for lid in [11, 12, 13, 14, 15, 16]:
        remap[lid] = 6

    return remap
