"""Utility helpers: inverse normalization, label colouring, etc."""

from __future__ import annotations

import numpy as np
import torchvision.transforms as T

from .labels import CLASS_COLORS

inv_normalize = T.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)


def label_to_color(lbl_np: np.ndarray) -> np.ndarray:
    """Convert a HxW label map to an HxWx3 RGB image."""
    h, w = lbl_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(CLASS_COLORS):
        rgb[lbl_np == cls_id] = color
    return rgb
