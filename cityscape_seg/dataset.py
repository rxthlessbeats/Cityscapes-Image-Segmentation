"""Cityscapes segmentation dataset with lazy augmentation."""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.segmentation import find_boundaries
from torch.utils.data import Dataset
from tqdm import tqdm

from .labels import build_label_remap


def _subsample_pairs_preferring_classes(
    pairs: list[tuple[str, str]],
    max_samples: int,
    seed: int,
    img_size: tuple[int, int],
    label_remap: np.ndarray,
    prefer_classes: list[int],
    min_rare_fraction: float = 0.0,
) -> list[tuple[str, str]]:
    """Prefer images whose remapped label at ``img_size`` has enough pixels in ``prefer_classes``; pad to ``max_samples``."""
    rng = np.random.default_rng(seed)
    prefer_arr = np.asarray(prefer_classes, dtype=np.int64)
    eligible_idx: list[int] = []
    for i, (_, lp) in enumerate(
        tqdm(pairs, desc="Scanning labels (subset selection)", unit="img", leave=False)
    ):
        lbl = Image.open(lp)
        lbl = TF.resize(lbl, img_size, interpolation=TF.InterpolationMode.NEAREST)
        arr = label_remap[np.array(lbl, dtype=np.uint8)]
        frac = float(np.isin(arr, prefer_arr).mean())
        if min_rare_fraction > 0:
            ok = frac > min_rare_fraction
        else:
            ok = frac > 0
        if ok:
            eligible_idx.append(i)

    n_pairs = len(pairs)
    if not eligible_idx:
        indices = rng.choice(n_pairs, size=max_samples, replace=False)
        return [pairs[j] for j in sorted(indices)]
    if len(eligible_idx) >= max_samples:
        pick = rng.choice(len(eligible_idx), size=max_samples, replace=False)
        return [pairs[eligible_idx[j]] for j in sorted(pick)]

    eligible_set = set(eligible_idx)
    ineligible_idx = [j for j in range(n_pairs) if j not in eligible_set]
    need = max_samples - len(eligible_idx)
    ineligible_arr = np.asarray(ineligible_idx, dtype=np.int64)
    if need <= len(ineligible_arr):
        extra = rng.choice(ineligible_arr, size=need, replace=False)
    else:
        extra = rng.choice(ineligible_arr, size=need, replace=True)
    indices = sorted(eligible_idx + list(extra))
    return [pairs[j] for j in indices]


class CityscapesSegDataset(Dataset):
    """Loads Cityscapes images + labels into memory as resized PIL images.

    Augmentation transforms are applied lazily in ``__getitem__`` so that each
    epoch sees different random variations of the training data.

    If ``max_samples`` is set, only that many pairs are loaded (controlled by ``seed``).
    With non-empty ``prefer_images_with_classes``, subsampling prefers images whose
    remapped labels at ``img_size`` contain enough pixels in those classes, then pads
    from the rest to reach ``max_samples``. If that list is empty or omitted, subsampling
    is uniform random (same as always picking ``max_samples`` indices at random).
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: tuple[int, int] = (256, 512),
        transform=None,
        max_samples: int | None = None,
        seed: int = 42,
        prefer_images_with_classes: list[int] | None = None,
        prefer_min_rare_fraction: float = 0.0,
    ) -> None:
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.transform = transform
        self.img_size = img_size
        self._label_remap = build_label_remap()

        image_paths = sorted(glob.glob(os.path.join(split_dir, "*_leftImg8bit.png")))
        label_paths = sorted(glob.glob(os.path.join(split_dir, "*_labelIds.png")))
        assert len(image_paths) == len(label_paths), (
            f"Mismatch: {len(image_paths)} images vs {len(label_paths)} labels"
        )

        pairs = list(zip(image_paths, label_paths))

        if max_samples is not None and max_samples < len(pairs):
            prefer = [] if prefer_images_with_classes is None else list(prefer_images_with_classes)
            if prefer:
                pairs = _subsample_pairs_preferring_classes(
                    pairs,
                    max_samples,
                    seed,
                    img_size,
                    self._label_remap,
                    prefer,
                    prefer_min_rare_fraction,
                )
            else:
                rng = np.random.default_rng(seed)
                indices = rng.choice(len(pairs), size=max_samples, replace=False)
                pairs = [pairs[i] for i in sorted(indices)]

        self.images: list[Image.Image] = []
        self.labels: list[Image.Image] = []
        for ip, lp in tqdm(pairs, desc=f"Loading [{split}]", unit="img"):
            img = Image.open(ip).convert("RGB")
            lbl = Image.open(lp)

            img = TF.resize(img, img_size, interpolation=TF.InterpolationMode.BILINEAR)
            lbl = TF.resize(lbl, img_size, interpolation=TF.InterpolationMode.NEAREST)

            self.images.append(img)
            self.labels.append(lbl)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img, lbl = self.images[idx], self.labels[idx]

        if self.transform is not None:
            img, lbl = self.transform(img, lbl)

        lbl_np = (
            self._label_remap[lbl]
            if isinstance(lbl, np.ndarray)
            else self._label_remap[np.array(lbl, dtype=np.uint8)]
        )
        lbl_t = torch.from_numpy(lbl_np).long()

        boundary = find_boundaries(lbl_np, mode="outer").astype(np.uint8)
        boundary += 1  # 1 = interior, 2 = boundary
        mask = torch.from_numpy(boundary).float()

        return img, lbl_t, mask
