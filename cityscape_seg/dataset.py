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


class CityscapesSegDataset(Dataset):
    """Loads Cityscapes images + labels into memory as resized PIL images.

    Augmentation transforms are applied lazily in ``__getitem__`` so that each
    epoch sees different random variations of the training data.

    If ``max_samples`` is set, only that many randomly chosen pairs are loaded
    (controlled by ``seed`` for reproducibility), saving both time and memory.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: tuple[int, int] = (256, 512),
        transform=None,
        max_samples: int | None = None,
        seed: int = 42,
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
