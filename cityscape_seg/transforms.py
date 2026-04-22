"""Paired image-label augmentation transforms for semantic segmentation."""

from __future__ import annotations

import random

import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from .config import TrainConfig
from .labels import build_label_remap


class PairedCompose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class PairedRandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = TF.hflip(image)
            target = TF.hflip(target)
        return image, target


class PairedColorJitter:
    """Applies color jitter to the image only (labels are unchanged)."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0) -> None:
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, target):
        return self.jitter(image), target


class PairedRandomResizedCrop:
    """Random crop + resize. With non-empty ``prefer_classes`` and ``num_samples`` > 1,
    try multiple crops and keep the one with the largest fraction of label pixels in those classes.
    """

    def __init__(
        self,
        size,
        scale=(0.5, 1.0),
        ratio=(0.75, 1.33),
        label_remap: np.ndarray | None = None,
        prefer_classes: tuple[int, ...] = (),
        num_samples: int = 1,
    ) -> None:
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self._label_remap = label_remap if label_remap is not None else build_label_remap()
        self.prefer_arr = np.asarray(prefer_classes, dtype=np.int64)
        self.num_samples = num_samples

    def __call__(self, image, target):
        best_score = -1.0
        best_box = (0, 0, image.height, image.width)
        for _ in range(self.num_samples):
            i, j, h, w = T.RandomResizedCrop.get_params(image, self.scale, self.ratio)
            patch = np.array(TF.crop(target, i, j, h, w), dtype=np.uint8)
            mapped = self._label_remap[patch]
            score = float(np.isin(mapped, self.prefer_arr).mean()) if self.prefer_arr.size else 0.0
            if score > best_score:
                best_score = score
                best_box = (i, j, h, w)
        i, j, h, w = best_box
        image = TF.resized_crop(
            image,
            i,
            j,
            h,
            w,
            self.size,
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        target = TF.resized_crop(
            target,
            i,
            j,
            h,
            w,
            self.size,
            interpolation=TF.InterpolationMode.NEAREST,
        )
        return image, target


class PairedToTensorAndNormalize:
    """Convert PIL image to tensor + ImageNet-normalize; convert label to numpy."""

    def __init__(
        self,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
    ) -> None:
        self.normalize = T.Normalize(mean=mean, std=std)

    def __call__(self, image, target):
        image = self.normalize(TF.to_tensor(image))
        target = np.array(target, dtype=np.uint8)
        return image, target


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def build_train_transform(config: TrainConfig) -> PairedCompose:
    label_remap = build_label_remap()
    use_prefer = config.prefer_train_classes_active
    prefer_tuple = (
        tuple(config.prefer_train_images_with_classes)
        if config.prefer_train_images_with_classes
        else ()
    )
    crop_samples = config.rare_crop_num_samples if use_prefer else 1
    return PairedCompose(
        [
            PairedRandomHorizontalFlip(p=0.5),
            PairedColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            PairedRandomResizedCrop(
                size=config.img_size,
                scale=(0.5, 1.0),
                label_remap=label_remap,
                prefer_classes=prefer_tuple,
                num_samples=crop_samples,
            ),
            PairedToTensorAndNormalize(),
        ]
    )


def build_val_transform() -> PairedCompose:
    return PairedCompose(
        [
            PairedToTensorAndNormalize(),
        ]
    )
