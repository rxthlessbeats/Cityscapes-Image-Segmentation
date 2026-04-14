"""Paired image-label augmentation transforms for semantic segmentation."""

from __future__ import annotations

import random

import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from .config import TrainConfig


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
    """Random crop + resize back to target size (same crop for both)."""

    def __init__(self, size, scale=(0.5, 1.0), ratio=(0.75, 1.33)) -> None:
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, image, target):
        i, j, h, w = T.RandomResizedCrop.get_params(image, self.scale, self.ratio)
        image = TF.resized_crop(
            image, i, j, h, w, self.size,
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        target = TF.resized_crop(
            target, i, j, h, w, self.size,
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
    return PairedCompose([
        PairedRandomHorizontalFlip(p=0.5),
        PairedColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        PairedRandomResizedCrop(size=config.img_size, scale=(0.5, 1.0)),
        PairedToTensorAndNormalize(),
    ])


def build_val_transform() -> PairedCompose:
    return PairedCompose([
        PairedToTensorAndNormalize(),
    ])
