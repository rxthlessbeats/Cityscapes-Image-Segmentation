"""Pydantic-based configuration for environment settings and training hyperparameters."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Machine / environment-level settings, loaded from ``.env``."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="CITYSEG_")

    data_root: str = "./data/small_data"
    device: str = "cuda"
    num_workers: int = 0
    pin_memory: bool = True
    log_dir: str = "runs"


class TrainConfig(BaseModel):
    """Training hyperparameters with validation."""

    model_name: str = "fcn8s"
    img_height: int = Field(256, gt=0)
    img_width: int = Field(512, gt=0)
    batch_size: int = Field(4, gt=0)
    num_classes: int = Field(8, gt=1)
    num_epochs: int = Field(30, gt=0)
    lr: float = Field(1e-4, gt=0)
    weight_decay: float = Field(1e-3, ge=0)
    num_train: int = Field(400, gt=0)
    num_val: int = Field(100, gt=0)
    seed: int = 42
    loss_type: Literal["cross_entropy", "focal"] = "cross_entropy"
    focal_gamma: float = Field(2.0, ge=0)
    use_amp: bool = True
    augment_train: bool = True
    use_class_weights: bool = True
    lr_scheduler: Literal["none", "plateau"] = "plateau"
    plateau_patience: int = Field(2, ge=0)
    plateau_factor: float = Field(0.5, gt=0, lt=1)
    plateau_min_lr: float = Field(1e-6, ge=0)
    load_best_checkpoint: bool = True
    # Prefer images (at img_size after remap) with pixels in these classes; pad to num_train.
    # null or [] disables: uniform random subset + single random resized crop (no best-of-N).
    prefer_train_images_with_classes: list[int] | None = None
    prefer_train_min_rare_fraction: float = Field(0.05, ge=0.0, le=1.0)
    rare_crop_num_samples: int = Field(10, ge=1)

    @field_validator("model_name")
    @classmethod
    def _validate_model_name(cls, v: str) -> str:
        from .model import MODEL_REGISTRY

        v = v.lower()
        if v not in MODEL_REGISTRY:
            allowed = ", ".join(sorted(MODEL_REGISTRY))
            raise ValueError(f"Unknown model '{v}'. Choose from: {allowed}")
        return v

    @field_validator("loss_type")
    @classmethod
    def _validate_loss(cls, v: str) -> str:
        if v not in ("cross_entropy", "focal"):
            raise ValueError("loss_type must be 'cross_entropy' or 'focal'")
        return v

    @model_validator(mode="after")
    def _validate_prefer_classes(self) -> TrainConfig:
        if self.prefer_train_images_with_classes:
            for c in self.prefer_train_images_with_classes:
                if c < 0 or c >= self.num_classes:
                    raise ValueError(
                        f"prefer_train_images_with_classes entry {c} out of range "
                        f"for num_classes={self.num_classes}"
                    )
        return self

    @property
    def img_size(self) -> tuple[int, int]:
        return (self.img_height, self.img_width)

    @property
    def prefer_train_classes_active(self) -> bool:
        return bool(self.prefer_train_images_with_classes)


def load_train_config(path: str | Path = "config.yaml") -> TrainConfig:
    """Load ``TrainConfig`` from a YAML file, falling back to defaults."""
    p = Path(path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return TrainConfig(**data)
    return TrainConfig()
