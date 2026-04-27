"""Evaluation: per-class IoU, mIoU, and qualitative visualisation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Settings, TrainConfig
from .dataset import CityscapesSegDataset
from .labels import CLASS_COLORS, CLASS_NAMES
from .model import build_model
from .transforms import build_val_transform
from .utils import inv_normalize, label_to_color


def compute_miou(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    num_classes: int,
    device: torch.device,
    use_amp: bool = False,
) -> list[float]:
    """Compute per-class IoU and mean IoU over a DataLoader."""
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        for imgs, lbls, _ in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu()
            t_flat = lbls.view(-1)
            p_flat = preds.view(-1)
            indices = t_flat * num_classes + p_flat
            confusion += torch.bincount(
                indices,
                minlength=num_classes**2,
            ).reshape(num_classes, num_classes)

    ious: list[float] = []
    for c in range(num_classes):
        tp = confusion[c, c].item()
        fp = confusion[:, c].sum().item() - tp
        fn = confusion[c, :].sum().item() - tp
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else 0.0)
    return ious


def print_miou_report(ious: list[float]) -> None:
    """Pretty-print per-class IoU and mIoU."""
    print(f"{'Class':<20s} {'IoU':>8s}")
    print("-" * 30)
    for name, iou in zip(CLASS_NAMES, ious):
        print(f"{name:<20s} {iou:>8.4f}")
    print("-" * 30)
    print(f"{'mIoU':<20s} {np.mean(ious):>8.4f}")


def visualize_predictions(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    num_samples: int = 4,
    use_amp: bool = False,
) -> None:
    """Show input / ground-truth / prediction side by side."""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(16, 14))
    col_titles = ["Input Image", "Ground Truth", "Prediction"]

    for i in range(num_samples):
        img, lbl, _ = dataset[i]
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(img.unsqueeze(0).to(device)).argmax(dim=1).squeeze(0).cpu().numpy()

        img_vis = inv_normalize(img).permute(1, 2, 0).clamp(0, 1).numpy()
        gt_vis = label_to_color(lbl.numpy())
        pred_vis = label_to_color(pred)

        axes[i, 0].imshow(img_vis)
        axes[i, 1].imshow(gt_vis)
        axes[i, 2].imshow(pred_vis)

        for j in range(3):
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=13)

    patches = [
        mpatches.Patch(color=np.array(c) / 255, label=n) for c, n in zip(CLASS_COLORS, CLASS_NAMES)
    ]
    num_classes = len(CLASS_NAMES)
    fig.legend(handles=patches, loc="lower center", ncol=num_classes, fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()


def run_evaluation(
    checkpoint_path: Path,
    settings: Settings,
    num_val: int = 100,
    batch_size: int = 4,
    show_predictions: bool = False,
) -> dict[str, Any]:
    """Load a checkpoint produced by ``train.run_training`` and report mIoU on the val split.

    The checkpoint is self-describing — it carries ``model_name``, ``base_ch``, ``num_classes``,
    and image size — so no YAML config is required.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {
        "model_state_dict",
        "model_name",
        "base_ch",
        "num_classes",
        "img_height",
        "img_width",
    }
    missing = required - set(ckpt)
    if missing:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} is missing required keys: {sorted(missing)}. "
            "Was it produced by a recent run of `train`?"
        )

    cfg = TrainConfig(
        model_name=ckpt["model_name"],
        base_ch=ckpt["base_ch"],
        num_classes=ckpt["num_classes"],
        img_height=ckpt["img_height"],
        img_width=ckpt["img_width"],
        num_val=num_val,
        batch_size=batch_size,
    )

    device = torch.device(settings.device)
    use_amp = device.type == "cuda"

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(
        f"Loaded {cfg.model_name} (base_ch={cfg.base_ch}, num_classes={cfg.num_classes}, "
        f"img={cfg.img_height}x{cfg.img_width}) from {checkpoint_path}"
    )
    if "epoch" in ckpt and "best_val_loss" in ckpt:
        print(
            f"  trained {ckpt['epoch']} epoch(s), "
            f"best_val_loss={ckpt['best_val_loss']:.4f}, "
            f"best_val_acc={ckpt.get('best_val_acc', float('nan')):.4f}"
        )

    val_ds = CityscapesSegDataset(
        settings.data_root,
        "valid",
        img_size=cfg.img_size,
        transform=build_val_transform(),
        max_samples=num_val,
        seed=42,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
    )
    print(f"Val: {len(val_ds)} samples ({len(val_loader)} batches)")

    ious = compute_miou(model, val_loader, cfg.num_classes, device, use_amp=use_amp)
    print_miou_report(ious)
    miou = float(np.mean(ious))

    if show_predictions:
        visualize_predictions(model, val_ds, device, use_amp=use_amp)

    return {
        "ious": ious,
        "mIoU": miou,
        "epoch": ckpt.get("epoch"),
        "best_val_loss": ckpt.get("best_val_loss"),
        "best_val_acc": ckpt.get("best_val_acc"),
    }
