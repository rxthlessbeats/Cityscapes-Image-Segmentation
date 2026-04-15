"""Evaluation: per-class IoU, mIoU, and qualitative visualisation."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from .labels import CLASS_COLORS, CLASS_NAMES
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
