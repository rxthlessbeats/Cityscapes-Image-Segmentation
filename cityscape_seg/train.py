"""Training and validation loops."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from .config import Settings, TrainConfig
from .dataset import CityscapesSegDataset
from .evaluate import compute_miou, print_miou_report, visualize_predictions
from .labels import CLASS_NAMES, NUM_CLASSES
from .loss import build_criterion
from .model import FCN8s
from .transforms import build_train_transform, build_val_transform


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    loss_type: str,
) -> tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, lbls, masks in loader:
        imgs, lbls, masks = imgs.to(device), lbls.to(device), masks.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = (
            criterion(out, lbls, masks)
            if loss_type == "focal"
            else criterion(out, lbls)
        )
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == lbls).sum().item()
        total += lbls.numel()

    n = len(loader.dataset)  # type: ignore[arg-type]
    return running_loss / n, correct / total


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    loss_type: str,
) -> tuple[float, float]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, lbls, masks in loader:
            imgs, lbls, masks = imgs.to(device), lbls.to(device), masks.to(device)
            out = model(imgs)
            loss = (
                criterion(out, lbls, masks)
                if loss_type == "focal"
                else criterion(out, lbls)
            )
            running_loss += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.numel()

    n = len(loader.dataset)  # type: ignore[arg-type]
    return running_loss / n, correct / total


def _compute_class_weights(
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    pixel_counts = torch.zeros(num_classes)
    for _, lbl, _ in loader:
        for c in range(num_classes):
            pixel_counts[c] += (lbl == c).sum()

    freq = pixel_counts / pixel_counts.sum()
    inv_freq = 1.0 / (freq + 1e-6)
    return (inv_freq / inv_freq.sum() * num_classes).to(device)


def run_training(config: TrainConfig, settings: Settings) -> None:
    """Main orchestrator: build dataset, model, optimizer, run epochs."""
    device = torch.device(settings.device)
    print(f"Using device: {device}")

    # Transforms
    train_transform = build_train_transform(config)
    val_transform = build_val_transform()

    # Datasets
    train_full = CityscapesSegDataset(
        settings.data_root, "train",
        img_size=config.img_size, transform=train_transform,
    )
    val_full = CityscapesSegDataset(
        settings.data_root, "valid",
        img_size=config.img_size, transform=val_transform,
    )

    # Subsets
    rng = np.random.default_rng(config.seed)
    train_idx = rng.choice(
        len(train_full), size=min(config.num_train, len(train_full)), replace=False,
    )
    val_idx = rng.choice(
        len(val_full), size=min(config.num_val, len(val_full)), replace=False,
    )
    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=settings.num_workers, pin_memory=settings.pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=settings.num_workers, pin_memory=settings.pin_memory,
    )

    print(f"Train subset: {len(train_ds)} samples  ({len(train_loader)} batches)")
    print(f"Val   subset: {len(val_ds)} samples  ({len(val_loader)} batches)")

    # Class weights & criterion
    class_weights = _compute_class_weights(train_loader, config.num_classes, device)
    print("\nClass pixel distribution:")
    for name, w in zip(CLASS_NAMES, class_weights):
        print(f"  {name:<20s}  weight: {w:.3f}")

    criterion = build_criterion(config, class_weights, device)

    # Model
    model = FCN8s(config.num_classes).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nCustom FCN-8s  |  Total params: {total_p:,}  |  Trainable: {trainable_p:,}")

    optimizer = optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )

    # Training loop
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, config.num_epochs + 1):
        t_loss, t_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config.loss_type,
        )
        v_loss, v_acc = validate(
            model, val_loader, criterion, device, config.loss_type,
        )

        train_losses.append(t_loss)
        train_accs.append(t_acc)
        val_losses.append(v_loss)
        val_accs.append(v_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:>3d}/{config.num_epochs}  "
                f"Train Loss: {t_loss:.4f}  Acc: {t_acc:.4f}  |  "
                f"Val Loss: {v_loss:.4f}  Acc: {v_acc:.4f}"
            )

    print("\nTraining complete.")

    # Training curves
    _plot_curves(config.num_epochs, train_losses, val_losses, train_accs, val_accs)

    # Evaluation
    ious = compute_miou(model, val_loader, config.num_classes, device)
    print_miou_report(ious)

    # Qualitative visualisation
    visualize_predictions(model, val_ds, device, num_samples=4)


def _plot_curves(
    num_epochs: int,
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    epochs_range = range(1, num_epochs + 1)

    ax1.plot(epochs_range, train_losses, label="Train")
    ax1.plot(epochs_range, val_losses, label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(epochs_range, train_accs, label="Train")
    ax2.plot(epochs_range, val_accs, label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Pixel Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.show()
