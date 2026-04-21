"""Training and validation loops."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import Settings, TrainConfig
from .dataset import CityscapesSegDataset
from .evaluate import compute_miou, print_miou_report
from .labels import CLASS_NAMES, NUM_CLASSES
from .loss import build_criterion
from .model import build_model
from .transforms import build_train_transform, build_val_transform
from .utils import inv_normalize, label_to_color


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    loss_type: str,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    model.train()
    use_amp = scaler is not None
    running_loss, correct, total = 0.0, 0, 0
    for imgs, lbls, masks in tqdm(loader, desc="Train", leave=False):
        imgs, lbls, masks = imgs.to(device), lbls.to(device), masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(imgs)
            loss = criterion(out, lbls, masks) if loss_type == "focal" else criterion(out, lbls)

        if use_amp:
            scaler.scale(loss).backward()  # type: ignore[union-attr]
            scaler.step(optimizer)  # type: ignore[union-attr]
            scaler.update()  # type: ignore[union-attr]
        else:
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
    use_amp: bool = False,
) -> tuple[float, float]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        for imgs, lbls, masks in tqdm(loader, desc="Val  ", leave=False):
            imgs, lbls, masks = imgs.to(device), lbls.to(device), masks.to(device)
            out = model(imgs)
            loss = criterion(out, lbls, masks) if loss_type == "focal" else criterion(out, lbls)
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
    # Inverse-frequency weights
    # class_weights = (inv_freq / inv_freq.sum() * num_classes).to(device)
    # Dampened weights
    class_weights = (inv_freq.sqrt() / inv_freq.sqrt().sum() * NUM_CLASSES).to(device)
    return class_weights


def _make_run_name(config: TrainConfig) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{config.model_name}_bs{config.batch_size}_lr{config.lr:.0e}_{config.loss_type}"


def _log_predictions(
    writer: SummaryWriter,
    model: nn.Module,
    dataset,
    device: torch.device,
    num_samples: int = 4,
    use_amp: bool = False,
) -> None:
    """Log input / ground-truth / prediction grids to TensorBoard."""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(8, 7))
    col_titles = ["Input", "Ground Truth", "Prediction"]

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

    plt.tight_layout()
    writer.add_figure("Predictions", fig)
    plt.close(fig)


def run_training(config: TrainConfig, settings: Settings) -> None:
    """Main orchestrator: build dataset, model, optimizer, run epochs."""
    device = torch.device(settings.device)
    print(f"Using device: {device}")

    # Transforms
    train_transform = (
        build_train_transform(config) if config.augment_train else build_val_transform()
    )
    val_transform = build_val_transform()
    print(f"Train augmentation: {'enabled' if config.augment_train else 'disabled'}")

    # Datasets (only load the number of samples we actually need)
    train_ds = CityscapesSegDataset(
        settings.data_root,
        "train",
        img_size=config.img_size,
        transform=train_transform,
        max_samples=config.num_train,
        seed=config.seed,
    )
    val_ds = CityscapesSegDataset(
        settings.data_root,
        "valid",
        img_size=config.img_size,
        transform=val_transform,
        max_samples=config.num_val,
        seed=config.seed,
    )

    # Oversampling (optional)
    sampler = None
    if config.oversample_classes:
        oversample_boost = 3.0
        sample_weights = torch.ones(len(train_ds))
        n_boosted = 0
        for i in range(len(train_ds)):
            _, lbl, _ = train_ds[i]
            if any((lbl == c).any() for c in config.oversample_classes):
                sample_weights[i] = oversample_boost
                n_boosted += 1
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
        boosted_names = [CLASS_NAMES[c] for c in config.oversample_classes]
        print(
            f"Oversampling: {n_boosted}/{len(train_ds)} images boosted {oversample_boost}x "
            f"(contain {', '.join(boosted_names)})"
        )
    else:
        print("Oversampling: disabled")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
    )

    print(f"Train: {len(train_ds)} samples  ({len(train_loader)} batches)")
    print(f"Val:   {len(val_ds)} samples  ({len(val_loader)} batches)")

    # Class weights & criterion
    if config.use_class_weights:
        class_weights = _compute_class_weights(train_loader, config.num_classes, device)
        print("\nClass weights (dampened inverse-frequency):")
        for name, w in zip(CLASS_NAMES, class_weights):
            print(f"  {name:<20s}  weight: {w:.3f}")
    else:
        class_weights = None
        print("\nClass weights: disabled (uniform)")

    criterion = build_criterion(config, class_weights, device)

    # Model
    model = build_model(config).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{config.model_name}  |  Total params: {total_p:,}  |  Trainable: {trainable_p:,}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    scheduler = None
    if config.lr_scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.plateau_factor,
            patience=config.plateau_patience,
            min_lr=config.plateau_min_lr,
        )
        print(
            f"LR scheduler: ReduceLROnPlateau "
            f"(patience={config.plateau_patience}, factor={config.plateau_factor}, "
            f"min_lr={config.plateau_min_lr})"
        )
    else:
        print("LR scheduler: disabled")

    # AMP scaler (only when enabled and on CUDA)
    use_amp = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"Mixed precision (AMP): {'enabled' if use_amp else 'disabled'}")

    # TensorBoard
    run_name = _make_run_name(config)
    log_path = Path(settings.log_dir) / run_name
    writer = SummaryWriter(log_dir=str(log_path))
    print(f"TensorBoard log dir: {log_path}")

    # Training loop
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    epoch_bar = tqdm(range(1, config.num_epochs + 1), desc="Epochs")
    for epoch in epoch_bar:
        t_loss, t_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            config.loss_type,
            scaler=scaler,
        )
        v_loss, v_acc = validate(
            model,
            val_loader,
            criterion,
            device,
            config.loss_type,
            use_amp=use_amp,
        )

        train_losses.append(t_loss)
        train_accs.append(t_acc)
        val_losses.append(v_loss)
        val_accs.append(v_acc)

        writer.add_scalars("Loss", {"train": t_loss, "val": v_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": t_acc, "val": v_acc}, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if scheduler is not None:
            scheduler.step(v_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        epoch_bar.set_postfix(
            t_loss=f"{t_loss:.4f}",
            t_acc=f"{t_acc:.4f}",
            v_loss=f"{v_loss:.4f}",
            v_acc=f"{v_acc:.4f}",
            lr=f"{lr_now:.2e}",
        )

    print("\nTraining complete.")
    if config.load_best_checkpoint and best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"Loaded best checkpoint (val_loss={best_val_loss:.4f}) for evaluation.")

    # Evaluation
    ious = compute_miou(model, val_loader, config.num_classes, device, use_amp=use_amp)
    print_miou_report(ious)
    miou = float(np.mean(ious))

    # Log per-class IoU and mIoU
    for name, iou in zip(CLASS_NAMES, ious):
        writer.add_scalar(f"IoU/{name}", iou, config.num_epochs)
    writer.add_scalar("IoU/mIoU", miou, config.num_epochs)

    # Log prediction visualisations
    _log_predictions(writer, model, val_ds, device, num_samples=4, use_amp=use_amp)

    # Log hyperparameters alongside final metrics
    writer.add_hparams(
        {
            "model": config.model_name,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "loss_type": config.loss_type,
            "num_epochs": config.num_epochs,
            "num_train": config.num_train,
            "img_size": f"{config.img_height}x{config.img_width}",
            "lr_scheduler": config.lr_scheduler,
            "load_best_checkpoint": int(config.load_best_checkpoint),
        },
        {
            "hparam/val_loss": val_losses[-1],
            "hparam/val_acc": val_accs[-1],
            "hparam/best_val_loss": best_val_loss,
            "hparam/mIoU": miou,
        },
    )

    writer.close()
    print(f"TensorBoard logs saved to {log_path}")


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
