"""CLI entry point: ``python -m cityscape_seg train`` / ``evaluate``."""

from __future__ import annotations

import argparse
import sys

from .config import Settings, load_train_config


def _train(args: argparse.Namespace) -> None:
    from .train import run_training

    config = load_train_config(args.config)
    settings = Settings()
    run_training(config, settings)


def _evaluate(args: argparse.Namespace) -> None:
    from pathlib import Path

    from .evaluate import run_evaluation

    run_evaluation(
        checkpoint_path=Path(args.checkpoint),
        settings=Settings(),
        num_val=args.num_val,
        batch_size=args.batch_size,
        show_predictions=args.show_predictions,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="cityscape_seg",
        description="Cityscapes semantic segmentation CLI",
    )
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="Run training")
    train_p.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )

    eval_p = sub.add_parser("evaluate", help="Evaluate a checkpoint")
    eval_p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    eval_p.add_argument(
        "--num-val",
        type=int,
        default=100,
        help="Number of validation samples to evaluate on (default: 100)",
    )
    eval_p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size during evaluation (default: 4)",
    )
    eval_p.add_argument(
        "--show-predictions",
        action="store_true",
        help="Pop a matplotlib window with input/GT/prediction grids",
    )

    args = parser.parse_args(argv)

    if args.command == "train":
        _train(args)
    elif args.command == "evaluate":
        _evaluate(args)
    else:
        parser.print_help()
        sys.exit(1)
