"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from ecg_arrhythmia.experiment import download_database, run_experiments
from ecg_arrhythmia.models import MODEL_FACTORIES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ecg-arrhythmia",
        description="Train and evaluate ECG beat classifiers across PhysioNet datasets.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Download public PhysioNet data.")
    download.add_argument(
        "--dataset", choices=("mitdb", "incartdb", "all"), required=True
    )
    download.add_argument("--data-dir", type=Path, default=Path("data"))

    run = commands.add_parser("run", help="Train models and write evaluation results.")
    run.add_argument("--data-dir", type=Path, default=Path("data"))
    run.add_argument("--output-dir", type=Path, default=Path("runs"))
    run.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_FACTORIES),
        default=tuple(MODEL_FACTORIES),
    )
    run.add_argument("--epochs", type=int, default=30)
    run.add_argument("--patience", type=int, default=6)
    run.add_argument("--batch-size", type=int, default=256)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    run.add_argument(
        "--adapt-incart",
        action="store_true",
        help="Fine-tune on INCART adaptation patients and evaluate held-out patients.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "download":
        datasets = ("mitdb", "incartdb") if args.dataset == "all" else (args.dataset,)
        for dataset in datasets:
            download_database(dataset, args.data_dir)
            print(f"Downloaded {dataset} to {args.data_dir / dataset}")
    elif args.command == "run":
        run_experiments(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_names=args.models,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            seed=args.seed,
            device_name=args.device,
            adapt_incart=args.adapt_incart,
        )


if __name__ == "__main__":
    main()
