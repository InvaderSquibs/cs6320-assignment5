#!/usr/bin/env python3
"""Plot training and validation learning curves from history CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_histories(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "run" not in frame.columns:
            frame["run"] = path.stem.replace("_history", "")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_curves(
    history: pd.DataFrame,
    y_columns: list[str],
    output_path: Path,
    title: str,
    ylabel: str,
    epoch_col: str = "epoch",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for run_name, run_frame in history.groupby("run", sort=False):
        for column in y_columns:
            if column not in run_frame.columns:
                continue
            label = run_name if len(y_columns) == 1 else f"{run_name} {column}"
            ax.plot(run_frame[epoch_col], run_frame[column], label=label, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Assignment 5 learning curves.")
    parser.add_argument("histories", nargs="+", type=Path, help="One or more history CSV files.")
    parser.add_argument("--loss-cols", nargs="*", default=["train_loss", "validation_loss"])
    parser.add_argument("--metric-cols", nargs="*", default=["validation_mae"])
    parser.add_argument("--output-dir", type=Path, default=Path("learning_curve_outputs"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = load_histories(args.histories)
    plot_curves(history, args.loss_cols, args.output_dir / "loss_learning_curve.png", "Loss Learning Curve", "Loss")
    plot_curves(
        history,
        args.metric_cols,
        args.output_dir / "validation_metric_learning_curve.png",
        "Validation Metric Learning Curve",
        "Metric",
    )


if __name__ == "__main__":
    main()
