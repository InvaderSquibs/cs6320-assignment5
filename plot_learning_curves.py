#!/usr/bin/env python3
"""Plot Beatport key CNN histories using the Week 5 student example pattern."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ASSIGNMENT_DIR = Path(__file__).resolve().parent
PLOT_EXAMPLE = ASSIGNMENT_DIR / "week05_student_examples" / "plot_learning_curves_example.py"
DEFAULT_HISTORY = ASSIGNMENT_DIR / "outputs" / "cnn_runs"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "histories",
        nargs="*",
        type=Path,
        help="History CSV files (default: all *_history.csv under outputs/cnn_runs).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_HISTORY / "plots")
    args = parser.parse_args()

    histories = args.histories
    if not histories:
        histories = sorted(DEFAULT_HISTORY.glob("*_history.csv"))
    if not histories:
        print("No history CSV files found.", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        str(PLOT_EXAMPLE),
        *[str(p) for p in histories],
        "--output-dir",
        str(args.output_dir),
        "--loss-cols",
        "train_loss",
        "validation_loss",
        "--metric-cols",
        "validation_macro_f1",
        "validation_accuracy",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Plots written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
