#!/usr/bin/env python3
"""Small split-audit examples for Assignment 5.

These helpers are intentionally generic. Copy the functions you need and adapt
column names to your portfolio or approved practice dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def split_counts(frame: pd.DataFrame, split_col: str = "split") -> pd.DataFrame:
    counts = frame[split_col].value_counts().rename_axis(split_col).reset_index(name="rows")
    counts["percent"] = counts["rows"] / len(frame) * 100
    return counts.sort_values(split_col).reset_index(drop=True)


def numeric_distributions_by_split(
    frame: pd.DataFrame,
    numeric_cols: list[str],
    split_col: str = "split",
) -> pd.DataFrame:
    rows = []
    for split_name, split_frame in frame.groupby(split_col, sort=False):
        for column in numeric_cols:
            values = split_frame[column].dropna().astype(float)
            rows.append(
                {
                    split_col: split_name,
                    "column": column,
                    "count": int(values.count()),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "p25": float(values.quantile(0.25)),
                    "median": float(values.quantile(0.50)),
                    "p75": float(values.quantile(0.75)),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def category_coverage_against_train(
    frame: pd.DataFrame,
    category_cols: list[str],
    split_col: str = "split",
    train_name: str = "train",
) -> pd.DataFrame:
    train = frame[frame[split_col] == train_name]
    rows = []
    for column in category_cols:
        train_values = set(train[column].dropna().astype(str).unique())
        for split_name, split_frame in frame.groupby(split_col, sort=False):
            if split_name == train_name:
                continue
            split_values = set(split_frame[column].dropna().astype(str).unique())
            unseen_values = split_values - train_values
            unseen_rows = int(split_frame[column].dropna().astype(str).isin(unseen_values).sum())
            rows.append(
                {
                    "column": column,
                    split_col: split_name,
                    "train_unique_values": len(train_values),
                    "split_unique_values": len(split_values),
                    "unseen_values_count": len(unseen_values),
                    "unseen_row_count": unseen_rows,
                    "unseen_row_percent": unseen_rows / len(split_frame) * 100 if len(split_frame) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create simple Assignment 5 split-audit tables.")
    parser.add_argument("data", type=Path, help="CSV or Parquet file containing a split column.")
    parser.add_argument("--split-col", default="split", help="Split column name.")
    parser.add_argument("--numeric", nargs="*", default=[], help="Numeric columns to summarize by split.")
    parser.add_argument("--category", nargs="*", default=[], help="Categorical columns to check against train.")
    parser.add_argument("--output-dir", type=Path, default=Path("split_audit_outputs"), help="Output directory.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = read_table(args.data)
    split_counts(frame, args.split_col).to_csv(args.output_dir / "split_counts.csv", index=False)
    if args.numeric:
        numeric_distributions_by_split(frame, args.numeric, args.split_col).to_csv(
            args.output_dir / "split_numeric_distributions.csv",
            index=False,
        )
    if args.category:
        category_coverage_against_train(frame, args.category, args.split_col).to_csv(
            args.output_dir / "split_category_coverage.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
