#!/usr/bin/env python3
"""Examples for error/slice analysis and confidence summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def regression_error_by_slice(
    predictions: pd.DataFrame,
    slice_col: str,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> pd.DataFrame:
    frame = predictions.copy()
    frame["absolute_error"] = (frame[y_true_col] - frame[y_pred_col]).abs()
    return (
        frame.groupby(slice_col, observed=False)["absolute_error"]
        .agg(count="size", mean_absolute_error="mean", median_absolute_error="median")
        .reset_index()
    )


def regression_error_by_target_range(
    predictions: pd.DataFrame,
    bins: list[float],
    labels: list[str],
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> pd.DataFrame:
    frame = predictions.copy()
    frame["target_range"] = pd.cut(frame[y_true_col], bins=bins, labels=labels, right=False)
    return regression_error_by_slice(frame, "target_range", y_true_col, y_pred_col)


def classification_summary(
    predictions: pd.DataFrame,
    true_col: str = "true_label",
    pred_col: str = "predicted_label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = sorted(set(predictions[true_col].dropna()) | set(predictions[pred_col].dropna()))
    confusion = pd.DataFrame(
        confusion_matrix(predictions[true_col], predictions[pred_col], labels=labels),
        index=labels,
        columns=labels,
    ).reset_index(names="true_label")
    report = pd.DataFrame(
        classification_report(
            predictions[true_col],
            predictions[pred_col],
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    return confusion, report.reset_index(names="label")


def confidence_summary(
    predictions: pd.DataFrame,
    confidence_col: str = "confidence",
    correct_col: str = "correct",
) -> pd.DataFrame:
    return (
        predictions.groupby(correct_col, observed=False)[confidence_col]
        .agg(count="size", mean_confidence="mean", median_confidence="median", max_confidence="max")
        .reset_index()
    )


def high_confidence_errors(
    predictions: pd.DataFrame,
    confidence_col: str = "confidence",
    correct_col: str = "correct",
    threshold: float = 0.90,
) -> pd.DataFrame:
    return predictions[(~predictions[correct_col]) & (predictions[confidence_col] >= threshold)].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create simple Assignment 5 error and confidence summaries.")
    parser.add_argument("predictions", type=Path, help="Prediction CSV file.")
    parser.add_argument("--task", choices=["regression", "classification"], required=True)
    parser.add_argument("--slice-col", default=None, help="Optional subgroup/slice column for regression.")
    parser.add_argument("--output-dir", type=Path, default=Path("error_analysis_outputs"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions)
    if args.task == "regression":
        if args.slice_col is not None:
            regression_error_by_slice(predictions, args.slice_col).to_csv(
                args.output_dir / "regression_error_by_slice.csv",
                index=False,
            )
        regression_error_by_target_range(
            predictions,
            bins=[float("-inf"), 300, 600, 1200, float("inf")],
            labels=["short", "medium", "long", "very_long"],
        ).to_csv(args.output_dir / "regression_error_by_target_range.csv", index=False)
    else:
        confusion, report = classification_summary(predictions)
        confusion.to_csv(args.output_dir / "classification_confusion_matrix.csv", index=False)
        report.to_csv(args.output_dir / "classification_per_class_report.csv", index=False)
        if {"confidence", "correct"}.issubset(predictions.columns):
            confidence_summary(predictions).to_csv(args.output_dir / "confidence_summary.csv", index=False)
            high_confidence_errors(predictions).to_csv(args.output_dir / "high_confidence_errors.csv", index=False)


if __name__ == "__main__":
    main()
