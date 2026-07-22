#!/usr/bin/env python3
"""Export predictions and error/slice/calibration reports for a chroma MLP run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
BEATPORT_DIR = REPO_ROOT / "beatport"
ASSIGNMENT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = ASSIGNMENT_DIR / "outputs" / "chroma_runs"
DEFAULT_OUT = ASSIGNMENT_DIR / "outputs" / "error_analysis"

sys.path.insert(0, str(BEATPORT_DIR))

from chroma_features import CHROMA_INPUT_DIM, filter_chunks_with_chroma  # noqa: E402
from config_loader import load_config  # noqa: E402
from dataset import BeatportKeyChromaDataset, collate_chroma_batch  # noqa: E402
from device import resolve_device  # noqa: E402
from models.key_chroma_mlp import KeyChromaMLP  # noqa: E402
from paths import key_chroma_dir, key_chunks_path  # noqa: E402


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, LabelEncoder]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder = LabelEncoder()
    encoder.classes_ = np.array(ckpt["label_classes"])
    model = KeyChromaMLP(
        n_classes=len(encoder.classes_),
        input_dim=int(ckpt.get("input_dim", CHROMA_INPUT_DIM)),
        hidden_dim=int(ckpt["hidden_dim"]),
        dropout=float(ckpt.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, encoder


def export_predictions(
    model: nn.Module,
    chunks: pd.DataFrame,
    chroma_dir: Path,
    encoder: LabelEncoder,
    device: torch.device,
    *,
    batch_size: int,
) -> pd.DataFrame:
    dataset = BeatportKeyChromaDataset(chunks, chroma_dir, encoder)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_chroma_batch,
    )
    rows: list[dict] = []
    chunk_df = chunks.reset_index(drop=True)

    with torch.no_grad():
        offset = 0
        for batch in loader:
            features = batch["features"].to(device)
            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            confidences = probs.max(dim=1).values.cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = batch["label"].cpu().numpy()
            batch_size_actual = len(labels)

            for i in range(batch_size_actual):
                meta = chunk_df.iloc[offset + i]
                true_name = encoder.inverse_transform([int(labels[i])])[0]
                pred_name = encoder.inverse_transform([int(preds[i])])[0]
                rows.append(
                    {
                        "split": meta["split"],
                        "track_id": int(meta["track_id"]),
                        "chunk_idx": int(meta["chunk_idx"]),
                        "true_label": true_name,
                        "predicted_label": pred_name,
                        "confidence": float(confidences[i]),
                        "correct": true_name == pred_name,
                        "mode": meta.get("mode", ""),
                        "pitch_class": int(meta["pitch_class"]) if pd.notna(meta.get("pitch_class")) else -1,
                        "key_24": meta["key_24"],
                        "start_sec": float(meta["start_sec"]),
                        "end_sec": float(meta["end_sec"]),
                    }
                )
            offset += batch_size_actual

    return pd.DataFrame(rows)


def chunk_position_bucket(chunk_idx: int) -> str:
    if chunk_idx <= 2:
        return "early_preview_0_45s"
    if chunk_idx <= 5:
        return "mid_preview_45_90s"
    return "late_preview_90s_plus"


def slice_metrics(df: pd.DataFrame, slice_col: str) -> pd.DataFrame:
    records: list[dict] = []
    for value, group in df.groupby(slice_col, observed=False):
        n = len(group)
        acc = group["correct"].mean() if n else 0.0
        macro_f1 = f1_score(
            group["true_label"],
            group["predicted_label"],
            average="macro",
            zero_division=0,
        )
        records.append(
            {
                slice_col: value,
                "n": n,
                "accuracy": round(acc, 4),
                "macro_f1": round(macro_f1, 4),
                "mean_confidence": round(group["confidence"].mean(), 4),
            }
        )
    return pd.DataFrame(records)


def plot_confusion_matrix_heatmap(cm_df: pd.DataFrame, out_path: Path) -> None:
    labels = cm_df["true_label"].tolist()
    matrix = cm_df.drop(columns=["true_label"]).values.astype(float)
    # Normalize by row (true class) for readability
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_norm = np.zeros_like(matrix)
    np.divide(matrix, row_sums, out=matrix_norm, where=row_sums > 0)

    n = len(labels)
    fig_size = max(10, n * 0.45)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(matrix_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short = [s.replace(" ", "\n") for s in labels]
    ax.set_xticklabels(short, rotation=90, fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel("Predicted key")
    ax.set_ylabel("True key")
    ax.set_title("Confusion matrix (row-normalized) — test set")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction of true class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_per_class_f1(report_df: pd.DataFrame, out_path: Path) -> None:
    class_rows = report_df[report_df["label"].apply(lambda x: x not in ("accuracy", "macro avg", "weighted avg"))]
    class_rows = class_rows.copy()
    class_rows["f1-score"] = class_rows["f1-score"].astype(float)
    class_rows = class_rows.sort_values("f1-score", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#c44e52" if v < 0.2 else "#4c72b0" for v in class_rows["f1-score"]]
    ax.barh(class_rows["label"], class_rows["f1-score"], color=colors)
    ax.set_xlabel("F1 score")
    ax.set_title("Per-class F1 — test set (24 keys)")
    ax.set_xlim(0, 1)
    ax.axvline(class_rows["f1-score"].mean(), color="gray", linestyle="--", label="mean class F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_slice_bars(slice_df: pd.DataFrame, slice_col: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(slice_df))
    width = 0.35
    ax.bar(x - width / 2, slice_df["accuracy"], width, label="accuracy", color="#4c72b0")
    ax.bar(x + width / 2, slice_df["macro_f1"], width, label="macro F1", color="#55a868")
    ax.set_xticks(x)
    ax.set_xticklabels(slice_df[slice_col].astype(str), rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    for i, n in enumerate(slice_df["n"]):
        ax.annotate(f"n={n}", (x[i], 0.02), ha="center", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confidence_distribution(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    correct = df[df["correct"]]["confidence"]
    wrong = df[~df["correct"]]["confidence"]
    bins = np.linspace(0, 1, 21)
    ax.hist(wrong, bins=bins, alpha=0.6, label=f"incorrect (n={len(wrong)})", color="#c44e52")
    ax.hist(correct, bins=bins, alpha=0.6, label=f"correct (n={len(correct)})", color="#4c72b0")
    ax.set_xlabel("Predicted confidence (max softmax)")
    ax.set_ylabel("Count")
    ax.set_title("Confidence distribution — test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_error_analysis_figures(
    out_dir: Path,
    test_df: pd.DataFrame,
    cm_df: pd.DataFrame,
    report_df: pd.DataFrame,
    slice_mode: pd.DataFrame,
    slice_position: pd.DataFrame,
    *,
    calibration_bins: int,
) -> Path:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix_heatmap(cm_df, plots_dir / "confusion_matrix_heatmap.png")
    plot_per_class_f1(report_df, plots_dir / "per_class_f1.png")
    plot_slice_bars(slice_mode, "mode", "Accuracy by mode — test set", plots_dir / "slice_by_mode.png")
    plot_slice_bars(
        slice_position,
        "chunk_position_bucket",
        "Accuracy by chunk position in preview — test set",
        plots_dir / "slice_by_chunk_position.png",
    )
    plot_confidence_distribution(test_df, plots_dir / "confidence_distribution.png")
    plot_reliability_diagram(test_df, plots_dir / "reliability_diagram.png", n_bins=calibration_bins)
    return plots_dir


def plot_reliability_diagram(df: pd.DataFrame, out_path: Path, n_bins: int = 10) -> None:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(df["confidence"], bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_acc: list[float] = []
    bin_conf: list[float] = []
    bin_counts: list[int] = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        subset = df.loc[mask]
        bin_acc.append(subset["correct"].mean())
        bin_conf.append(subset["confidence"].mean())
        bin_counts.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax.scatter(bin_conf, bin_acc, s=[max(20, c / 5) for c in bin_counts], alpha=0.8)
    ax.set_xlabel("Mean predicted confidence (bin)")
    ax.set_ylabel("Fraction correct (bin)")
    ax.set_title("Reliability diagram — exp3 chroma MLP (test)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_summary_md(out_dir: Path, test_df: pd.DataFrame, meta: dict) -> None:
    acc = test_df["correct"].mean()
    macro_f1 = f1_score(test_df["true_label"], test_df["predicted_label"], average="macro", zero_division=0)
    hi_err = int((~test_df["correct"] & (test_df["confidence"] >= 0.9)).sum())
    lines = [
        "# Error analysis summary — exp3 chroma baseline",
        "",
        f"- Test chunks: {len(test_df)}",
        f"- Test accuracy: {acc:.3f}",
        f"- Test macro F1: {macro_f1:.3f}",
        f"- High-confidence errors (conf ≥ 0.9): {hi_err}",
        "",
        "## Artifacts in this folder",
        "",
        "- `predictions_test.csv` / `predictions_val.csv`",
        "- `classification_confusion_matrix.csv`",
        "- `classification_per_class_report.csv`",
        "- `slice_by_mode.csv`, `slice_by_chunk_position.csv`",
        "- `confidence_summary.csv`, `high_confidence_errors.csv`",
        "- `plots/confusion_matrix_heatmap.png`",
        "- `plots/per_class_f1.png`",
        "- `plots/slice_by_mode.png`, `plots/slice_by_chunk_position.png`",
        "- `plots/confidence_distribution.png`",
        "- `plots/reliability_diagram.png`",
        "",
        "## Checkpoint",
        "",
        f"- Best val loss (training): {meta.get('best_val_loss', 'n/a')}",
        f"- Epochs ran: {meta.get('epochs_ran', 'n/a')}",
    ]
    (out_dir / "error_analysis_summary.md").write_text("\n".join(lines) + "\n")


def run_analysis(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    checkpoint_path = run_dir / f"{args.run_name}_checkpoint.pt"
    if not checkpoint_path.exists():
        print(
            f"Missing checkpoint: {checkpoint_path}\n"
            "Run once (writes checkpoint, ~15 min):\n"
            "  PYTHONUNBUFFERED=1 python ../assignment_5/train_key_chroma.py "
            f"--run-name {args.run_name} --learning-rate 0.00275 --epochs 30 --quiet",
            file=sys.stderr,
        )
        raise SystemExit(1)

    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    data_dir = Path(cfg["output_dir"])
    if not data_dir.is_absolute():
        data_dir = BEATPORT_DIR / data_dir

    chunks = pd.read_csv(key_chunks_path(data_dir, args.clip_sec))
    chroma_dir = key_chroma_dir(data_dir, args.clip_sec)
    chunks = filter_chunks_with_chroma(chunks, chroma_dir)

    device = resolve_device(force_cpu=args.cpu, preference=args.device)
    model, encoder = load_model_from_checkpoint(checkpoint_path, device)

    ckpt_meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if not args.skip_export:
        print("Exporting predictions...")
        for split_name in ("test", "val"):
            split_df = chunks[chunks["split"] == split_name]
            preds = export_predictions(
                model, split_df, chroma_dir, encoder, device, batch_size=args.batch_size
            )
            preds["chunk_position_bucket"] = preds["chunk_idx"].map(chunk_position_bucket)
            out_csv = out_dir / f"predictions_{split_name}.csv"
            preds.to_csv(out_csv, index=False)
            print(f"  Wrote {out_csv} ({len(preds)} rows)")

    if args.export_only:
        print(f"Export complete under {out_dir}")
        return

    test_path = out_dir / "predictions_test.csv"
    if not test_path.exists():
        print(f"Missing {test_path}; run without --skip-export first.", file=sys.stderr)
        raise SystemExit(1)

    test_df = pd.read_csv(test_path)
    if "chunk_position_bucket" not in test_df.columns:
        test_df["chunk_position_bucket"] = test_df["chunk_idx"].map(chunk_position_bucket)

    labels = sorted(set(test_df["true_label"]) | set(test_df["predicted_label"]))
    cm = pd.DataFrame(
        confusion_matrix(test_df["true_label"], test_df["predicted_label"], labels=labels),
        index=labels,
        columns=labels,
    ).reset_index(names="true_label")
    cm.to_csv(out_dir / "classification_confusion_matrix.csv", index=False)

    report = pd.DataFrame(
        classification_report(
            test_df["true_label"],
            test_df["predicted_label"],
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report = report.reset_index(names="label")
    report.to_csv(out_dir / "classification_per_class_report.csv", index=False)

    conf_summary = (
        test_df.groupby("correct", observed=False)["confidence"]
        .agg(count="size", mean_confidence="mean", median_confidence="median", max_confidence="max")
        .reset_index()
    )
    conf_summary.to_csv(out_dir / "confidence_summary.csv", index=False)

    hi_conf = test_df[(~test_df["correct"]) & (test_df["confidence"] >= args.confidence_threshold)]
    hi_conf.to_csv(out_dir / "high_confidence_errors.csv", index=False)

    slice_mode_df = slice_metrics(test_df, "mode")
    slice_position_df = slice_metrics(test_df, "chunk_position_bucket")
    slice_mode_df.to_csv(out_dir / "slice_by_mode.csv", index=False)
    slice_position_df.to_csv(out_dir / "slice_by_chunk_position.csv", index=False)

    plots_dir = plot_error_analysis_figures(
        out_dir,
        test_df,
        cm,
        report,
        slice_mode_df,
        slice_position_df,
        calibration_bins=args.calibration_bins,
    )
    print(f"  Wrote plots under {plots_dir}")

    write_summary_md(out_dir, test_df, ckpt_meta)
    print(f"Reports written to {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Error/slice/calibration reports for chroma key MLP.")
    parser.add_argument("--run-name", default="exp3_chroma_baseline")
    parser.add_argument("--clip-sec", type=int, default=15)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--export-only", action="store_true", help="Only write prediction CSVs.")
    parser.add_argument("--skip-export", action="store_true", help="Use existing prediction CSVs.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.export_only:
        args.skip_export = False
    run_analysis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
