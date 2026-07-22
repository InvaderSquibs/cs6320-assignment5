#!/usr/bin/env python3
"""Plot key label counts by train/val/test split (Assignment 5 split audit visuals)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BEATPORT_DIR = REPO_ROOT / "beatport"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "split_audit"

SPLIT_ORDER = ["train", "val", "test"]
SPLIT_COLORS = {"train": "#2ecc71", "val": "#3498db", "test": "#e74c3c"}


def key_count_table(frame: pd.DataFrame, unit_name: str) -> pd.DataFrame:
    counts = (
        frame.groupby(["split", "key_24"], observed=False)
        .size()
        .reset_index(name="count")
    )
    wide = counts.pivot(index="key_24", columns="split", values="count").fillna(0).astype(int)
    for split in SPLIT_ORDER:
        if split not in wide.columns:
            wide[split] = 0
    wide = wide[SPLIT_ORDER]
    wide["total"] = wide.sum(axis=1)
    wide = wide.sort_values("total", ascending=False)
    wide.index.name = unit_name
    return wide.reset_index()


def key_share_table(counts_wide: pd.DataFrame, unit_name: str) -> pd.DataFrame:
    share = counts_wide.copy()
    for split in SPLIT_ORDER:
        total = share[split].sum()
        share[split] = (share[split] / total * 100).round(2) if total else 0.0
    share.index.name = unit_name
    return share


def plot_grouped_bars(counts_wide: pd.DataFrame, title: str, ylabel: str, path: Path) -> None:
    keys = counts_wide["key_24"].tolist()
    x = np.arange(len(keys))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, split in enumerate(SPLIT_ORDER):
        offset = (i - 1) * width
        ax.bar(
            x + offset,
            counts_wide[split],
            width,
            label=split,
            color=SPLIT_COLORS[split],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="split")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_key_curves_by_split(
    counts_wide: pd.DataFrame,
    title: str,
    ylabel: str,
    path: Path,
    *,
    normalize: bool = False,
) -> None:
    """Line per split across keys (x = key class, y = count or % of split)."""
    fig, ax = plt.subplots(figsize=(14, 6))
    keys = counts_wide["key_24"].tolist()
    x = np.arange(len(keys))

    plot_df = counts_wide.copy()
    if normalize:
        for split in SPLIT_ORDER:
            total = plot_df[split].sum()
            plot_df[split] = plot_df[split] / total * 100 if total else 0

    for split in SPLIT_ORDER:
        ax.plot(
            x,
            plot_df[split],
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=split,
            color=SPLIT_COLORS[split],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="split")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_mode_by_split(segments: pd.DataFrame, path: Path) -> None:
    counts = (
        segments.groupby(["split", "mode"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(SPLIT_ORDER)
    )
    counts.plot(kind="bar", color=["#9b59b6", "#f39c12"], figsize=(7, 4), rot=0)
    plt.title("Mode counts by split (key segments)")
    plt.ylabel("segments")
    plt.xlabel("split")
    plt.legend(title="mode")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-sec", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    datasets_dir = BEATPORT_DIR / "datasets"
    segments = pd.read_csv(datasets_dir / "key_splits.csv")
    chunks = pd.read_csv(datasets_dir / f"clip_{args.clip_sec}s" / "key_chunks.csv")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    seg_counts = key_count_table(segments, "key_24")
    chunk_counts = key_count_table(chunks, "key_24")
    seg_share = key_share_table(seg_counts.drop(columns=["total"]), "key_24")
    chunk_share = key_share_table(chunk_counts.drop(columns=["total"]), "key_24")

    seg_counts.to_csv(args.output_dir / "segment_key_counts_by_split.csv", index=False)
    chunk_counts.to_csv(args.output_dir / "chunk_key_counts_by_split.csv", index=False)
    seg_share.to_csv(args.output_dir / "segment_key_share_by_split.csv", index=False)
    chunk_share.to_csv(args.output_dir / "chunk_key_share_by_split.csv", index=False)

    plot_grouped_bars(
        seg_counts,
        "Key segment counts by split (one row per timed key period)",
        "segments",
        args.output_dir / "segment_key_counts_grouped.png",
    )
    plot_key_curves_by_split(
        seg_counts,
        "Key segment counts across splits (line per split)",
        "segment count",
        args.output_dir / "segment_key_counts_curves.png",
    )
    plot_key_curves_by_split(
        seg_share,
        "Key segment share of each split (% — should track closely if stratified)",
        "% of split",
        args.output_dir / "segment_key_share_curves.png",
        normalize=False,
    )

    plot_grouped_bars(
        chunk_counts,
        f"Chunk key counts by split ({args.clip_sec}s clips, sample-weighted rows)",
        "chunks",
        args.output_dir / "chunk_key_counts_grouped.png",
    )
    plot_key_curves_by_split(
        chunk_counts,
        f"Chunk key counts across splits ({args.clip_sec}s)",
        "chunk count",
        args.output_dir / "chunk_key_counts_curves.png",
    )

    plot_mode_by_split(segments, args.output_dir / "mode_counts_by_split.png")

    summary_lines = [
        "# Key distribution by split",
        "",
        f"Clip length: **{args.clip_sec}s**",
        "",
        "## Segment-level (split unit for modeling)",
        "",
        f"- Total segments: {len(segments)} ({segments['split'].value_counts().to_dict()})",
        f"- Unique keys in train: {segments.loc[segments['split']=='train', 'key_24'].nunique()}",
        f"- Timed segments: {int(segments.get('is_timed_segment', pd.Series(False)).sum()) if 'is_timed_segment' in segments.columns else 'n/a'}",
        "",
        "See `segment_key_counts_by_split.csv` and PNG plots in this folder.",
        "",
        "## Chunk-level (what the classifier trains on)",
        "",
        f"- Total chunks: {len(chunks)}",
        "",
        "Use **share curves** to verify stratification: train/val/test lines should follow similar shapes.",
        "",
    ]
    (args.output_dir / "key_distribution_summary.md").write_text("\n".join(summary_lines))

    print(f"Wrote key distribution tables and plots to {args.output_dir}")
    print("\nSegment totals by split:")
    print(segments["split"].value_counts().reindex(SPLIT_ORDER))
    print("\nTop 5 keys (segments, train):")
    print(
        segments[segments["split"] == "train"]["key_24"]
        .value_counts()
        .head()
        .to_string()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
