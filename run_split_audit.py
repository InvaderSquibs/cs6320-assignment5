#!/usr/bin/env python3
"""Split audit for Beatport key task (Assignment 5 Phase 1)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BEATPORT_DIR = REPO_ROOT / "beatport"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "split_audit"


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def split_counts(frame: pd.DataFrame, split_col: str = "split") -> pd.DataFrame:
    counts = frame[split_col].value_counts().rename_axis(split_col).reset_index(name="rows")
    counts["percent"] = counts["rows"] / len(frame) * 100
    order = {"train": 0, "val": 1, "test": 2}
    counts["_order"] = counts[split_col].map(order)
    return counts.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def label_distribution_by_split(
    frame: pd.DataFrame,
    label_col: str,
    split_col: str = "split",
) -> pd.DataFrame:
    counts = (
        frame.groupby([split_col, label_col], observed=False)
        .size()
        .reset_index(name="count")
    )
    totals = frame.groupby(split_col, observed=False).size().rename("split_total")
    counts = counts.merge(totals, on=split_col)
    counts["percent_of_split"] = counts["count"] / counts["split_total"] * 100
    return counts.sort_values([split_col, label_col]).reset_index(drop=True)


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
            unseen_values = sorted(split_values - train_values)
            unseen_rows = int(
                split_frame[column].dropna().astype(str).isin(unseen_values).sum()
            )
            rows.append(
                {
                    "column": column,
                    split_col: split_name,
                    "train_unique_values": len(train_values),
                    "split_unique_values": len(split_values),
                    "unseen_values_count": len(unseen_values),
                    "unseen_values_sample": ", ".join(unseen_values[:5]),
                    "unseen_row_count": unseen_rows,
                    "unseen_row_percent": unseen_rows / len(split_frame) * 100
                    if len(split_frame)
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def track_overlap_check(splits: pd.DataFrame) -> pd.DataFrame:
    """Each track_id should appear in exactly one split."""
    per_track = splits.groupby("track_id")["split"].nunique()
    bad = per_track[per_track > 1]
    rows = [
        {
            "check": "track_in_multiple_splits",
            "passed": len(bad) == 0,
            "violation_count": int(len(bad)),
            "sample_track_ids": ", ".join(str(t) for t in bad.index[:10]),
        }
    ]
    return pd.DataFrame(rows)


def chunk_split_consistency(chunks: pd.DataFrame) -> pd.DataFrame:
    """Every chunk from a track should share the same split."""
    per_track = chunks.groupby("track_id")["split"].nunique()
    bad = per_track[per_track > 1]
    rows = [
        {
            "check": "chunk_split_consistency",
            "passed": len(bad) == 0,
            "violation_count": int(len(bad)),
            "sample_track_ids": ", ".join(str(t) for t in bad.index[:10]),
        }
    ]
    return pd.DataFrame(rows)


def chunk_position_by_split(chunks: pd.DataFrame) -> pd.DataFrame:
    if "chunk_idx" not in chunks.columns:
        return pd.DataFrame()
    counts = (
        chunks.groupby(["split", "chunk_idx"], observed=False)
        .size()
        .reset_index(name="count")
    )
    totals = chunks.groupby("split", observed=False).size().rename("split_total")
    counts = counts.merge(totals, on="split")
    counts["percent_of_split"] = counts["count"] / counts["split_total"] * 100
    return counts.sort_values(["split", "chunk_idx"]).reset_index(drop=True)


def _df_to_md_table(frame: pd.DataFrame) -> str:
    headers = "| " + " | ".join(frame.columns) + " |"
    sep = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([headers, sep, *rows])


def write_summary_markdown(
    output_dir: Path,
    song_counts: pd.DataFrame,
    chunk_counts: pd.DataFrame,
    overlap_song: pd.DataFrame,
    overlap_chunk: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    lines = [
        "# Split Audit Summary (Beatport Key Task)",
        "",
        "## Song-level counts (`key_splits.csv`)",
        "",
        _df_to_md_table(song_counts),
        "",
        "## Chunk-level counts (`clip_15s/key_chunks.csv`)",
        "",
        _df_to_md_table(chunk_counts),
        "",
        "## Leakage checks",
        "",
        _df_to_md_table(overlap_song),
        "",
        _df_to_md_table(overlap_chunk),
        "",
        "## Label coverage vs train (val/test keys unseen in train?)",
        "",
        _df_to_md_table(coverage),
        "",
    ]
    (output_dir / "split_audit_summary.md").write_text("\n".join(lines))


def maybe_plot_label_distribution(
    chunks: pd.DataFrame,
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    for col in ("mode", "key_24"):
        pivot = (
            chunks.groupby(["split", col], observed=False)
            .size()
            .unstack(fill_value=0)
            .reindex(["train", "val", "test"])
        )
        ax = pivot.T.plot(kind="bar", figsize=(10, 5), rot=45)
        ax.set_title(f"{col} distribution by split (chunk counts)")
        ax.set_ylabel("chunks")
        ax.legend(title="split")
        plt.tight_layout()
        plt.savefig(output_dir / f"{col}_by_split.png", dpi=120)
        plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Beatport key split audit for Assignment 5.")
    parser.add_argument("--clip-sec", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    datasets_dir = BEATPORT_DIR / "datasets"
    key_splits_path = datasets_dir / "key_splits.csv"
    key_chunks_path = datasets_dir / f"clip_{args.clip_sec}s" / "key_chunks.csv"

    for path in (key_splits_path, key_chunks_path):
        if not path.exists():
            print(f"Missing required file: {path}", file=sys.stderr)
            print("Run the beatport pipeline first: cd beatport && python run_pipeline.py", file=sys.stderr)
            return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    key_splits = read_table(key_splits_path)
    key_chunks = read_table(key_chunks_path)

    song_counts = split_counts(key_splits)
    chunk_counts = split_counts(key_chunks)

    key24_dist_song = label_distribution_by_split(key_splits, "key_24")
    mode_dist_song = label_distribution_by_split(key_splits, "mode")
    pitch_dist_song = label_distribution_by_split(key_splits, "pitch_class")

    key24_dist_chunk = label_distribution_by_split(key_chunks, "key_24")
    mode_dist_chunk = label_distribution_by_split(key_chunks, "mode")

    coverage = category_coverage_against_train(
        key_splits, ["key_24", "mode", "pitch_class"]
    )
    overlap_song = track_overlap_check(key_splits)
    overlap_chunk = chunk_split_consistency(key_chunks)
    chunk_pos = chunk_position_by_split(key_chunks)

    song_counts.to_csv(args.output_dir / "song_split_counts.csv", index=False)
    chunk_counts.to_csv(args.output_dir / "chunk_split_counts.csv", index=False)
    key24_dist_song.to_csv(args.output_dir / "song_key24_distribution_by_split.csv", index=False)
    mode_dist_song.to_csv(args.output_dir / "song_mode_distribution_by_split.csv", index=False)
    pitch_dist_song.to_csv(args.output_dir / "song_pitch_distribution_by_split.csv", index=False)
    key24_dist_chunk.to_csv(args.output_dir / "chunk_key24_distribution_by_split.csv", index=False)
    mode_dist_chunk.to_csv(args.output_dir / "chunk_mode_distribution_by_split.csv", index=False)
    coverage.to_csv(args.output_dir / "category_coverage_vs_train.csv", index=False)
    overlap_song.to_csv(args.output_dir / "leakage_check_song.csv", index=False)
    overlap_chunk.to_csv(args.output_dir / "leakage_check_chunks.csv", index=False)
    if not chunk_pos.empty:
        chunk_pos.to_csv(args.output_dir / "chunk_position_by_split.csv", index=False)

    metadata = {
        "clip_sec": args.clip_sec,
        "n_songs": int(len(key_splits)),
        "n_chunks": int(len(key_chunks)),
        "split_ratios_config": "70/15/15 song-level, stratified by key_24, seed 42",
        "song_counts": song_counts.to_dict(orient="records"),
        "chunk_counts": chunk_counts.to_dict(orient="records"),
        "leakage_song_passed": bool(overlap_song["passed"].iloc[0]),
        "leakage_chunk_passed": bool(overlap_chunk["passed"].iloc[0]),
        "unseen_key24_in_val": int(
            coverage.loc[
                (coverage["column"] == "key_24") & (coverage["split"] == "val"),
                "unseen_values_count",
            ].iloc[0]
        ),
        "unseen_key24_in_test": int(
            coverage.loc[
                (coverage["column"] == "key_24") & (coverage["split"] == "test"),
                "unseen_values_count",
            ].iloc[0]
        ),
    }
    (args.output_dir / "split_audit_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    write_summary_markdown(
        args.output_dir,
        song_counts,
        chunk_counts,
        overlap_song,
        overlap_chunk,
        coverage,
    )
    maybe_plot_label_distribution(key_chunks, args.output_dir)

    print(f"Wrote split audit artifacts to {args.output_dir}")
    print("\nSong counts:")
    print(song_counts.to_string(index=False))
    print("\nChunk counts:")
    print(chunk_counts.to_string(index=False))
    print("\nLeakage checks:")
    print(overlap_song.to_string(index=False))
    print(overlap_chunk.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
