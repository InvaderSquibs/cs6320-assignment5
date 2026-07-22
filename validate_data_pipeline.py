#!/usr/bin/env python3
"""Validate Beatport splits, audio paths, spectrograms, and PyTorch waveform samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

REPO_ROOT = Path(__file__).resolve().parents[1]
BEATPORT_DIR = REPO_ROOT / "beatport"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "data_validation"

sys.path.insert(0, str(BEATPORT_DIR))

from config_loader import load_config  # noqa: E402
from waveforms import (  # noqa: E402
    expected_waveform_samples,
    filter_chunks_with_waveforms,
    load_waveform_for_chunk,
)
from paths import key_chunks_path, key_waveforms_dir  # noqa: E402


def split_summary(chunks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        sub = chunks[chunks["split"] == split]
        rows.append(
            {
                "split": split,
                "n_chunks": len(sub),
                "n_segments": sub["segment_id"].nunique()
                if "segment_id" in sub.columns
                else sub["track_id"].nunique(),
                "n_tracks": sub["track_id"].nunique(),
                "n_keys": sub["key_24"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def pick_listen_samples(chunks: pd.DataFrame, per_split: int, seed: int) -> pd.DataFrame:
    picks = []
    for split in ("train", "val", "test"):
        sub = chunks[chunks["split"] == split].copy()
        sub = sub.sample(min(per_split, len(sub)), random_state=seed + hash(split) % 1000)
        picks.append(sub)
    return pd.concat(picks, ignore_index=True)


def write_listen_manifest(samples: pd.DataFrame, out_dir: Path) -> None:
    cols = [
        "split",
        "track_id",
        "chunk_idx",
        "start_sec",
        "end_sec",
        "key_24",
        "segment_id",
        "audio_path",
    ]
    manifest = samples[cols].copy()
    manifest["audio_exists"] = manifest["audio_path"].apply(lambda p: Path(p).exists())
    manifest.to_csv(out_dir / "listen_samples.csv", index=False)

    html_lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Beatport split listen samples</title></head><body>",
        "<h1>Beatport key chunks — listen by split</h1>",
        "<p>Open this file in a browser. Paths are local MP3 previews from the dataset.</p>",
    ]
    for split in ("train", "val", "test"):
        html_lines.append(f"<h2>{split}</h2><ul>")
        for _, row in manifest[manifest["split"] == split].iterrows():
            path = Path(row["audio_path"])
            label = (
                f"{row['key_24']} | track {row['track_id']} chunk {row['chunk_idx']} "
                f"({row['start_sec']:.0f}s–{row['end_sec']:.0f}s)"
            )
            if path.exists():
                uri = path.as_uri()
                html_lines.append(
                    f"<li><strong>{label}</strong><br>"
                    f"<audio controls preload='none' src='{uri}'></audio><br>"
                    f"<code>{path}</code></li>"
                )
            else:
                html_lines.append(f"<li><strong>{label}</strong> — missing file</li>")
        html_lines.append("</ul>")
    html_lines.append("</body></html>")
    (out_dir / "listen_samples.html").write_text("\n".join(html_lines))


def plot_spectrogram(row: pd.Series, cfg: dict, out_path: Path) -> None:
    sample_rate = int(cfg["sample_rate"])
    channels = int(cfg.get("channels", 2))
    mono = channels == 1
    duration = row["end_sec"] - row["start_sec"]
    try:
        y, _ = librosa.load(
            row["audio_path"],
            sr=sample_rate,
            mono=mono,
            offset=row["start_sec"],
            duration=duration,
            res_type="kaiser_fast",
        )
    except Exception:
        return

    if y.ndim > 1:
        y = librosa.to_mono(y.T)

    mel = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sample_rate)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        x_axis="time",
        y_axis="mel",
        ax=axes[0],
        cmap="magma",
    )
    axes[0].set_title(
        f"Mel spectrogram — {row['split']} | {row['key_24']} | "
        f"track {row['track_id']} chunk {row['chunk_idx']} "
        f"({row['start_sec']:.0f}–{row['end_sec']:.0f}s)"
    )
    librosa.display.specshow(
        chroma,
        sr=sample_rate,
        x_axis="time",
        y_axis="chroma",
        ax=axes[1],
        cmap="coolwarm",
    )
    axes[1].set_title("Chroma CQT (visual QA only — model input is raw waveform)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def build_pytorch_sample_records(
    chunks: pd.DataFrame,
    waveform_dir: Path,
    sample_rows: pd.DataFrame,
    cfg: dict,
) -> list[dict]:
    train = chunks[chunks["split"] == "train"]
    encoder = LabelEncoder()
    encoder.fit(train["key_24"])

    sample_rate = int(cfg["sample_rate"])
    channels = int(cfg.get("channels", 2))

    records = []
    for _, row in sample_rows.iterrows():
        waveform = load_waveform_for_chunk(row, waveform_dir)
        class_idx = int(encoder.transform([row["key_24"]])[0])
        tensor = torch.from_numpy(waveform)
        expected_samples = expected_waveform_samples(row["clip_sec"], sample_rate)

        records.append(
            {
                "split": row["split"],
                "track_id": int(row["track_id"]),
                "chunk_idx": int(row["chunk_idx"]),
                "segment_id": row.get("segment_id"),
                "audio_path": row["audio_path"],
                "clip_sec": float(row["clip_sec"]),
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
                "label": {
                    "key_24": row["key_24"],
                    "mode": row["mode"],
                    "pitch_class": float(row["pitch_class"]),
                    "class_index_for_pytorch": class_idx,
                    "all_classes": encoder.classes_.tolist(),
                },
                "cached_waveform": {
                    "shape": list(waveform.shape),
                    "dtype": str(waveform.dtype),
                    "description": f"peak-normalized stereo PCM @ {sample_rate} Hz",
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "samples_per_channel": int(waveform.shape[1]),
                    "expected_samples_per_channel": expected_samples,
                    "min": round(float(waveform.min()), 6),
                    "max": round(float(waveform.max()), 6),
                    "mean": round(float(waveform.mean()), 6),
                    "left_channel_preview_first_8": [
                        round(float(v), 6) for v in waveform[0, :8].tolist()
                    ],
                    "right_channel_preview_first_8": [
                        round(float(v), 6) for v in waveform[1, :8].tolist()
                    ]
                    if waveform.shape[0] > 1
                    else [],
                },
                "pytorch_input": {
                    "tensor_shape": list(tensor.shape),
                    "tensor_dtype": str(tensor.dtype),
                    "note": f"CNN receives shape (batch, {channels}, {expected_samples}) — raw waveform",
                },
                "sample_weight": float(row["sample_weight"]),
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-sec", type=int, default=15)
    parser.add_argument("--listen-per-split", type=int, default=3)
    parser.add_argument("--spectrogram-samples", type=int, default=6)
    parser.add_argument("--pytorch-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    out_dir = Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = BEATPORT_DIR / out_dir

    chunks = pd.read_csv(key_chunks_path(out_dir, args.clip_sec))
    waveform_dir = key_waveforms_dir(out_dir, args.clip_sec)
    chunks = filter_chunks_with_waveforms(chunks, waveform_dir)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = split_summary(chunks)
    summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)

    listen = pick_listen_samples(chunks, args.listen_per_split, args.seed)
    write_listen_manifest(listen, OUTPUT_DIR)

    spec_rows = chunks.sample(
        min(args.spectrogram_samples, len(chunks)), random_state=args.seed
    )
    spec_dir = OUTPUT_DIR / "spectrograms"
    spec_dir.mkdir(exist_ok=True)
    for _, row in spec_rows.iterrows():
        out = spec_dir / (
            f"{row['split']}_track{row['track_id']}_chunk{row['chunk_idx']}_"
            f"{row['key_24'].replace(' ', '_')}.png"
        )
        plot_spectrogram(row, cfg, out)

    pytorch_rows = chunks.sample(
        min(args.pytorch_samples, len(chunks)), random_state=args.seed + 1
    )
    sample_records = build_pytorch_sample_records(chunks, waveform_dir, pytorch_rows, cfg)
    (OUTPUT_DIR / "pytorch_sample_records.json").write_text(
        json.dumps(sample_records, indent=2)
    )

    sample_rate = int(cfg["sample_rate"])
    channels = int(cfg.get("channels", 2))
    expected = expected_waveform_samples(args.clip_sec, sample_rate)
    readme = f"""# Data validation artifacts

## Splits
See `split_summary.csv` — chunk/segment/track counts per train/val/test.

## Listen manually
- `listen_samples.csv` — paths and labels for {args.listen_per_split} clips per split
- `listen_samples.html` — open in a browser to play MP3 segments

## Visual audio check
`spectrograms/` — mel + chroma plots for {args.spectrogram_samples} random chunks (QA only)

## PyTorch input check
`pytorch_sample_records.json` — for each sample:
- row metadata (split, key, segment, time range)
- cached stereo waveform stats ({channels} x {expected} float32 PCM @ {sample_rate} Hz)
- label class index and tensor shape the CNN sees

This matches `assignment_5/train_key_cnn.py`.
"""
    (OUTPUT_DIR / "README.md").write_text(readme)

    print(f"Wrote artifacts to {OUTPUT_DIR}")
    print("\nSplit summary:")
    print(summary.to_string(index=False))
    print(f"\nOpen for listening: {OUTPUT_DIR / 'listen_samples.html'}")
    print(f"Spectrograms: {spec_dir}")
    print(f"PyTorch samples: {OUTPUT_DIR / 'pytorch_sample_records.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
