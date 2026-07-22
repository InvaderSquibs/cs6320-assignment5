#!/usr/bin/env python3
"""Export cached .npy waveforms to WAV + HTML for manual listening."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
BEATPORT_DIR = REPO_ROOT / "beatport"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "waveform_listen"

sys.path.insert(0, str(BEATPORT_DIR))

from config_loader import load_config  # noqa: E402
from paths import key_chunks_path, key_waveforms_dir  # noqa: E402
from waveforms import filter_chunks_with_waveforms, load_waveform_for_chunk  # noqa: E402


def safe_key_slug(key_24: str) -> str:
    """Filesystem- and URL-safe key label (avoid '#' — breaks HTML audio src)."""
    return key_24.replace("#", "sharp").replace(" ", "_")


def wav_src_for_html(wav_file: str) -> str:
    """Percent-encode path segments so '#' and spaces never break <audio src>."""
    parts = wav_file.split("/")
    return "/".join(quote(part, safe="/") for part in parts)


def export_samples(clip_sec: int, per_split: int, seed: int) -> Path:
    cfg = load_config()
    out_dir = Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = BEATPORT_DIR / out_dir

    sample_rate = int(cfg["sample_rate"])
    chunks = pd.read_csv(key_chunks_path(out_dir, clip_sec))
    waveform_dir = key_waveforms_dir(out_dir, clip_sec)
    chunks = filter_chunks_with_waveforms(chunks, waveform_dir)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wav_dir = OUTPUT_DIR / "wav"
    wav_dir.mkdir(exist_ok=True)

    manifest_rows = []
    for split in ("train", "val", "test"):
        sub = chunks[chunks["split"] == split]
        picks = sub.sample(min(per_split, len(sub)), random_state=seed + hash(split) % 1000)
        for _, row in picks.iterrows():
            waveform = load_waveform_for_chunk(row, waveform_dir, mmap_mode=None)
            wav_name = (
                f"{split}_track{int(row['track_id'])}_chunk{int(row['chunk_idx'])}_"
                f"{safe_key_slug(str(row['key_24']))}.wav"
            )
            wav_path = wav_dir / wav_name
            sf.write(wav_path, waveform.T, sample_rate, subtype="PCM_16")

            npy_path = waveform_dir / f"{int(row['track_id'])}_{int(row['chunk_idx'])}.npy"
            manifest_rows.append(
                {
                    "split": split,
                    "track_id": int(row["track_id"]),
                    "chunk_idx": int(row["chunk_idx"]),
                    "key_24": row["key_24"],
                    "start_sec": float(row["start_sec"]),
                    "end_sec": float(row["end_sec"]),
                    "wav_file": f"wav/{wav_name}",
                    "npy_file": str(npy_path),
                    "npy_exists": npy_path.exists(),
                    "npy_bytes": npy_path.stat().st_size if npy_path.exists() else 0,
                    "shape": f"{waveform.shape[0]} x {waveform.shape[1]}",
                    "duration_sec": round(waveform.shape[1] / sample_rate, 2),
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUTPUT_DIR / "waveform_listen_manifest.csv", index=False)

    html_lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Cached waveform listen samples</title>",
        "<style>body{font-family:system-ui;max-width:900px;margin:2rem auto;padding:0 1rem}",
        "li{margin-bottom:1.5rem} code{font-size:0.85em}</style></head><body>",
        "<h1>Cached stereo waveforms (.npy → WAV)</h1>",
        "<p>These clips were loaded from <code>beatport/datasets/clip_15s/waveforms/key/*.npy</code>, "
        "not sliced from MP3 at play time. Each should be 15 s @ 44.1 kHz stereo.</p>",
    ]
    for split in ("train", "val", "test"):
        html_lines.append(f"<h2>{split}</h2><ul>")
        for row in manifest_rows:
            if row["split"] != split:
                continue
            label = (
                f"{row['key_24']} | track {row['track_id']} chunk {row['chunk_idx']} "
                f"({row['start_sec']:.0f}s–{row['end_sec']:.0f}s in preview) | "
                f"{row['duration_sec']}s cached | shape {row['shape']}"
            )
            html_lines.append(
                f"<li><strong>{label}</strong><br>"
                f"<audio controls preload='metadata' src='{wav_src_for_html(row['wav_file'])}'></audio><br>"
                f"<code>{row['npy_file']}</code> ({row['npy_bytes']:,} bytes)</li>"
            )
        html_lines.append("</ul>")
    html_lines.append("</body></html>")
    html_path = OUTPUT_DIR / "listen_cached_waveforms.html"
    html_path.write_text("\n".join(html_lines))
    return html_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-sec", type=int, default=15)
    parser.add_argument("--per-split", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    html_path = export_samples(args.clip_sec, args.per_split, args.seed)
    print(f"Wrote {html_path}")
    print(f"WAV files: {OUTPUT_DIR / 'wav'}")
    print(f"Manifest: {OUTPUT_DIR / 'waveform_listen_manifest.csv'}")
    print(f"\nOpen in browser:\n  {html_path.as_uri()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
