#!/usr/bin/env python3
"""Train a 1D CNN on cached stereo waveforms with per-epoch history CSV."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
BEATPORT_DIR = REPO_ROOT / "beatport"
ASSIGNMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ASSIGNMENT_DIR / "outputs" / "cnn_runs"

sys.path.insert(0, str(BEATPORT_DIR))

from config_loader import load_config  # noqa: E402
from dataset import BeatportKeyChunkDataset, collate_key_batch  # noqa: E402
from device import describe_devices, log_device_detection, pin_memory_for_device, resolve_device  # noqa: E402
from models.key_cnn import KeyWaveformCNN  # noqa: E402
from paths import key_chunks_path, key_waveforms_dir  # noqa: E402
from waveforms import filter_chunks_with_waveforms  # noqa: E402

_LOG_FILE: Path | None = None


def log(msg: str) -> None:
    """Print with flush; also append to run log file when configured."""
    print(msg, flush=True)
    if _LOG_FILE is not None:
        with _LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")


@dataclass
class SplitLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    label_encoder: LabelEncoder
    n_train: int
    n_val: int
    n_test: int


def build_loaders(
    chunks: pd.DataFrame,
    waveform_dir: Path,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> SplitLoaders:
    encoder = LabelEncoder()
    train_df = chunks[chunks["split"] == "train"]
    encoder.fit(train_df["key_24"])

    def make_loader(split_name: str, shuffle: bool) -> DataLoader:
        subset = chunks[chunks["split"] == split_name]
        dataset = BeatportKeyChunkDataset(subset, waveform_dir, encoder)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_key_batch,
        )

    train_loader = make_loader("train", shuffle=True)
    val_loader = make_loader("val", shuffle=False)
    test_loader = make_loader("test", shuffle=False)

    return SplitLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        label_encoder=encoder,
        n_train=len(train_df),
        n_val=len(chunks[chunks["split"] == "val"]),
        n_test=len(chunks[chunks["split"] == "test"]),
    )


def should_log_batch(batch_idx: int, n_batches: int, interval: int) -> bool:
    """Log first batch, last batch, and every N batches (interval=0 → never)."""
    if interval <= 0:
        return False
    return batch_idx == 1 or batch_idx == n_batches or batch_idx % interval == 0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    epoch: int,
    total_epochs: int,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_log_interval: int,
) -> float:
    model.train()
    losses: list[float] = []
    n_batches = len(loader)
    log(f"Epoch {epoch}/{total_epochs} — training ({n_batches} batches)")
    t0 = time.perf_counter()

    for batch_idx, batch in enumerate(loader, start=1):
        waveforms = batch["waveform"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(waveforms)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        loss_val = float(loss.item())
        losses.append(loss_val)

        if should_log_batch(batch_idx, n_batches, batch_log_interval):
            elapsed = time.perf_counter() - t0
            pct = 100.0 * batch_idx / n_batches
            log(
                f"  [train] epoch {epoch}/{total_epochs}  "
                f"batch {batch_idx}/{n_batches} ({pct:.0f}%)  "
                f"loss={loss_val:.4f}  elapsed={elapsed:.1f}s"
            )

    avg_loss = float(np.mean(losses)) if losses else 0.0
    log(
        f"Epoch {epoch}/{total_epochs} — train done  "
        f"mean_loss={avg_loss:.4f}  elapsed={time.perf_counter() - t0:.1f}s"
    )
    return avg_loss


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    *,
    split_name: str,
    epoch: int,
    total_epochs: int,
    device: torch.device,
    loss_fn: nn.Module,
    eval_log_interval: int,
    quiet: bool = False,
) -> tuple[float, float, float]:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    n_batches = len(loader)
    if not quiet:
        log(f"Epoch {epoch}/{total_epochs} — evaluating {split_name} ({n_batches} batches)")
    t0 = time.perf_counter()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            waveforms = batch["waveform"].to(device)
            labels = batch["label"].to(device)
            logits = model(waveforms)
            loss = loss_fn(logits, labels)
            losses.append(float(loss.item()))
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.cpu().numpy().tolist())

            if should_log_batch(batch_idx, n_batches, eval_log_interval):
                pct = 100.0 * batch_idx / n_batches
                log(
                    f"  [{split_name}] epoch {epoch}/{total_epochs}  "
                    f"batch {batch_idx}/{n_batches} ({pct:.0f}%)  "
                    f"loss={float(loss.item()):.4f}"
                )

    avg_loss = float(np.mean(losses)) if losses else 0.0
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    log(
        f"Epoch {epoch}/{total_epochs} — {split_name}  "
        f"loss={avg_loss:.4f}  acc={acc:.3f}  macro_f1={macro_f1:.3f}  "
        f"elapsed={time.perf_counter() - t0:.1f}s"
    )
    return avg_loss, acc, macro_f1


def train_key_cnn(args: argparse.Namespace) -> dict:
    global _LOG_FILE

    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_file) if args.log_file else run_dir / f"{args.run_name}.log"
    log_path.write_text("", encoding="utf-8")
    _LOG_FILE = log_path

    log(f"Run {args.run_name!r} starting — loading data...")
    log(f"Training log file: {log_path}")
    cfg = load_config()
    out_dir = Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = BEATPORT_DIR / out_dir

    chunks = pd.read_csv(key_chunks_path(out_dir, args.clip_sec))
    waveform_dir = key_waveforms_dir(out_dir, args.clip_sec)
    chunks = filter_chunks_with_waveforms(chunks, waveform_dir)
    log(f"Loaded {len(chunks)} chunks with cached waveforms from {waveform_dir}")

    device_info = log_device_detection(log)
    device = resolve_device(force_cpu=args.cpu, preference=args.device)
    pin_memory = pin_memory_for_device(device)
    log(f"  Selected device: {device}")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    loaders = build_loaders(
        chunks,
        waveform_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    n_classes = len(loaders.label_encoder.classes_)
    in_channels = int(cfg.get("channels", 2))
    model = KeyWaveformCNN(
        n_classes=n_classes,
        in_channels=in_channels,
        dropout=args.dropout,
    ).to(device)

    train_batches = len(loaders.train)
    val_batches = len(loaders.val)
    log(
        f"Device={device}  batch_size={args.batch_size}  "
        f"train={loaders.n_train} ({train_batches} batches/epoch)  "
        f"val={loaders.n_val} ({val_batches} batches)  test={loaders.n_test}  "
        f"classes={n_classes}  "
        f"batch_log_interval={args.batch_log_interval}  "
        f"eval_log_interval={args.eval_log_interval}"
    )
    log(f"Training for up to {args.epochs} epochs (patience={args.patience})")

    class_weights = compute_class_weight(
        "balanced",
        classes=np.arange(n_classes),
        y=loaders.label_encoder.transform(chunks[chunks["split"] == "train"]["key_24"]),
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    best_state: dict | None = None
    epochs_without_improvement = 0
    history_rows: list[dict] = []
    run_t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            model,
            loaders.train,
            epoch=epoch,
            total_epochs=args.epochs,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
            batch_log_interval=args.batch_log_interval,
        )
        train_eval_loss, train_acc, train_f1 = evaluate_loader(
            model,
            loaders.train,
            split_name="train",
            epoch=epoch,
            total_epochs=args.epochs,
            device=device,
            loss_fn=loss_fn,
            eval_log_interval=args.eval_log_interval,
            quiet=True,
        )
        val_loss, val_acc, val_f1 = evaluate_loader(
            model,
            loaders.val,
            split_name="val",
            epoch=epoch,
            total_epochs=args.epochs,
            device=device,
            loss_fn=loss_fn,
            eval_log_interval=args.eval_log_interval,
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_eval_loss,
                "validation_loss": val_loss,
                "train_accuracy": train_acc,
                "validation_accuracy": val_acc,
                "train_macro_f1": train_f1,
                "validation_macro_f1": val_f1,
                "run": args.run_name,
            }
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        log(
            f"=== Epoch {epoch}/{args.epochs} summary ===  "
            f"train_loss={train_eval_loss:.4f}  val_loss={val_loss:.4f}  "
            f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  "
            f"val_f1={val_f1:.3f}  "
            f"best_val={'yes' if improved else 'no'}  "
            f"no_improve={epochs_without_improvement}/{args.patience or 'off'}"
        )

        if args.patience is not None and epochs_without_improvement >= args.patience:
            log(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    log("Evaluating test split with best val checkpoint...")
    test_loss, test_acc, test_f1 = evaluate_loader(
        model,
        loaders.test,
        split_name="test",
        epoch=args.epochs,
        total_epochs=args.epochs,
        device=device,
        loss_fn=loss_fn,
        eval_log_interval=args.eval_log_interval,
    )

    run_dir = Path(args.output_dir)
    history_path = run_dir / f"{args.run_name}_history.csv"
    pd.DataFrame(history_rows).to_csv(history_path, index=False)

    sample_rate = int(cfg.get("sample_rate", 44100))
    channels = int(cfg.get("channels", 2))
    expected_samples = int(round(args.clip_sec * sample_rate))

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "run_name": args.run_name,
        "clip_sec": args.clip_sec,
        "epochs_requested": args.epochs,
        "epochs_ran": len(history_rows),
        "batch_size": args.batch_size,
        "batch_log_interval": args.batch_log_interval,
        "eval_log_interval": args.eval_log_interval,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "seed": args.seed,
        "device": str(device),
        "device_detection": device_info,
        "device_preference": args.device,
        "input": {
            "sample_rate": sample_rate,
            "channels": channels,
            "samples_per_channel": expected_samples,
            "tensor_shape": [channels, expected_samples],
        },
        "n_train": loaders.n_train,
        "n_val": loaders.n_val,
        "n_test": loaders.n_test,
        "n_classes": n_classes,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "history_csv": str(history_path),
        "training_log": str(log_path),
        "total_elapsed_sec": round(time.perf_counter() - run_t0, 1),
    }
    meta_path = run_dir / f"{args.run_name}_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    log(f"Wrote {history_path}")
    log(f"Wrote {meta_path}")
    log(f"Wrote {log_path}")
    log(
        f"Done in {metadata['total_elapsed_sec']}s — "
        f"test acc={test_acc:.3f} macro_f1={test_f1:.3f} loss={test_loss:.3f}"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Beatport key CNN on cached waveforms.")
    parser.add_argument("--clip-sec", type=int, default=15)
    parser.add_argument("--run-name", default="baseline_cnn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--batch-log-interval",
        type=int,
        default=50,
        help="Training batch check-ins every N batches (also logs batch 1 and last). 0=off.",
    )
    parser.add_argument(
        "--eval-log-interval",
        type=int,
        default=0,
        help="Eval batch check-ins every N batches. 0=epoch result lines only (default).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log every train and eval batch (equivalent to --batch-log-interval 1 --eval-log-interval 1).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="No batch check-ins — epoch summaries only.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Training device: auto (CUDA > MPS > CPU), or force cpu/cuda/mps.",
    )
    parser.add_argument("--cpu", action="store_true", help="Shortcut for --device cpu")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path for full training log (default: outputs/cnn_runs/{run_name}.log).",
    )
    args = parser.parse_args()
    if args.verbose and args.quiet:
        parser.error("Use either --verbose or --quiet, not both.")
    if args.verbose:
        args.batch_log_interval = 1
        args.eval_log_interval = 1
    elif args.quiet:
        args.batch_log_interval = 0
        args.eval_log_interval = 0
    train_key_cnn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
