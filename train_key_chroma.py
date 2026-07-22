#!/usr/bin/env python3
"""Train a small MLP on cached chroma CQT vectors with per-epoch history CSV."""

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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
BEATPORT_DIR = REPO_ROOT / "beatport"
ASSIGNMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ASSIGNMENT_DIR / "outputs" / "chroma_runs"

sys.path.insert(0, str(BEATPORT_DIR))

from chroma_features import (  # noqa: E402
    CHROMA_INPUT_DIM,
    CHROMA_RICH_INPUT_DIM,
    filter_chunks_with_chroma,
    load_chroma_matrix,
)
from config_loader import load_config  # noqa: E402
from dataset import BeatportKeyChromaDataset, collate_chroma_batch  # noqa: E402
from device import log_device_detection, pin_memory_for_device, resolve_device  # noqa: E402
from models.key_chroma_mlp import KeyChromaMLP  # noqa: E402
from paths import key_chroma_dir, key_chroma_rich_dir, key_chunks_path  # noqa: E402

_LOG_FILE: Path | None = None


def log(msg: str) -> None:
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
    chroma_dir: Path,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    scaler: StandardScaler | None = None,
) -> SplitLoaders:
    encoder = LabelEncoder()
    train_df = chunks[chunks["split"] == "train"]
    encoder.fit(train_df["key_24"])

    def make_loader(split_name: str, shuffle: bool) -> DataLoader:
        subset = chunks[chunks["split"] == split_name]
        dataset = BeatportKeyChromaDataset(subset, chroma_dir, encoder, scaler=scaler)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_chroma_batch,
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
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(features)
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
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            logits = model(features)
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


def train_key_chroma(args: argparse.Namespace) -> dict:
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
    if args.feature_variant == "rich":
        chroma_dir = key_chroma_rich_dir(out_dir, args.clip_sec)
        input_dim = CHROMA_RICH_INPUT_DIM
    else:
        chroma_dir = key_chroma_dir(out_dir, args.clip_sec)
        input_dim = CHROMA_INPUT_DIM
    chunks = filter_chunks_with_chroma(chunks, chroma_dir)
    log(f"Loaded {len(chunks)} chunks with cached chroma from {chroma_dir}")

    device_info = log_device_detection(log)
    device = resolve_device(force_cpu=args.cpu, preference=args.device)
    pin_memory = pin_memory_for_device(device)
    log(f"  Selected device: {device}")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    scaler: StandardScaler | None = None
    if args.scale:
        train_df = chunks[chunks["split"] == "train"]
        X_train = load_chroma_matrix(train_df, chroma_dir)
        scaler = StandardScaler()
        scaler.fit(X_train)
        log(
            f"Fitted StandardScaler on train split ({len(train_df)} rows): "
            f"mean range [{scaler.mean_.min():.3f}, {scaler.mean_.max():.3f}]  "
            f"scale range [{scaler.scale_.min():.3f}, {scaler.scale_.max():.3f}]"
        )

    loaders = build_loaders(
        chunks,
        chroma_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        scaler=scaler,
    )

    n_classes = len(loaders.label_encoder.classes_)
    model = KeyChromaMLP(
        n_classes=n_classes,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    train_batches = len(loaders.train)
    val_batches = len(loaders.val)
    log(
        f"Device={device}  batch_size={args.batch_size}  hidden_dim={args.hidden_dim}  "
        f"train={loaders.n_train} ({train_batches} batches/epoch)  "
        f"val={loaders.n_val} ({val_batches} batches)  test={loaders.n_test}  "
        f"classes={n_classes}  input_dim={input_dim}  feature_variant={args.feature_variant}  "
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

    checkpoint_path: Path | None = None
    if best_state is not None:
        model.load_state_dict(best_state)
        checkpoint_path = run_dir / f"{args.run_name}_checkpoint.pt"
        checkpoint_payload = {
            "model_state_dict": best_state,
            "label_classes": list(loaders.label_encoder.classes_),
            "run_name": args.run_name,
            "clip_sec": args.clip_sec,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "input_dim": input_dim,
            "standardized": args.scale,
            "feature_variant": args.feature_variant,
            "best_val_loss": best_val_loss,
            "epochs_ran": len(history_rows),
        }
        if scaler is not None:
            checkpoint_payload["scaler_mean"] = scaler.mean_.tolist()
            checkpoint_payload["scaler_scale"] = scaler.scale_.tolist()
        torch.save(checkpoint_payload, checkpoint_path)
        log(f"Wrote {checkpoint_path}")

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

    history_path = run_dir / f"{args.run_name}_history.csv"
    pd.DataFrame(history_rows).to_csv(history_path, index=False)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "run_name": args.run_name,
        "clip_sec": args.clip_sec,
        "epochs_requested": args.epochs,
        "epochs_ran": len(history_rows),
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
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
            "feature_type": "chroma_cqt_rich" if args.feature_variant == "rich" else "chroma_cqt_mean_std",
            "input_dim": input_dim,
            "description": (
                "4x12 binned means + 12 stds + 24 K-S correlations"
                if args.feature_variant == "rich"
                else "12 pitch-class means + 12 stds from librosa chroma_cqt"
            ),
            "feature_variant": args.feature_variant,
            "standardized": args.scale,
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
        "checkpoint_pt": str(checkpoint_path) if best_state is not None else None,
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
    parser = argparse.ArgumentParser(description="Train Beatport key MLP on cached chroma vectors.")
    parser.add_argument("--clip-sec", type=int, default=15)
    parser.add_argument("--run-name", default="exp3_chroma_baseline")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-log-interval", type=int, default=50)
    parser.add_argument("--eval-log-interval", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--feature-variant", choices=("baseline", "rich"), default="baseline")
    parser.add_argument("--scale", action="store_true", help="Standardize chroma using train-split mean/std")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args()
    if args.verbose and args.quiet:
        parser.error("Use either --verbose or --quiet, not both.")
    if args.verbose:
        args.batch_log_interval = 1
        args.eval_log_interval = 1
    elif args.quiet:
        args.batch_log_interval = 0
        args.eval_log_interval = 0
    train_key_chroma(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
