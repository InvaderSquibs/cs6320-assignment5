#!/usr/bin/env python3
"""Part 04 learning-rate range test for Beatport key chroma MLP.

Same diagnostic as run_lr_range_test.py (Exp 2), but on cached chroma CQT vectors.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
BEATPORT_DIR = REPO_ROOT / "beatport"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "lr_range_tests"

sys.path.insert(0, str(BEATPORT_DIR))

from chroma_features import CHROMA_INPUT_DIM, filter_chunks_with_chroma  # noqa: E402
from config_loader import load_config  # noqa: E402
from dataset import BeatportKeyChromaDataset, collate_chroma_batch  # noqa: E402
from device import log_device_detection, pin_memory_for_device, resolve_device  # noqa: E402
from models.key_chroma_mlp import KeyChromaMLP  # noqa: E402
from paths import key_chroma_dir, key_chunks_path  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def build_train_loader(
    chunks: pd.DataFrame,
    chroma_dir: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    train_df = chunks[chunks["split"] == "train"]
    encoder = LabelEncoder()
    encoder.fit(train_df["key_24"])
    dataset = BeatportKeyChromaDataset(train_df, chroma_dir, encoder)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_chroma_batch,
    )
    return loader, encoder, len(train_df)


def run_lr_range_test(
    *,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    start_lr: float,
    end_lr: float,
    max_batches: int | None,
) -> pd.DataFrame:
    num_steps = len(train_loader) if max_batches is None else min(max_batches, len(train_loader))
    lr_mult = (end_lr / start_lr) ** (1 / max(num_steps - 1, 1))
    lr = start_lr
    rows: list[dict] = []

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr)

    for step, batch in enumerate(train_loader, start=1):
        if max_batches is not None and step > max_batches:
            break

        for group in optimizer.param_groups:
            group["lr"] = lr

        features = batch["features"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        rows.append({"step": step, "lr": lr, "loss": loss_val, "finite": np.isfinite(loss_val)})

        if step % 20 == 0 or step == 1:
            log(f"  step {step}/{num_steps}  lr={lr:.2e}  loss={loss_val:.4f}")

        if not np.isfinite(loss_val):
            log(f"  stopped at step {step}: non-finite loss")
            break

        lr *= lr_mult

    return pd.DataFrame(rows)


def suggest_learning_rate(history: pd.DataFrame) -> dict:
    finite = history[history["finite"]].copy()
    if finite.empty:
        return {"suggested_lr": None, "reason": "no finite loss values"}

    finite = finite.reset_index(drop=True)
    min_idx = int(finite["loss"].idxmin())
    min_lr = float(finite.loc[min_idx, "lr"])
    min_loss = float(finite.loc[min_idx, "loss"])

    if len(finite) >= 5:
        rolling = finite["loss"].diff(4)
        steepest_idx = int(rolling.idxmin())
        steepest_lr = float(finite.loc[steepest_idx, "lr"])
    else:
        steepest_lr = min_lr

    suggested = max(steepest_lr / 10.0, float(finite["lr"].min()))

    return {
        "suggested_lr": suggested,
        "min_loss_lr": min_lr,
        "min_loss": min_loss,
        "steepest_drop_lr": steepest_lr,
        "reason": (
            "Use range test as candidate generator (Part 04 p.19). "
            f"Suggested starting point ≈ {suggested:.2e} (order of magnitude below "
            f"steepest-drop region ~{steepest_lr:.2e})."
        ),
    }


def plot_lr_range(history: pd.DataFrame, out_path: Path, suggestion: dict) -> None:
    finite = history[history["finite"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(finite["lr"], finite["loss"], linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Mini-batch loss")
    ax.set_title("Part 04 LR range test (chroma MLP) — loss vs learning rate")
    ax.grid(True, which="both", alpha=0.3)

    sug = suggestion.get("suggested_lr")
    if sug:
        ax.axvline(sug, color="green", linestyle="--", label=f"suggested ≈ {sug:.1e}")
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Part 04 LR range test for key chroma MLP.")
    parser.add_argument("--run-name", default="exp3_chroma_lr_range")
    parser.add_argument("--clip-sec", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--start-lr", type=float, default=1e-6)
    parser.add_argument("--end-lr", type=float, default=1e-1)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    data_dir = Path(cfg["output_dir"])
    if not data_dir.is_absolute():
        data_dir = BEATPORT_DIR / data_dir

    chunks = pd.read_csv(key_chunks_path(data_dir, args.clip_sec))
    chroma_dir = key_chroma_dir(data_dir, args.clip_sec)
    chunks = filter_chunks_with_chroma(chunks, chroma_dir)

    log_device_detection(log)
    device = resolve_device(force_cpu=args.cpu, preference=args.device)
    pin_memory = pin_memory_for_device(device)
    log(f"Selected device: {device}")
    log(f"Loaded {len(chunks)} chunks with cached chroma from {chroma_dir}")

    torch.manual_seed(args.seed)
    train_loader, encoder, n_train = build_train_loader(
        chunks, chroma_dir, args.batch_size, args.num_workers, pin_memory
    )

    n_classes = len(encoder.classes_)
    model = KeyChromaMLP(
        n_classes=n_classes,
        input_dim=CHROMA_INPUT_DIM,
        hidden_dim=args.hidden_dim,
    ).to(device)

    class_weights = compute_class_weight(
        "balanced",
        classes=np.arange(n_classes),
        y=encoder.transform(chunks[chunks["split"] == "train"]["key_24"]),
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    n_steps = len(train_loader) if args.max_batches is None else min(args.max_batches, len(train_loader))
    log(
        f"LR range test (chroma MLP): {args.start_lr:.1e} → {args.end_lr:.1e} over "
        f"{n_steps} batches (n_train={n_train}, input_dim={CHROMA_INPUT_DIM})"
    )

    history = run_lr_range_test(
        model=model,
        train_loader=train_loader,
        device=device,
        loss_fn=loss_fn,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        max_batches=args.max_batches,
    )

    suggestion = suggest_learning_rate(history)
    csv_path = out_dir / f"{args.run_name}_lr_range.csv"
    history.to_csv(csv_path, index=False)

    plot_path = out_dir / f"{args.run_name}_lr_range.png"
    plot_lr_range(history, plot_path, suggestion)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "model": "KeyChromaMLP",
        "input_dim": CHROMA_INPUT_DIM,
        "hidden_dim": args.hidden_dim,
        "part04_reference": "assignment_4/part040-000-training-neural-networks-debugging.pdf p.21",
        "start_lr": args.start_lr,
        "end_lr": args.end_lr,
        "n_steps": len(history),
        "device": str(device),
        "suggestion": suggestion,
        "csv": str(csv_path),
        "plot": str(plot_path),
    }
    meta_path = out_dir / f"{args.run_name}_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    log(f"Wrote {csv_path}")
    log(f"Wrote {plot_path}")
    log(f"Wrote {meta_path}")
    log(f"Suggested LR for chroma baseline run: {suggestion.get('suggested_lr')}")
    log(suggestion.get("reason", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
