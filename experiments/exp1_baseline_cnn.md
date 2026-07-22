# Experiment Report: baseline_cnn (Exp 1)

> **Reproducibility tag:** `seed=42` · device `mps` · clip `15s`  
> Local run evidence: `outputs/cnn_runs/baseline_cnn_*`

## One-change experiment log

| Field | Value |
| --- | --- |
| run_id | `baseline_cnn` |
| seed | **42** |
| change | First neural key run: stereo raw waveform + small 1D CNN, AdamW `lr=1e-3` |
| hypothesis | Stereo PCM + a small 1D CNN at `lr=1e-3` should learn a non-trivial 24-class key mapping on held-out songs. |
| expected_signal | Train/val loss should fall below ~3.18 (random 24-class CE); test accuracy clearly above ~4% random / ~10% majority. |
| observed_signal | underfitting_near_random — loss flat ~3.0–3.3; test acc **2.2%**, macro F1 **0.01** |
| fixed_conditions | song-level split, 15s clips, architecture, batch 16, weight decay 0, seed 42 |
| limitation | single_baseline_before_intervention |

## Hypothesis (pre-run)

**Lesson grounding:** Week 5 trustworthy evaluation — establish a baseline learning curve before claiming generalization.

**Hypothesis:** Stereo raw waveforms + a small 1D CNN with AdamW `lr=1e-3` should learn a non-trivial 24-class key mapping on cached Beatport chunks.

**Expected behavior:** Loss curves trend down; validation accuracy rises above chance; test metrics beat majority (~10%) and random (~4%).

## Setup

| Setting | Value |
| --- | --- |
| Input | 15s stereo PCM @ 44.1 kHz `(2, 661500)` |
| Model | `KeyWaveformCNN` |
| Optimizer | AdamW, `lr=1e-3`, `weight_decay=0` |
| Batch size | 16 |
| Epochs | 30 (early stop patience 8) |
| Device | MPS |
| Splits | Song-level 70/15/15 (6285 / 1350 / 1334 chunks) |

## Command

```bash
cd beatport && source .venv/bin/activate
PYTHONUNBUFFERED=1 python ../assignment_5/train_key_cnn.py \
  --run-name baseline_cnn \
  --batch-log-interval 20
```

## Artifacts

| File | Purpose |
| --- | --- |
| `outputs/cnn_runs/baseline_cnn_history.csv` | Per-epoch metrics |
| `outputs/cnn_runs/baseline_cnn_metadata.json` | Config + test scores + command |
| `outputs/cnn_runs/baseline_cnn.log` | Full training log |
| `outputs/cnn_runs/plots/loss_learning_curve.png` | Train vs val loss |
| `outputs/cnn_runs/plots/validation_metric_learning_curve.png` | Val acc / macro F1 |

## Observed behavior

| Metric | Value |
| --- | --- |
| Epochs ran | 10 (early stop) |
| Best val loss | 3.183 (epoch 2) |
| Test accuracy | **2.2%** |
| Test macro F1 | **0.01** |
| Train / val loss | ~3.0–3.3 (flat; ≈ random CE ≈ 3.18) |

**Vs expected:** Failed. No sustained learning — curves are flat, not exploding.

## Diagnosis

- **Symptom:** Loss stuck near random-chance cross-entropy; accuracy at/below chance.
- **Not classic “LR too high”:** No divergence or NaNs — loss is *too stable*.
- **Likely causes to test next:** LR poorly matched (Part 04 range test), and/or raw waveform is the wrong representation for key.
- **Next:** Exp 2 — LR range test, then full run with **only** learning rate changed.
