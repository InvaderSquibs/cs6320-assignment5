# Experiment Report: exp2_lr_range + exp2_lr_tuned (Exp 2)

> **Reproducibility tag:** `seed=42` · device `mps` · clip `15s`  
> **Primary generalization intervention** for Assignment 5 (waveform path).  
> Local run evidence: `outputs/lr_range_tests/exp2_lr_range/`, `outputs/cnn_runs/exp2_lr_tuned_*`

## One-change experiment log

| Field | Value |
| --- | --- |
| run_id | `exp2_lr_tuned` (after `exp2_lr_range`) |
| seed | **42** |
| change | Learning rate only: `1e-3` → Part 04 range-test suggestion `≈2e-3` |
| hypothesis | Exp 1 sat at random CE because `lr=1e-3` was poorly matched; a range-test-guided LR should unlock learning on the same waveform CNN. |
| expected_signal | Train loss drops below ~3.0 early; val accuracy climbs above chance and keeps improving vs Exp 1. |
| observed_signal | mild_metric_bump_still_near_random — test acc 2.2% → **3.9%**; loss still flat ~3.0–3.2 |
| fixed_conditions | architecture, splits, batch size, weight decay, dropout, seed, data |
| limitation | single_intervention_only (LR); representation held fixed |

## Hypothesis (pre-run)

**Lesson grounding:** Part 04 Learning-Rate Range Test (PDF p.19–21) — diagnose LR before architecture/regularization changes.

**Hypothesis:** Exp 1 stayed near random because the learning rate was wrong. A one-pass range test (`1e-6` → `1e-1`) will find a promising region; retraining with only that LR changed will produce clear downward loss and above-chance accuracy.

**Expected behavior:** Tuned run beats Exp 1 on val loss trend and test accuracy; curves show sustained learning rather than a flat ~3.18 CE line.

## Step 1 — LR range test

```bash
cd beatport && source .venv/bin/activate
PYTHONUNBUFFERED=1 python ../assignment_5/run_lr_range_test.py \
  --run-name exp2_lr_range
```

| Artifact | Path |
| --- | --- |
| Per-step CSV | `outputs/lr_range_tests/exp2_lr_range/exp2_lr_range_lr_range.csv` |
| Chart | `outputs/lr_range_tests/exp2_lr_range/exp2_lr_range_lr_range.png` |
| Metadata | `outputs/lr_range_tests/exp2_lr_range/exp2_lr_range_metadata.json` |

**Range-test finding:** Loss stayed ~3.1–3.3 until LR ≈ `2e-2` (brief min loss **2.89**). Script suggested fixed LR **`≈2.11e-3`** (≈ one decade below steepest-drop region).

## Step 2 — Full run at suggested LR

```bash
PYTHONUNBUFFERED=1 python ../assignment_5/train_key_cnn.py \
  --run-name exp2_lr_tuned \
  --learning-rate 0.002 \
  --batch-log-interval 20 \
  --epochs 30
```

## Observed behavior (vs Exp 1)

| Metric | Exp 1 `lr=1e-3` | Exp 2 `lr=2e-3` |
| --- | --- | --- |
| Epochs (early stop) | 10 | 10 |
| Best val loss | 3.183 | 3.183 (epoch 2) |
| Test accuracy | 2.2% | **3.9%** |
| Test macro F1 | 0.010 | 0.015 |
| Loss trend | Flat ~3.0–3.3 | Still flat ~3.0–3.2 |

**Vs expected:** Partially contradicted. Tiny accuracy bump, but **no sustained learning**. Intervention did not fix the failure mode.

## Diagnosis

- Range test was useful evidence (`1e-3` likely too low; dip near `~2e-2`).
- Full run at `2e-3` was still too weak (or LR is not the binding constraint).
- **Conclusion for Assignment 5:** LR alone on raw waveforms is an insufficient generalization intervention. Next staged change = **representation** (Exp 3 chroma), not another LR sweep.

## Artifacts

| File | Purpose |
| --- | --- |
| `outputs/cnn_runs/exp2_lr_tuned_history.csv` | Epoch metrics |
| `outputs/cnn_runs/exp2_lr_tuned_metadata.json` | Config + test scores + command |
| `outputs/cnn_runs/exp2_lr_tuned.log` | Training log |
| `outputs/cnn_runs/plots/` | Learning curves (with Exp 1 when plotted together) |
