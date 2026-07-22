# Experiment Report: exp3_chroma_baseline (Exp 3)

> **Reproducibility tag:** `seed=42` · device `mps` · clip `15s`  
> Working model for Assignment 5 error analysis / reliability judgment.  
> Local run evidence: `outputs/lr_range_tests/exp3_chroma_lr_range/`, `outputs/chroma_runs/exp3_chroma_baseline_*`

## One-change experiment log

| Field | Value |
| --- | --- |
| run_id | `exp3_chroma_baseline` |
| seed | **42** |
| change | Input representation: raw stereo PCM → 24-d chroma CQT mean/std; model → small MLP (`hidden_dim=64`); LR from chroma range test `≈2.75e-3` |
| hypothesis | Waveform CNN failed because pitch-class structure is hard to extract from raw PCM at this capacity; chroma summaries (same recipe as the sklearn baseline) should yield learnable curves and ~33–40% test accuracy. |
| expected_signal | Val loss well below ~3.18 in early epochs; test accuracy in the classical chroma band (~35–40%); mild overfit possible late. |
| observed_signal | plausible_generalization_mild_overfit — test acc **36.7%**, macro F1 **0.33**, best val loss **1.985** |
| fixed_conditions | same song-level splits, seed 42, 15s chunks, early-stop patience 8 |
| limitation | representation_change_not_hyperparameter_sweep |

## Hypothesis (pre-run)

**Lesson grounding:** Staged model plan from Assignment 4 — if LR/architecture on waveforms fail, switch to harmonic features before claiming the task is impossible.

**Hypothesis:** Cached chroma CQT summaries (12 pitch-class means + 12 stds) fed to a small MLP will learn a non-trivial 24-class mapping, matching the prior logistic-regression chroma baseline (~34.6% test acc).

**Expected behavior:** Clear downward train/val loss in the first epochs; test accuracy ~33–40%; learning curves show real training (unlike Exp 1–2 flat CE).

## Setup

| Setting | Value |
| --- | --- |
| Input | 24-d chroma CQT mean/std (`features/key_chroma/*.npz`) |
| Model | `KeyChromaMLP` (`hidden_dim=64`) |
| Optimizer | AdamW, `lr=0.00275`, `weight_decay=0` |
| Batch size | 64 |
| Epochs | 30 (early stop patience 8) → ran 22 |
| Device | MPS |

## Commands

```bash
cd beatport && source .venv/bin/activate
python chroma_features.py --clip-sec 15

PYTHONUNBUFFERED=1 python ../assignment_5/run_chroma_lr_range_test.py \
  --run-name exp3_chroma_lr_range

PYTHONUNBUFFERED=1 python ../assignment_5/train_key_chroma.py \
  --run-name exp3_chroma_baseline \
  --learning-rate 0.00275 \
  --epochs 30

python ../assignment_5/plot_learning_curves.py \
  ../assignment_5/outputs/chroma_runs/exp3_chroma_baseline_history.csv \
  --output-dir ../assignment_5/outputs/chroma_runs/plots
```

## Observed behavior

| Metric | Exp 1 waveform | Exp 3 chroma MLP |
| --- | --- | --- |
| Best val loss | 3.183 | **1.985** |
| Test accuracy | 2.2% | **36.7%** |
| Test macro F1 | 0.01 | **0.33** |
| Epochs ran | 10 | 22 (early stop) |

**Learning curve:** Loss 3.0 → ~2.0 and val acc ~25% → ~42% in first ~8 epochs; after ~epoch 8–14 val loss plateaus while train loss keeps falling — mild late overfit (best val ≈ epoch 14).

**Vs expected:** Confirmed. Representation was the binding constraint; pipeline and LR on chroma behave as predicted.

## Diagnosis

- Chroma MLP is the **working baseline** for trustworthy evaluation (slices, calibration, Part B).
- Waveform path remains the documented failed intervention / wrong-representation story.
- Next evaluation step (not a new train sweep): Exp 4 error / slice / confidence analysis.

## Artifacts

| File | Purpose |
| --- | --- |
| `outputs/chroma_runs/exp3_chroma_baseline_history.csv` | Epoch metrics |
| `outputs/chroma_runs/exp3_chroma_baseline_metadata.json` | Config + test scores + command |
| `outputs/chroma_runs/exp3_chroma_baseline.log` | Training log |
| `outputs/chroma_runs/plots/*.png` | Learning curves |
| `outputs/lr_range_tests/exp3_chroma_lr_range/` | Range test CSV / plot / metadata |
