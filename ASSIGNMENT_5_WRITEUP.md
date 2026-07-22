# CS 6320 Assignment 5 Writeup

**Portfolio project:** Beatport EDM Key Dataset — 24-class major/minor key from audio chunks  
**Work type:** Portfolio (not TLC practice)  
**Repo:** [https://github.com/InvaderSquibs/cs6320-assignment5](https://github.com/InvaderSquibs/cs6320-assignment5)  
**Primary clip length this assignment:** 15 seconds  
**Random seed:** 42  

Artifacts live under `assignment_5/outputs/`. Commands and run names are recorded in each `*_metadata.json`.

**Run evidence:** local MPS training (seed `42`) with history CSVs, metadata JSON, logs, and plots — same detail level as a CHPC log package. CHPC is optional for this submission.

**Hypothesis-case experiment reports** (changed setting → hypothesis → expected → observed → diagnosis):

| Exp | Report |
| --- | --- |
| 1 Waveform CNN baseline | `experiments/exp1_baseline_cnn.md` |
| 2 LR intervention | `experiments/exp2_lr_range_and_tuned.md` |
| 3 Chroma MLP candidate | `experiments/exp3_chroma_lr_range_and_baseline.md` |
| 4 Error / slice / calibration | `experiments/exp4_error_analysis.md` |

---

## Part A: Model Evaluation and Generalization Evidence

### Dataset, task, and model

| Item | Detail |
| --- | --- |
| **Dataset** | [Beatport EDM Key Dataset](https://doi.org/10.5281/zenodo.1101082) (Faraldo, 2017), Tier A filters (`confidence ≥ 2`, single standard major/minor) |
| **Task** | Multiclass classification — predict `key_24` (24 major/minor keys) from a single audio chunk |
| **Stakeholder** | DJs/producers seeking a *practice* key hint for harmonic mixing — not an authoritative analyzer |
| **Models evaluated** | (1) **Waveform 1D CNN** on stereo PCM — diagnostic baseline; (2) **Chroma MLP** — primary working neural candidate after representation change |

**Inputs:**

- **Waveform CNN:** 15s stereo PCM @ 44.1 kHz, shape `(2, 661500)`, peak-normalized per chunk (`beatport/datasets/clip_15s/waveforms/key/`).
- **Chroma MLP:** 24-d chroma CQT summary per chunk (12 pitch-class means + 12 stds over time), cached in `features/key_chroma/` — same recipe as the earlier logistic-regression baseline, fed to a small two-layer MLP (`hidden_dim=64`).

### Split strategy and plausibility evidence

| Setting | Value |
| --- | --- |
| Split ratios | 70% train / 15% val / 15% test |
| Split unit | **Song (`track_id`)** — no chunk from a song appears in more than one split |
| Key segments | Enabled when applicable (`key_segments.enabled`) to reduce leakage across modulation periods within a song |
| Stratification | By `key_24` |

**Counts** (`outputs/split_audit/split_audit_summary.md`):

| Split | Songs | Chunks (15s) |
| --- | --- | --- |
| train | 785 | ~6,285 |
| val | 168 | ~1,350 |
| test | 169 | ~1,334 |

**Leakage checks:** No track in multiple splits; chunk split consistent with song split. All 24 keys, 2 modes, and 12 pitch classes appear in val and test (no unseen labels vs train).

**Why this split is appropriate:** Key labels are song/segment properties; evaluating on chunks from held-out songs approximates “new track at prediction time.” Segment-level labeling further limits leakage when a preview changes key mid-track.

### Aggregate metrics (test set, 15s chunks)

| Model | Test accuracy | Test macro F1 | Test loss | Notes |
| --- | --- | --- | --- | --- |
| Majority class (reference) | ~10% | — | — | From Phase 2 baselines |
| Logistic regression on chroma | **34.6%** | **0.31** | — | `outputs/baseline_metrics_summary.md` |
| Waveform CNN `baseline_cnn` (`lr=1e-3`) | **2.2%** | **0.01** | 3.23 | Near random (~4% for 24 classes) |
| Waveform CNN `exp2_lr_tuned` (`lr=2e-3`) | **3.9%** | **0.02** | 3.21 | LR intervention — little change |
| **Chroma MLP `exp3_chroma_baseline`** (`lr=2.75e-3`) | **36.7%** | **0.33** | 1.99 | Primary neural candidate |

Chroma MLP matches the classical chroma baseline band (~35–40%), which supports that the neural training pipeline is sound and that **representation** was the binding issue for the waveform CNN, not a broken optimizer.

### Training vs validation behavior

**Waveform CNN (Exp 1–2):** Train and val loss stay near **~3.0–3.2** (≈ random 24-class cross-entropy ~3.18). Val accuracy jitters between ~4–8% with no sustained downward trend. Diagnosis: **underfitting / wrong input scale** — the model never extracts usable pitch-class structure from raw PCM.

- Learning curves: `outputs/cnn_runs/plots/loss_learning_curve.png`, `validation_metric_learning_curve.png`
- LR range test (Exp 2): `outputs/lr_range_tests/exp2_lr_range/exp2_lr_range_lr_range.png` — brief loss dip at high LR, but full training at `2e-3` did not fix learning.

**Chroma MLP (Exp 3):** Clear learning in the first ~8 epochs (loss 3.0 → ~2.0; val accuracy ~25% → ~42%). After epoch ~8–14, **val loss plateaus** (~1.98–2.05) while train loss continues to fall (~1.69 by epoch 22) — mild **overfitting** with early stopping at epoch 22 (best val checkpoint ≈ epoch 14, val loss 1.985).

- Learning curves: `outputs/chroma_runs/plots/loss_learning_curve.png`, `validation_metric_learning_curve.png`
- LR range test: `outputs/lr_range_tests/exp3_chroma_lr_range/exp3_chroma_lr_range_lr_range.png`

**Train vs val gap (chroma MLP):** Test accuracy 36.7% vs train eval accuracy ~44% at late epochs — plausible generalization, not severe overfit, but not high enough for deployment.

### Generalization intervention

**Primary intervention (waveform path):** Learning-rate tuning guided by Part 04 LR range test (`1e-6` → `1e-1` over one train pass).

- **Before:** `baseline_cnn`, `lr=1e-3` — test acc 2.2%
- **After:** `exp2_lr_tuned`, `lr=2e-3` — test acc 3.9%
- **What changed:** Metrics moved slightly but remained near random; curves still flat. **Conclusion:** LR alone does not address the failure mode.

**Effective change (representation):** Switch input from raw waveforms to chroma CQT summaries and train `KeyChromaMLP` with LR chosen from range test (`≈2.75e-3`). This is the staged model improvement that actually produced learnable curves and ~37% test accuracy — framed here as the **post-intervention candidate** for trustworthy evaluation, not as a second hyperparameter sweep.

### Error and slice analysis (chroma MLP, test set)

Artifacts: `outputs/error_analysis/exp3_chroma_baseline/`

| Slice | Finding |
| --- | --- |
| **Mode** | Minor chunks slightly higher accuracy than major (38.2% vs 33.1%) — see `plots/slice_by_mode.png` |
| **Chunk position in 120s preview** | Mid preview (45–90s) best (~38.3% acc); early and late slightly lower — see `plots/slice_by_chunk_position.png` |
| **Per-class** | Spread in F1 across 24 keys; some keys near zero F1 — `plots/per_class_f1.png`, `classification_per_class_report.csv` |
| **Confusion** | Full 24×24 matrix — `plots/confusion_matrix_heatmap.png` — shows key confusions (relative keys / circle-of-fifths neighbors) |

### Calibration and confidence

- **Reliability diagram:** `outputs/error_analysis/exp3_chroma_baseline/plots/reliability_diagram.png`
- **Confidence when correct vs wrong:** Mean confidence ~0.44 when correct vs ~0.33 when incorrect (`confidence_summary.csv`).
- **High-confidence errors (≥0.9):** None on test set; max wrong confidence ~0.85 — model is **not** severely overconfident on mistakes, but overall confidence is modest.
- **Interpretation:** Softmax probabilities are usable for *ranking* uncertain cases, not for strong “I am 95% sure” claims.

### Diagnosis summary

| Pattern | Evidence |
| --- | --- |
| Waveform CNN | Underfitting / representation failure |
| Chroma MLP | Plausible generalization with mild late overfit |
| Leakage | No evidence of song-level split violations |
| Unstable val (waveform) | Metric noise on a model that never learns |
| Unstable val (chroma, late epochs) | Plateau + mild overfit, not collapse |

---

## Part B: Portfolio Audit Update and Reliability Judgment

### Trace: Assignment 4 risks → Week 5 evidence

| A4 risk / assumption | Week 5 evidence | Status |
| --- | --- | --- |
| **Chunk leakage if split by row** | Split audit: 0 tracks in multiple splits; chunk/song consistency checks pass | **Reduced** — split procedure behaves as intended |
| **Short clips weaken harmonic signal** | Waveform CNN fails; chroma works modestly; slice by chunk position shows variation (mid preview slightly better) | **Confirmed** — 15s is hard; position in preview matters |
| **Class imbalance (24 keys, minor-heavy)** | Per-class F1 spread; mode slice shows major harder than minor | **Confirmed** in error analysis |
| **Key stays near majority baseline** | Chroma MLP ~37% vs majority ~10% | **Contradicted** for chroma path — beats trivial baselines |
| **Human label noise / segment ambiguity** | Confusion among related keys; no single dominant failure mode | **Partially confirmed** — needs more qualitative examples |

### Is the model reliable enough for the intended use **at this stage**?

**No — not for client-facing or “trust this key” use.**

Supported today:

- Chroma-based model (classical or small MLP) learns a **non-trivial** 24-class mapping on held-out songs (~35–40% chunk accuracy).
- Evaluation pipeline (splits, curves, slices, calibration plots) is **inspectable and reproducible**.
- Raw waveform + small CNN **without** harmonic features is **not** a viable path on this setup.

Not supported:

- High-accuracy key detection on every 15s chunk.
- Calibrated high-confidence predictions (most confidences &lt; 0.5).
- Generalization beyond Beatport EDM Tier A, 15s non-overlapping chunks.

**Bounded recommendation:** Treat current outputs as a **research / practice baseline** — useful for comparing models and identifying failure slices, not for automated DJ key labeling without human review.

### Evidence still needed before any deployment-style claim

- Comparison on **longer clip lengths** and song-level aggregation.
- Richer inputs (full time–frequency chroma or spectrogram CNN, or pretrained audio embeddings).
- Per-key error budget for rare classes.
- Optional: listener QA on high-confidence errors (none at 0.9 threshold today).

### Staged model improvement — **next priority: other sample lengths**

This assignment locked **15s** chunks (`clip_15s`). The pipeline already supports **20s and 30s** as separate datasets (`config.yaml` → `clip_lengths: [15, 20, 30]`) with the same song-level splits and caching layout (`clip_20s/`, `clip_30s/`).

**Planned next work (not yet run for Assignment 5):**

1. **Extract chroma caches** for `clip_20s` and `clip_30s` (`chroma_features.py --clip-sec 20 30` after waveforms or audio prep).
2. **Train the same chroma MLP protocol** (LR range test → baseline run → learning curves → error analysis) at each length.
3. **Compare** chunk accuracy, macro F1, and slice behavior (especially chunk position effects) across lengths.
4. **Hypothesis:** Longer windows expose more stable harmony → better key accuracy without changing model family — consistent with A4’s “short clips weaken signal” risk and `BASELINE_WRITEUP.md` clip-length comparison notes.

We are **not** claiming that longer clips will reach very high accuracy by themselves; we expect them to be the **next controlled experiment** on the same evaluation harness before considering heavier architectures (spectrogram CNN, embeddings).

Other staged options after clip-length comparison:

- Regularization on chroma MLP (weight decay, dropout, LR schedule) where val plateaued.
- Song-level majority vote across chunks per track.
- Richer chroma representation (full `(12 × time)` input vs 24-d summary).

### What transfers from this assignment

- Split audit + leakage checks before trusting any new metric.
- Learning curves required to distinguish underfitting vs overfitting vs noise.
- Error slices (mode, position in preview) and calibration plots before claiming trust.
- Representation choice matters more than LR tuning when loss sits at random-chance CE.

---

## Artifact index (for submission zip)

| Category | Path |
| --- | --- |
| Split audit | `outputs/split_audit/` |
| Waveform CNN runs | `outputs/cnn_runs/baseline_cnn_*`, `exp2_lr_tuned_*`, `plots/` |
| LR range tests | `outputs/lr_range_tests/exp2_lr_range/`, `exp3_chroma_lr_range/` |
| Chroma MLP run | `outputs/chroma_runs/exp3_chroma_baseline_*`, `plots/` |
| Error / calibration | `outputs/error_analysis/exp3_chroma_baseline/` (+ `plots/`) |
| Phase 2 LR baseline | `outputs/baseline_metrics_summary.md` |
| CHPC (optional) | `run.slurm`, `CHPC.md`, `chpc_evidence/` |

## Commands (reference)

```bash
cd beatport && source .venv/bin/activate

# Chroma cache (15s — done)
python chroma_features.py --clip-sec 15

# Chroma MLP training
PYTHONUNBUFFERED=1 python ../assignment_5/train_key_chroma.py \
  --run-name exp3_chroma_baseline --learning-rate 0.00275 --epochs 30

# Error analysis + plots
python ../assignment_5/run_key_error_analysis.py --run-name exp3_chroma_baseline --skip-export

# Planned: 20s / 30s
python chroma_features.py --clip-sec 20 30
# then same train + analysis pattern per clip length
```
