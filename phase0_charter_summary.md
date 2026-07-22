# Phase 0: Portfolio Charter Summary (Assignment 5 Reference)

**Status:** No separate Assignment 4 Part B charter file was found in this repo. This document consolidates the locked scope from Assignment 3 (`assignment_3/part_b_proposal.md`), Assignment 4 requirements, and what is already implemented in `beatport/` (including `BASELINE_WRITEUP.md` and `config.yaml`). Update this file if your submitted A4 charter differs.

---

## Portfolio problem statement

Predict **musical key** (24 major/minor classes) from short **EDM audio clips**, using **human-annotated labels** from the Beatport EDM Key Dataset.

## Intended stakeholder / use case

DJs and producers who want a **suggested key** for harmonic mixing or composition. Predictions are practice aids, not authoritative analysis.

## Dataset

| Item | Detail |
| --- | --- |
| Source | [Beatport EDM Key Dataset](https://doi.org/10.5281/zenodo.1101082) (Faraldo, 2017) |
| License | CC BY-SA 4.0 — academic portfolio use only |
| Local path | `datasets/beatport EDM Key Dataset/` |
| Label source | Human labels only (`keys/` / xlsx `main_key`); **not** Beatport JSON `key` |

**Locked filtering (implemented):** `confidence ≥ 2`, single standard major/minor key (Tier A) → **1,122 songs** for key modeling.

## Prediction target and inputs

| Field | Value |
| --- | --- |
| **Primary target** | `key_24` — 24-class major/minor key |
| **Secondary target** | BPM (tempo tier; separate evaluation) |
| **Inputs** | Audio chunks → hand-crafted features (chroma CQT for key) |
| **Baseline** | Majority class; logistic regression on chroma |
| **Initial candidate** | Same LR baseline with song-level aggregation |
| **Revised model (staged)** | Neural embeddings (VGGish/MusicNN) or spectrogram CNN |

## Evaluation strategy (locked in `beatport/config.yaml`)

| Setting | Value |
| --- | --- |
| Split | 70% train / 15% val / 15% test |
| Unit | **Song (`track_id`)** — all chunks from a song share one split |
| Stratification | By `key_24` (key task) |
| Random seed | 42 |
| Chunk length (primary) | 15s non-overlapping (8 chunks/song) |
| Sample weighting | `1 / n_chunks` per song |

## Metrics (Assignment 5 primary: key task)

- Top-1 chunk accuracy (sample-weighted)
- Song-level accuracy (majority vote across chunks)
- Macro F1
- Pitch circle-of-fifths distance
- Mode accuracy and pitch accuracy (decomposed)

## Leakage and quality risks (audit → test in A5)

| Risk / assumption | A4 lock | A5 evidence to collect |
| --- | --- | --- |
| Chunk leakage if split by row | Split by track ID | Split audit: no track in multiple splits |
| Class imbalance across 24 keys | Known (~2:1 minor:major) | Label distribution by split; per-class F1 |
| Short clips weaken harmonic signal | 15s default; 20s/30s staged | Slice by `chunk_idx`; optional clip-length compare |
| Key stays near majority baseline | Success = beat ~10% majority | Baseline vs LR metrics |
| EDM-only scope | No cross-genre data | Still untested — note as limitation |
| Human label noise / dual keys | Filtered to confidence=2, single key | Error examples; high-confidence mistakes |
| Beatport JSON key ≠ human key | Human labels only | Documented; not re-tested |

## Success criteria (course project)

- Beat trivial baselines (majority, stratified dummy) on key task
- Reproducible pipeline with documented split and metrics
- Evidence of **where** the model works/fails (slices), not just aggregate accuracy
- Bounded claim: useful learning artifact, **not** deployment-ready DJ tool

## Scope limits

- Will: key classification baseline + trustworthy evaluation on Beatport Tier A
- Will not: claim production DJ readiness; exhaustive hyperparameter search; cross-genre generalization this week
- Tempo: secondary; optional parallel work

## Fallback plan

1. Tighten labels (`confidence = 2`) — **already applied**
2. Reduce to 24-class key space — **already applied**
3. Hand-crafted features before neural model — **current stage**
4. Move tempo to Groove dataset if Beatport BPM insufficient

## Staged model-improvement plan

| Stage | Model | Evidence needed |
| --- | --- | --- |
| **Baseline (now)** | LR on chroma CQT, 15s chunks | Split audit, train/val/test metrics, error slices |
| **Intervention (A5)** | One generalization change (e.g. regularization strength) | Before/after train vs val behavior |
| **Next** | Longer clips or neural embeddings | Clip-length comparison; embedding ablation |
| **Final presentation** | Best justified candidate + reliability judgment | Calibration, subgroup checks, deployment bounds |

## Assignment 5 evaluation contract

**Part A:** Evaluate `Key24Classifier` (logistic regression, `C=1.0` default) on 15s key chunks. Portfolio work (not TLC practice).

**Part B:** Trace ≥3 rows from the risk table above to Week 5 artifacts; state bounded reliability judgment.

---

## Note on `C` (for later phases)

`C` is scikit-learn's **inverse regularization strength** for `LogisticRegression` — already in `beatport/models/key_baseline.py` and `config.yaml` (`key_model.C: 1.0`). It is **not** a custom project term.

- **Higher `C`** → weaker penalty → model can fit training data more closely (more overfitting risk)
- **Lower `C`** → stronger L2 penalty → simpler decision boundary (more regularization)

We have **not** run the regularization study yet; that is a candidate intervention for Phase 4 after we see train vs val gaps in Phase 2.
