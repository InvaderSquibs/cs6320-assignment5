# Phase 2: Key Baseline Metrics

- Model: logistic regression 24-class (`C=1.0`)
- Clip length: 15s
- Split unit: key segment (`segment_id`) when enabled
- Random seed: 42
- Features: chroma CQT (cached, no re-extraction needed)

## Metrics by split

| split | n_chunks | n_segments | key24_accuracy | segment_accuracy | key24_macro_f1 | key24_pitch_distance | mode_accuracy | pitch_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 6289 | 800 | 0.481 | 0.559 | 0.468 | 1.014 | 0.674 | 0.604 |
| val | 1352 | 171 | 0.405 | 0.462 | 0.361 | 1.052 | 0.605 | 0.564 |
| test | 1335 | 171 | 0.346 | 0.409 | 0.313 | 1.242 | 0.586 | 0.526 |

## Train vs validation diagnosis

- Pattern: **plausible_generalization**
- Train − val accuracy: 0.076
- Train − val macro F1: 0.108
- Val − test accuracy: 0.059

- Train (48.1%) and validation (40.5%) are within ~7.6% accuracy.
- Macro F1 gap train−val is 10.8% (rarer keys may be overfit).
- See split_audit/*_key_share_curves.png: parallel train/val/test lines mean stratification worked; diverging keys explain macro-F1 gaps.

**Intervention hint:** Next levers: longer clips, richer features, or per-key error analysis.
