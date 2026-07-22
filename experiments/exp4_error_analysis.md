# Experiment Report: error / slice / calibration (Exp 4)

> **Reproducibility tag:** inference on `exp3_chroma_baseline` checkpoint · `seed=42`  
> Local run evidence: `outputs/error_analysis/exp3_chroma_baseline/`

## One-change experiment log

| Field | Value |
| --- | --- |
| run_id | `exp3_chroma_baseline` error analysis |
| seed | **42** (same trained model) |
| change | No retrain — export predictions; slice by mode & preview position; confidence / calibration checks |
| hypothesis | Errors will concentrate by mode (minor easier than major given label imbalance) and by short-clip position; softmax confidence will be higher when correct but not strongly calibrated for “95% sure” claims. |
| expected_signal | Measurable mode accuracy gap; mid-preview chunks slightly better than edges; few/no ≥0.9 confidence mistakes; mean conf(correct) > mean conf(wrong). |
| observed_signal | mode_gap_and_position_effect_confirmed — minor 38.2% vs major 33.1%; mid preview 38.3% best; **0** errors at conf ≥ 0.9 |
| fixed_conditions | model weights, splits, 15s chroma inputs |
| limitation | slice_analysis_only |

## Hypothesis (pre-run)

**Lesson grounding:** Assignment 5 Part A — analyze errors by subgroup/slice and discuss calibration when probabilities are meaningful.

**Hypothesis:** Given A4 risks (class imbalance, short clips, segment ambiguity), the chroma MLP will (1) do better on minor than major, (2) vary by position in the 120s preview, and (3) show higher confidence when correct without extreme overconfidence on mistakes.

**Expected behavior:** Slice tables show non-uniform accuracy; reliability / confidence summaries show usable ranking signal but not deployment-grade calibration.

## Commands

```bash
cd beatport && source .venv/bin/activate
python ../assignment_5/run_key_error_analysis.py \
  --run-name exp3_chroma_baseline
```

## Observed behavior

| Slice / check | Finding |
| --- | --- |
| Test accuracy / macro F1 | 36.7% / 0.329 (`error_analysis_summary.md`) |
| Mode | Minor **38.2%** vs major **33.1%** (`slice_by_mode.csv`) |
| Chunk position | Mid preview (45–90s) **38.3%**; early 36.1%; late 35.0% |
| Confidence | Correct mean conf **0.44** vs wrong **0.33**; max wrong ≈ **0.85** |
| High-confidence errors (≥0.9) | **0** on test |
| Per-class / confusion | Wide F1 spread; related-key confusions in 24×24 matrix |

**Vs expected:** Confirmed on mode gap, position effect, and modest (not extreme) confidence. Softmax is usable for ranking uncertain cases, not for strong certainty claims.

## Diagnosis

- Aggregate ~37% hides mode and position structure — slices matter for trust.
- No ≥0.9 confident mistakes is good news, but overall confidence is still low.
- Feeds Part B: imbalance and short-clip risks **confirmed**; model not client-ready.

## Artifacts

| File | Purpose |
| --- | --- |
| `predictions_test.csv` / `predictions_val.csv` | Per-chunk preds + confidence |
| `classification_confusion_matrix.csv` | 24×24 confusion |
| `classification_per_class_report.csv` | Per-key precision/recall/F1 |
| `slice_by_mode.csv` / `slice_by_chunk_position.csv` | Subgroup metrics |
| `confidence_summary.csv` / `high_confidence_errors.csv` | Calibration-style summaries |
| `plots/*.png` | Confusion, F1, slices, reliability diagram |
| `error_analysis_summary.md` | One-page summary |
