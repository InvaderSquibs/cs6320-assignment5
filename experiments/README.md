# Assignment 5 Experiments — Hypothesis Before You Run

Each experiment records **changed setting → hypothesis → expected behavior → observed → diagnosis** (same shape as Assignment 4). Artifacts live under `outputs/`.

| # | Report | Role |
| --- | --- | --- |
| 1 | [exp1_baseline_cnn.md](exp1_baseline_cnn.md) | Waveform CNN baseline (underfit / representation failure) |
| 2 | [exp2_lr_range_and_tuned.md](exp2_lr_range_and_tuned.md) | Primary **generalization intervention**: LR range test → tuned LR |
| 3 | [exp3_chroma_lr_range_and_baseline.md](exp3_chroma_lr_range_and_baseline.md) | Representation change → working chroma MLP candidate |
| 4 | [exp4_error_analysis.md](exp4_error_analysis.md) | Slice / error / calibration on Exp 3 |

**Reproducibility:** seed `42`, device recorded in each `*_metadata.json` (local MPS runs).

**CHPC:** optional for this assignment — local metadata, history CSVs, logs, and plots are the run evidence.
