# Assignment 5 submission package

**Author:** Zach Walton  
**GitHub (code + run evidence):** [https://github.com/InvaderSquibs/cs6320-assignment5](https://github.com/InvaderSquibs/cs6320-assignment5)  
**CHPC:** Optional — local MPS runs with history CSVs, metadata JSON, logs, and plots are the run evidence.

Re-run `scripts/build_submission_zip.sh` after updates.

## Layout

```
assignment_5_submission_v1.zip
├── README_SUBMISSION.txt
├── writeup/
│   └── ASSIGNMENT_5_WRITEUP.md          # Part A + Part B
└── part_a/
    ├── code/                            # scripts + experiment reports (no caches / weights)
    └── run_evidence/                    # lightweight local proof of computation
        ├── experiments/                 # hypothesis-case report cards
        ├── split_audit/
        ├── cnn_runs/                    # Exp 1–2 history, metadata, logs, plots
        ├── chroma_runs/                 # Exp 3 history, metadata, logs, plots (no .pt)
        ├── lr_range_tests/
        ├── error_analysis/              # Exp 4 slices, confusion, confidence, plots
        └── baseline_metrics/            # classical chroma LR baseline summary
```

## What proves the experiments ran locally

| Role | Run | Evidence |
| --- | --- | --- |
| Baseline failure | `baseline_cnn` | `*_history.csv`, `*_metadata.json`, `*.log`, plots |
| Primary intervention | `exp2_lr_tuned` (+ LR range) | same + `lr_range_tests/exp2_lr_range/` |
| Working candidate | `exp3_chroma_baseline` | same + learning-curve plots |
| Error / calibration | Exp 4 on Exp 3 | slices, confidence, confusion, reliability diagram |

Each `*_metadata.json` records device (`mps`), seed (`42`), command, and test metrics.

Hypothesis / expected / observed / diagnosis live in `experiments/exp*.md`.

## Excluded from the zip (intentional)

- Model weights (`*.pt`)
- Audio / waveform caches / chroma `.npz` feature stores
- `outputs/waveform_listen/` (large listen samples)
- CHPC logs (not required when local evidence matches this detail)

## Build

```bash
./scripts/build_submission_zip.sh
```

Output: `dist/assignment_5_submission_v1.zip`
