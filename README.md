# CS6320 Assignment 5

Generalization and trustworthy evaluation on the Beatport EDM Key portfolio task (24-class major/minor key from 15s audio chunks).

**Repository:** [https://github.com/InvaderSquibs/cs6320-assignment5](https://github.com/InvaderSquibs/cs6320-assignment5)

**Author:** Zach Walton

## What’s in this repo

| Path | Purpose |
| --- | --- |
| `ASSIGNMENT_5_WRITEUP.md` | Part A + Part B writeup |
| `experiments/` | Hypothesis-case report cards (changed setting → expected → observed → diagnosis) |
| `train_key_cnn.py` / `train_key_chroma.py` | Waveform CNN and chroma MLP training |
| `run_lr_range_test.py` / `run_chroma_lr_range_test.py` | Part 04 LR range tests |
| `run_key_error_analysis.py` | Slice / confusion / confidence exports |
| `run_split_audit.py` | Song-level split plausibility checks |
| `outputs/` | Lightweight local run evidence (history, metadata, logs, plots) — **no model weights** |
| `SUBMISSION.md` | Canvas zip layout |
| `scripts/build_submission_zip.sh` | Build `dist/assignment_5_submission_v1.zip` |

## Experiment summary

| Exp | Hypothesis focus | Local result (MPS, seed 42) |
| --- | --- | --- |
| 1 `baseline_cnn` | Raw waveform CNN learns key | Underfit — **2.2%** test acc |
| 2 `exp2_lr_tuned` | LR range test unlocks learning | Still near random — **3.9%** |
| 3 `exp3_chroma_baseline` | Chroma representation works | **36.7%** test acc, mild late overfit |
| 4 error analysis | Mode / position / confidence slices | Minor > major; mid-preview best; no ≥0.9 conf errors |

Full report cards: [`experiments/README.md`](experiments/README.md).

## Setup (local Mac)

Training depends on the sibling `beatport/` package and cached chroma features (not shipped in this repo — too large / audio-derived).

```bash
cd ../beatport
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# After Beatport data + chroma cache exist:
PYTHONUNBUFFERED=1 python ../assignment_5/train_key_chroma.py \
  --run-name exp3_chroma_baseline --learning-rate 0.00275 --epochs 30
```

Reproduce commands for each run are in the matching `outputs/*/*_metadata.json` and experiment markdown.

## Submission zip

```bash
./scripts/build_submission_zip.sh
```

Output: `dist/assignment_5_submission_v1.zip` (~1 MB: writeup + code + report artifacts; no `.pt`, no audio).

## Run evidence

All primary runs were executed **locally on Apple MPS** (`device=mps`, `seed=42`). History CSVs, metadata JSON, logs, and plots are under `outputs/`. CHPC is optional and not required for this submission.
