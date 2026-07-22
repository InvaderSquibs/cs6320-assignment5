# Assignment 5 — CHPC notes

**Shared CHPC workflow** (SSH, Slurm basics, account/partition, upload rules, troubleshooting): **[`../../CHPC.md`](../../CHPC.md)**.

This file only covers what is **unique to Assignment 5** (Beatport data paths and sync).

---

## What runs on CHPC

Exp 3 chroma MLP: LR range test → baseline train → learning-curve plots.

```bash
cd ~/cs6320/assignment_5
export BEATPORT_DATA_MODE=chroma_only
sbatch run.slurm
```

Logs: `slurm-a5-chroma-<jobid>.out` / `.err` → copy to `chpc_evidence/`. See `chpc_evidence/sbatch_command.txt`.

---

## Sync from Mac

```bash
cd /Users/zach.walton/Dev/shule/cs6320
bash assignment_5/scripts/sync_chpc_code.sh      # code (first time or after edits)
bash assignment_5/scripts/sync_chpc_chroma.sh    # chroma_only: ~35 MB caches
```

Manual rsync (same as sync scripts):

```bash
rsync -avz beatport/datasets/clip_15s/key_chunks.csv \
  u0887388@granite.chpc.utah.edu:~/cs6320/beatport/datasets/clip_15s/

rsync -avz beatport/datasets/clip_15s/features/key_chroma/ \
  u0887388@granite.chpc.utah.edu:~/cs6320/beatport/datasets/clip_15s/features/key_chroma/
```

---

## `BEATPORT_DATA_MODE` options

| Mode | When to use |
| --- | --- |
| `chroma_only` | **Recommended** — `key_chunks.csv` + `features/key_chroma/*.npz` already on cluster |
| `transfer` | Rsync full Zenodo folder and/or `beatport/datasets/` |
| `download` | Cluster downloads Zenodo + rebuilds caches — **4+ hours**, **32G+ RAM** |
| `auto` | Chroma if ≥ ~8000 files, else prep, else download |

Training only needs `key_chunks.csv` + chroma npz files (8969 files) for `chroma_only`.

---

## Local parity (reference)

```bash
cd beatport && source .venv/bin/activate
python chroma_features.py --clip-sec 15
PYTHONUNBUFFERED=1 python ../assignment_5/run_chroma_lr_range_test.py --run-name exp3_chroma_lr_range
PYTHONUNBUFFERED=1 python ../assignment_5/train_key_chroma.py --run-name exp3_chroma_baseline --learning-rate 0.00275 --epochs 30
python ../assignment_5/plot_learning_curves.py ../assignment_5/outputs/chroma_runs/exp3_chroma_baseline_history.csv --output-dir ../assignment_5/outputs/chroma_runs/plots
```

---

## Wrap-up checklist

| Done | Item |
| --- | --- |
| ✅ | Split audit, waveform CNN underfitting (Exp 1–2), LR range |
| ✅ | Chroma representation + learning curves (Exp 3) |
| ✅ | Error / slice analysis + calibration on `exp3_chroma_baseline` |
| ✅ | Part A + Part B writeup (`ASSIGNMENT_5_WRITEUP.md`) |
| ✅ | Hypothesis-case experiment reports (`experiments/exp1`–`exp4`) |
| ✅ | Local run evidence (history / metadata / logs / plots) — CHPC optional |
| ✅ | Lightweight zip: `./scripts/build_submission_zip.sh` → `dist/assignment_5_submission_v1.zip` |
| ✅ | Public GitHub repo [cs6320-assignment5](https://github.com/InvaderSquibs/cs6320-assignment5) |

Exp 3 is the **working model** for trustworthy evaluation; waveform runs document the failed-intervention / wrong-representation story.

### Pull evidence (Mac)

```bash
bash assignment_5/scripts/pull_chpc_evidence.sh <JOBID>
```
