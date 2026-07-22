#!/bin/bash
#SBATCH --job-name=cs6320-a5-chroma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-a5-chroma-%j.out
#SBATCH --error=slurm-a5-chroma-%j.err

# CHPC GPU (same allocation as assignment_3)
#SBATCH --partition=utucset-gpu-grn
#SBATCH --qos=utucset-gpu-grn
#SBATCH --account=larsenc
#SBATCH --gres=gpu:1

set -euo pipefail

# Slurm spools a copy of this script; submit dir is where you ran sbatch.
ASSIGNMENT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="$(cd "${ASSIGNMENT_DIR}/.." && pwd)"
BEATPORT_DIR="${REPO_ROOT}/beatport"
OUTPUT_DIR="${ASSIGNMENT_DIR}/outputs"
CHROMA_RUNS="${OUTPUT_DIR}/chroma_runs"
LR_TESTS="${OUTPUT_DIR}/lr_range_tests"

# Data strategy (see assignment_5/CHPC.md):
#   download — fetch Zenodo + full prep (waveforms + chroma); slow, ~2 GB download
#   transfer — expect BEATPORT_DATA_ROOT + beatport/datasets/... already on cluster
#   chroma_only — only train; requires transferred key_chunks.csv + key_chroma/*.npz
BEATPORT_DATA_MODE="${BEATPORT_DATA_MODE:-auto}"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Assignment dir: ${ASSIGNMENT_DIR}"
echo "Repo root: ${REPO_ROOT}"
echo "BEATPORT_DATA_MODE=${BEATPORT_DATA_MODE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if command -v module >/dev/null 2>&1; then
  module load python/3.12.4 2>/dev/null || module load python/3.11 2>/dev/null || true
fi

GPU_VISIBLE=false
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L; then
  GPU_VISIBLE=true
else
  echo "No NVIDIA GPU visible; training will use CPU or MPS if available."
fi

cd "${BEATPORT_DIR}"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

if [[ "${GPU_VISIBLE}" == true ]]; then
  if ! python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    echo "Installing CUDA PyTorch wheel..."
    pip install -q torch --index-url https://download.pytorch.org/whl/cu124
  fi
fi

CHROMA_DIR="${BEATPORT_DIR}/datasets/clip_15s/features/key_chroma"
CHUNKS_CSV="${BEATPORT_DIR}/datasets/clip_15s/key_chunks.csv"
CHROMA_COUNT="$(find "${CHROMA_DIR}" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')"

need_download=false
need_prep=false

case "${BEATPORT_DATA_MODE}" in
  download)
    need_download=true
    need_prep=true
    ;;
  transfer)
    need_download=false
    if [[ "${CHROMA_COUNT}" -lt 8000 ]]; then
      echo "ERROR: transfer mode but chroma cache incomplete (${CHROMA_COUNT} npz files)." >&2
      echo "Rsync beatport/datasets/clip_15s/ from your laptop (see assignment_5/CHPC.md)." >&2
      exit 1
    fi
    ;;
  chroma_only)
    if [[ "${CHROMA_COUNT}" -lt 8000 ]] || [[ ! -f "${CHUNKS_CSV}" ]]; then
      echo "ERROR: chroma_only mode needs key_chunks.csv + ~8000+ key_chroma npz files." >&2
      exit 1
    fi
    ;;
  auto|*)
    if [[ "${CHROMA_COUNT}" -ge 8000 ]] && [[ -f "${CHUNKS_CSV}" ]]; then
      echo "Auto: using existing chroma cache (${CHROMA_COUNT} files)."
    elif [[ -f "${REPO_ROOT}/datasets/beatport EDM Key Dataset/Beatport-EDM-Key-Dataset.xlsx" ]]; then
      echo "Auto: raw dataset present; running prep pipeline."
      need_prep=true
    else
      echo "Auto: no chroma cache or raw data; downloading from Zenodo."
      need_download=true
      need_prep=true
    fi
    ;;
esac

if [[ "${need_download}" == true ]]; then
  bash scripts/prepare_beatport_data.sh
fi

if [[ "${need_prep}" == true ]]; then
  bash scripts/run_key_chroma_prep.sh
fi

echo ">>> LR range test (Exp 3 chroma)"
PYTHONUNBUFFERED=1 python ../assignment_5/run_chroma_lr_range_test.py \
  --run-name exp3_chroma_lr_range \
  --output-dir "${LR_TESTS}"

CHROMA_LR="$(python - <<'PY'
import json
from pathlib import Path
meta = sorted(Path("../assignment_5/outputs/lr_range_tests").glob("exp3_chroma_lr_range/*metadata.json"))
if meta:
    print(json.loads(meta[-1].read_text())["suggestion"]["suggested_lr"])
else:
    print("0.00275")
PY
)"
echo "Using learning rate from range test: ${CHROMA_LR}"

echo ">>> Chroma baseline training"
PYTHONUNBUFFERED=1 python ../assignment_5/train_key_chroma.py \
  --run-name exp3_chroma_baseline \
  --learning-rate "${CHROMA_LR}" \
  --epochs 30 \
  --batch-log-interval 50 \
  --output-dir "${CHROMA_RUNS}"

echo ">>> Learning curve plots"
python ../assignment_5/plot_learning_curves.py \
  "${CHROMA_RUNS}/exp3_chroma_baseline_history.csv" \
  --output-dir "${CHROMA_RUNS}/plots"

mkdir -p "${ASSIGNMENT_DIR}/chpc_evidence"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  cp -f "${ASSIGNMENT_DIR}/slurm-a5-chroma-${SLURM_JOB_ID}.out" "${ASSIGNMENT_DIR}/chpc_evidence/" || true
  cp -f "${ASSIGNMENT_DIR}/slurm-a5-chroma-${SLURM_JOB_ID}.err" "${ASSIGNMENT_DIR}/chpc_evidence/" || true
fi
cp -f "${CHROMA_RUNS}/exp3_chroma_baseline_metadata.json" "${ASSIGNMENT_DIR}/chpc_evidence/" 2>/dev/null || true

echo "Done. Artifacts:"
echo "  ${LR_TESTS}/exp3_chroma_lr_range/"
echo "  ${CHROMA_RUNS}/exp3_chroma_baseline_*"
echo "  ${CHROMA_RUNS}/plots/"
