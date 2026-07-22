#!/usr/bin/env bash
# Copy pre-built 15s chroma caches to CHPC (Option A: chroma_only).
# Run from your laptop, repo root: bash assignment_5/scripts/sync_chpc_chroma.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHPC_USER="${CHPC_USER:-u0887388}"
CHPC_HOST="${CHPC_HOST:-granite.chpc.utah.edu}"
CHPC_REPO="${CHPC_REPO:-~/cs6320}"
REMOTE="${CHPC_USER}@${CHPC_HOST}"

echo "Local repo:  ${REPO_ROOT}"
echo "Remote repo: ${REMOTE}:${CHPC_REPO}"
echo ""
echo "Transferring key_chunks.csv + key_chroma cache (~35 MB, 8969 npz files)..."

rsync -avz --progress \
  "${REPO_ROOT}/beatport/datasets/clip_15s/key_chunks.csv" \
  "${REMOTE}:${CHPC_REPO}/beatport/datasets/clip_15s/"

rsync -avz --progress \
  "${REPO_ROOT}/beatport/datasets/clip_15s/features/key_chroma/" \
  "${REMOTE}:${CHPC_REPO}/beatport/datasets/clip_15s/features/key_chroma/"

echo ""
echo "Done. On CHPC:"
echo "  cd ${CHPC_REPO}/assignment_5"
echo "  export BEATPORT_DATA_MODE=chroma_only"
echo "  sbatch run.slurm"
