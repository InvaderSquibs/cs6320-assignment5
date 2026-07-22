#!/usr/bin/env bash
# Push assignment_5 code + beatport training deps to CHPC (no audio, no large caches).
# Run from repo root: bash assignment_5/scripts/sync_chpc_code.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHPC_USER="${CHPC_USER:-u0887388}"
CHPC_HOST="${CHPC_HOST:-granite.chpc.utah.edu}"
CHPC_REPO="${CHPC_REPO:-~/cs6320}"
REMOTE="${CHPC_USER}@${CHPC_HOST}"

echo "Syncing code to ${REMOTE}:${CHPC_REPO} ..."

rsync -avz --progress \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '*.pt' \
  --exclude 'slurm-*.out' \
  --exclude 'slurm-*.err' \
  --exclude 'datasets/' \
  --exclude 'beatport/datasets/' \
  "${REPO_ROOT}/assignment_5/" \
  "${REMOTE}:${CHPC_REPO}/assignment_5/"

rsync -avz --progress \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude 'datasets/' \
  "${REPO_ROOT}/beatport/" \
  "${REMOTE}:${CHPC_REPO}/beatport/"

echo "Done. Next: bash assignment_5/scripts/sync_chpc_chroma.sh"
