#!/usr/bin/env bash
# Pull CHPC Slurm logs and training artifacts back to local chpc_evidence/.
#
# Usage:
#   bash assignment_5/scripts/pull_chpc_evidence.sh <JOB_ID>

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <SLURM_JOB_ID>"
  exit 1
fi

JOB_ID="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/chpc_evidence"
REMOTE="u0887388@granite.chpc.utah.edu:cs6320/assignment_5"

mkdir -p "${DEST}"

echo "Pulling Slurm logs for job ${JOB_ID}..."
scp "${REMOTE}/slurm-a5-chroma-${JOB_ID}.out" "${DEST}/" 2>/dev/null || true
scp "${REMOTE}/slurm-a5-chroma-${JOB_ID}.err" "${DEST}/" 2>/dev/null || true

echo "Pulling training artifacts..."
scp "${REMOTE}/outputs/chroma_runs/exp3_chroma_baseline_metadata.json" "${DEST}/" 2>/dev/null || true
scp "${REMOTE}/outputs/chroma_runs/exp3_chroma_baseline_history.csv" "${DEST}/" 2>/dev/null || true
scp -r "${REMOTE}/outputs/chroma_runs/plots" "${DEST}/chroma_plots" 2>/dev/null || true
scp -r "${REMOTE}/outputs/lr_range_tests/exp3_chroma_lr_range" "${DEST}/exp3_chroma_lr_range" 2>/dev/null || true

echo "Done. Evidence under ${DEST}/"
