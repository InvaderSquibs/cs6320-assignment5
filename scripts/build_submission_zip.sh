#!/usr/bin/env bash
# Build Assignment 5 submission zip (writeup + code + lightweight local run evidence).
# Excludes model weights, audio/feature caches, and bulky listen samples.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING="${ROOT}/dist/staging"
ZIP="${ROOT}/dist/assignment_5_submission_v1.zip"
EVID="${STAGING}/part_a/run_evidence"

rm -rf "${STAGING}"
mkdir -p \
  "${STAGING}/writeup" \
  "${STAGING}/part_a/code" \
  "${EVID}/experiments" \
  "${EVID}/split_audit" \
  "${EVID}/cnn_runs/plots" \
  "${EVID}/chroma_runs/plots" \
  "${EVID}/lr_range_tests" \
  "${EVID}/error_analysis" \
  "${EVID}/baseline_metrics" \
  "${ROOT}/dist"

# --- Writeup ---
cp "${ROOT}/ASSIGNMENT_5_WRITEUP.md" "${STAGING}/writeup/ASSIGNMENT_5_WRITEUP.md"
cp "${ROOT}/SUBMISSION.md" "${STAGING}/README_SUBMISSION.txt"
if [[ -d "${ROOT}/figures" ]]; then
  mkdir -p "${STAGING}/writeup/figures"
  rsync -a "${ROOT}/figures/" "${STAGING}/writeup/figures/"
fi

# --- Runnable code (no outputs, no weights, no student-example noise optional keep) ---
rsync -a \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.git/' \
  --exclude 'dist/' \
  --exclude 'outputs/' \
  --exclude 'chpc_evidence/' \
  --exclude 'ASSIGNMENT_5_WRITEUP.md' \
  --exclude 'SUBMISSION.md' \
  --exclude '*.pt' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  "${ROOT}/" "${STAGING}/part_a/code/"

# --- Hypothesis-case experiment reports ---
cp "${ROOT}/experiments/"*.md "${EVID}/experiments/"

# --- Split audit (tables + summaries, no heavy extras) ---
rsync -a \
  --include '*/' \
  --include '*.md' \
  --include '*.csv' \
  --include '*.json' \
  --exclude '*' \
  "${ROOT}/outputs/split_audit/" "${EVID}/split_audit/"

# --- CNN runs (Exp 1–2): history, metadata, logs, plots — no checkpoints ---
for name in baseline_cnn exp2_lr_tuned; do
  for suffix in history.csv metadata.json; do
    src="${ROOT}/outputs/cnn_runs/${name}_${suffix}"
    [[ -f "${src}" ]] && cp "${src}" "${EVID}/cnn_runs/"
  done
  [[ -f "${ROOT}/outputs/cnn_runs/${name}.log" ]] && cp "${ROOT}/outputs/cnn_runs/${name}.log" "${EVID}/cnn_runs/"
done
if [[ -d "${ROOT}/outputs/cnn_runs/plots" ]]; then
  rsync -a --include '*.png' --exclude '*' \
    "${ROOT}/outputs/cnn_runs/plots/" "${EVID}/cnn_runs/plots/"
fi

# --- Chroma runs (Exp 3) ---
for name in exp3_chroma_baseline; do
  for suffix in history.csv metadata.json; do
    src="${ROOT}/outputs/chroma_runs/${name}_${suffix}"
    [[ -f "${src}" ]] && cp "${src}" "${EVID}/chroma_runs/"
  done
  [[ -f "${ROOT}/outputs/chroma_runs/${name}.log" ]] && cp "${ROOT}/outputs/chroma_runs/${name}.log" "${EVID}/chroma_runs/"
done
if [[ -d "${ROOT}/outputs/chroma_runs/plots" ]]; then
  rsync -a --include '*.png' --exclude '*' \
    "${ROOT}/outputs/chroma_runs/plots/" "${EVID}/chroma_runs/plots/"
fi

# --- LR range tests ---
for run in exp2_lr_range exp3_chroma_lr_range; do
  src="${ROOT}/outputs/lr_range_tests/${run}"
  if [[ -d "${src}" ]]; then
    mkdir -p "${EVID}/lr_range_tests/${run}"
    rsync -a \
      --include '*/' \
      --include '*.csv' \
      --include '*.json' \
      --include '*.png' \
      --exclude '*' \
      "${src}/" "${EVID}/lr_range_tests/${run}/"
  fi
done

# --- Error analysis (Exp 4): summaries + plots; keep predictions (small) ---
EA_SRC="${ROOT}/outputs/error_analysis/exp3_chroma_baseline"
EA_DST="${EVID}/error_analysis/exp3_chroma_baseline"
if [[ -d "${EA_SRC}" ]]; then
  mkdir -p "${EA_DST}/plots"
  rsync -a \
    --include '*/' \
    --include '*.md' \
    --include '*.csv' \
    --include '*.png' \
    --exclude '*.pt' \
    --exclude '*' \
    "${EA_SRC}/" "${EA_DST}/"
fi

# --- Classical baseline summary (for aggregate comparison table) ---
for f in baseline_metrics_summary.md baseline_metrics.json; do
  [[ -f "${ROOT}/outputs/${f}" ]] && cp "${ROOT}/outputs/${f}" "${EVID}/baseline_metrics/"
done

rm -f "${ZIP}"
(
  cd "${STAGING}"
  zip -r "${ZIP}" writeup part_a README_SUBMISSION.txt
)

echo "Wrote ${ZIP}"
unzip -l "${ZIP}" | tail -30
du -h "${ZIP}"

# Sanity: no model weights in the zip
if unzip -l "${ZIP}" | grep -E '\.pt$' >/dev/null; then
  echo "ERROR: zip contains .pt weights" >&2
  exit 1
fi
echo "OK: no .pt weights in zip"
