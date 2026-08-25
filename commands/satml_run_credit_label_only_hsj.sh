#!/usr/bin/env bash
set -euo pipefail

# Resume the Credit attack stage specifically from its label-only baselines.
# The older validation-anchor chord output is deliberately not read here.
PYTHON_BIN="${PYTHON_BIN:-python}"
QURIFT_GPUS="${QURIFT_GPUS:-auto}"
QURIFT_LABEL_JOBS_PER_GPU="${QURIFT_LABEL_JOBS_PER_GPU:-1}"
QURIFT_LABEL_MAX_QUERIES="${QURIFT_LABEL_MAX_QUERIES:-512}"
QURIFT_LABEL_INIT_QUERIES="${QURIFT_LABEL_INIT_QUERIES:-128}"
TARGETS="satml_targets/credit_factorial_targets.csv"

mkdir -p satml_logs satml_results/credit_factorial

"${PYTHON_BIN}" -u reviewer_tools/label_only_correctness_attack.py \
  --attack-data-dir satml_runs/satml_credit_factorial \
  --targets "${TARGETS}" \
  --out-dir satml_results/credit_factorial/label_only_correctness \
  --bootstrap 10000 \
  --seed 2026 \
  2>&1 | tee satml_logs/credit_label_only_correctness.log

"${PYTHON_BIN}" -u reviewer_tools/run_label_only_hsj_multigpu.py \
  --targets "${TARGETS}" \
  --repo-root . \
  --run-root satml_runs \
  --out-dir satml_results/credit_factorial/label_only_hsj \
  --n-member 200 \
  --n-nonmember 200 \
  --max-queries "${QURIFT_LABEL_MAX_QUERIES}" \
  --init-queries "${QURIFT_LABEL_INIT_QUERIES}" \
  --init-batch-size 32 \
  --iterations 8 \
  --gradient-samples 32 \
  --binary-steps 10 \
  --step-search-steps 10 \
  --bootstrap 10000 \
  --seed 2026 \
  --gpus "${QURIFT_GPUS}" \
  --jobs-per-gpu "${QURIFT_LABEL_JOBS_PER_GPU}" \
  --cpu-threads 2 \
  --resume \
  2>&1 | tee satml_logs/credit_label_only_hsj.log

printf '%s\n' '[OK] Credit correctness-only and hard-label HSJ results are complete.'
printf '%s\n' '[NEXT] Run commands/satml_analyze_credit_all_attacks.sh to refresh the paired all-attack analysis.'
