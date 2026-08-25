#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
QURIFT_GPUS="${QURIFT_GPUS:-auto}"
QURIFT_LABEL_JOBS_PER_GPU="${QURIFT_LABEL_JOBS_PER_GPU:-1}"
QURIFT_LABEL_MAX_QUERIES="${QURIFT_LABEL_MAX_QUERIES:-512}"
QURIFT_LABEL_INIT_QUERIES="${QURIFT_LABEL_INIT_QUERIES:-128}"
TARGETS="satml_targets/credit_factorial_targets.csv"

mkdir -p satml_logs satml_results/credit_factorial

"${PYTHON_BIN}" -u experiments/gen_results/run_train_mia_attack_cvholdout_multigpu.py \
  --launcher \
  --attack-data-dir satml_runs/satml_credit_factorial \
  --out satml_results/credit_factorial/learned_mia \
  --test-ratio 0.2 \
  --cv-folds 5 \
  --tune \
  --n-trials 20 \
  --max-epochs 150 \
  --patience 15 \
  --device cuda \
  --seed 2026 \
  --cpu-threads 2 \
  --resume \
  --jobs-per-gpu 1 \
  --gpus "${QURIFT_GPUS}" \
  2>&1 | tee satml_logs/credit_learned_mia.log

"${PYTHON_BIN}" -u reviewer_tools/run_lira_reference_multigpu.py \
  --targets "${TARGETS}" \
  --repo-root . \
  --run-root satml_runs \
  --out-dir satml_results/credit_factorial/lira \
  --num-references 16 \
  --bootstrap 10000 \
  --seed 2026 \
  --gpus "${QURIFT_GPUS}" \
  --jobs-per-gpu 1 \
  --cpu-threads 2 \
  --resume \
  2>&1 | tee satml_logs/credit_lira.log

PYTHON_BIN="${PYTHON_BIN}" \
QURIFT_GPUS="${QURIFT_GPUS}" \
QURIFT_LABEL_JOBS_PER_GPU="${QURIFT_LABEL_JOBS_PER_GPU}" \
QURIFT_LABEL_MAX_QUERIES="${QURIFT_LABEL_MAX_QUERIES}" \
QURIFT_LABEL_INIT_QUERIES="${QURIFT_LABEL_INIT_QUERIES}" \
  bash commands/satml_run_credit_label_only_hsj.sh

PYTHON_BIN="${PYTHON_BIN}" bash commands/satml_analyze_credit_all_attacks.sh
