#!/usr/bin/env bash
set -euo pipefail

: "${QURIFT_NOISE_SNAPSHOT:?Set QURIFT_NOISE_SNAPSHOT to the same frozen snapshot used for N1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
QURIFT_GPUS="${QURIFT_GPUS:-auto}"
QURIFT_NOISE_JOBS_PER_GPU="${QURIFT_NOISE_JOBS_PER_GPU:-auto}"
QURIFT_NOISE_SHARDS_PER_TARGET="${QURIFT_NOISE_SHARDS_PER_TARGET:-0}"
TARGETS="satml_targets/noise/mnist_noise_n2_query_targets.csv"
OUT="satml_results/noise/n2_query_policy"
PAIRS="1x128,1x512,1x2560,5x128,5x512,20x128"

mkdir -p "${OUT}" satml_logs
"${PYTHON_BIN}" satml_tools/build_noise_study_targets.py
"${PYTHON_BIN}" -u satml_tools/run_noise_targets.py \
  --targets "${TARGETS}" --repo-root . --run-root reviewer_runs \
  --out-dir "${OUT}/conditions" --snapshot "${QURIFT_NOISE_SNAPSHOT}" \
  --study-name n2_query_policy --modes exact,ideal_shot,noisy_shot \
  --query-shot-pairs "${PAIRS}" --simulator-seeds 0,1,2,3,4,5,6,7,8,9 \
  --n-member 200 --n-nonmember 200 --sample-seed 2026 \
  --transpiler-seed 2026 --optimization-level 1 --qiskit-batch-size 16 \
  --bootstrap 5000 --bootstrap-seed 2026 --device cuda \
  --gpus "${QURIFT_GPUS}" --jobs-per-gpu "${QURIFT_NOISE_JOBS_PER_GPU}" \
  --condition-shards-per-target "${QURIFT_NOISE_SHARDS_PER_TARGET}" \
  --cpu-threads 2 --resume 2>&1 | tee satml_logs/noise_n2_queries.log

"${PYTHON_BIN}" -u satml_tools/noisy_learned_mia.py \
  --root "${OUT}/conditions" --out-dir "${OUT}/learned_mia" \
  --feature-modes pv,pv_mean_std --folds 5 --split-seed 2026 \
  --attacker-seed 41 --resume 2>&1 | tee satml_logs/noise_n2_learned_mia.log

"${PYTHON_BIN}" -u satml_tools/analyze_noise_studies.py \
  --study n2 --root "${OUT}/conditions" \
  --learned "${OUT}/learned_mia/learned_mia_raw.csv" \
  --out-dir "${OUT}/analysis" --bootstrap 10000 --bootstrap-seed 2026 \
  2>&1 | tee satml_logs/noise_n2_analysis.log
