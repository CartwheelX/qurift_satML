#!/usr/bin/env bash
set -euo pipefail

: "${QURIFT_NOISE_SNAPSHOT:?Set QURIFT_NOISE_SNAPSHOT to the same frozen snapshot used for N1/N2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
QURIFT_GPUS="${QURIFT_GPUS:-auto}"
QURIFT_LIRA_JOBS_PER_GPU="${QURIFT_LIRA_JOBS_PER_GPU:-auto}"
QURIFT_NOISE_JOBS_PER_GPU="${QURIFT_NOISE_JOBS_PER_GPU:-1}"
TARGETS="satml_targets/noise/mnist_noise_n3_lira_targets.csv"
OUT="satml_results/noise/n3_attack_breadth"
REFERENCES="${OUT}/lira_references"

mkdir -p "${OUT}" satml_logs
"${PYTHON_BIN}" satml_tools/build_noise_study_targets.py

"${PYTHON_BIN}" -u reviewer_tools/run_lira_reference_multigpu.py \
  --targets "${TARGETS}" --repo-root . --run-root reviewer_runs \
  --out-dir "${REFERENCES}" --num-references 16 --save-reference-checkpoints \
  --bootstrap 5000 --seed 2026 --gpus "${QURIFT_GPUS}" \
  --jobs-per-gpu "${QURIFT_LIRA_JOBS_PER_GPU}" --cpu-threads 2 --phase all --resume \
  2>&1 | tee satml_logs/noise_n3_lira_reference_training.log

"${PYTHON_BIN}" -u satml_tools/run_noisy_lira_targets.py \
  --targets "${TARGETS}" --repo-root . --run-root reviewer_runs \
  --reference-dir "${REFERENCES}" --out-dir "${OUT}/noisy_lira" \
  --snapshot "${QURIFT_NOISE_SNAPSHOT}" --num-references 16 \
  --modes ideal_shot,noisy_shot --shots 512 --simulator-seeds 0,1,2,3,4 \
  --gpus "${QURIFT_GPUS}" --jobs-per-gpu "${QURIFT_NOISE_JOBS_PER_GPU}" \
  --cpu-threads 2 --resume \
  2>&1 | tee satml_logs/noise_n3_lira_scoring.log

"${PYTHON_BIN}" -u satml_tools/analyze_noisy_lira.py \
  --targets "${TARGETS}" \
  --exact "${REFERENCES}/lira_reference_mia_raw.csv" \
  --noisy "${OUT}/noisy_lira/noisy_lira_raw.csv" \
  --out-dir "${OUT}/analysis" --bootstrap 10000 --bootstrap-seed 2026 \
  2>&1 | tee satml_logs/noise_n3_lira_analysis.log

printf '%s\n' '[INFO] Noisy label-only remains an optional, separately budgeted pilot.'
