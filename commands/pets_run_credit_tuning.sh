#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
QURIFT_GPUS="${QURIFT_GPUS:-auto}"
QURIFT_JOBS_PER_GPU="${QURIFT_JOBS_PER_GPU:-auto}"
# The initial utility-only grid {1,4,8} failed the frozen utility gate for both
# structural roles.  The documented second stage adds {16,32,64}; keeping the
# initial values here lets --resume reuse their completed checkpoints and lets
# the selector choose the smallest epsilon across the combined grid.
QURIFT_PETS_DP_TUNING_EPSILONS="${QURIFT_PETS_DP_TUNING_EPSILONS:-1,4,8,16,32,64}"

# Refresh the defense manifest so confirmatory DP rows carry the faithful
# Watkins optimizer/batch/epoch settings even when an older pilot manifest is
# already present.
"${PYTHON_BIN}" pets_tools/build_defense_training_variants.py \
  --targets pets_targets/credit_defense_targets.csv \
  --out pets_targets/credit_defense_training_targets.csv \
  --l2-weight-decay 0.01 \
  --hamp-gamma 0.95 \
  --hamp-alpha 0.001 \
  --dp-target-epsilon 4.0 \
  --dp-max-grad-norm 1.0 \
  --dp-delta 1e-5 \
  --dp-batch-size 32 \
  --dp-epochs 30 \
  --dp-learning-rate 0.05

"${PYTHON_BIN}" pets_tools/build_defense_tuning_targets.py \
  --base-targets pets_targets/credit_defense_targets.csv \
  --block-id pets_b01 \
  --l2-weight-decays 0.001,0.0001 \
  --dp-epsilons "${QURIFT_PETS_DP_TUNING_EPSILONS}" \
  --dp-batch-size 32 \
  --dp-epochs 30 \
  --dp-learning-rate 0.05 \
  --dp-max-grad-norm 1.0 \
  --dp-delta 1e-5 \
  --out pets_targets/credit_defense_tuning_targets.csv

"${PYTHON_BIN}" pets_tools/run_defenses_multigpu.py \
  --targets pets_targets/credit_defense_tuning_targets.csv \
  --phase train \
  --gpus "${QURIFT_GPUS}" \
  --jobs-per-gpu "${QURIFT_JOBS_PER_GPU}" \
  --out-dir pets_results/tuning_launcher \
  --resume

"${PYTHON_BIN}" pets_tools/select_defense_tuning.py \
  --tuning-targets pets_targets/credit_defense_tuning_targets.csv \
  --run-root pets_runs \
  --source-training-targets pets_targets/credit_defense_training_targets.csv \
  --out-dir pets_results/tuning \
  --frozen-targets pets_targets/credit_defense_training_targets_confirmatory.csv \
  --development-block pets_b01 \
  --minimum-roc-auc 0.65 \
  --minimum-average-precision 0.30 \
  --minimum-minority-recall 0.02 \
  --minimum-balanced-accuracy 0.55

echo "[DONE] Utility-only tuning selection and confirmatory manifest are ready."
