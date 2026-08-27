#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p pets_targets pets_runs pets_results pets_logs

"${PYTHON_BIN}" pets_tools/build_defense_targets.py \
  --source-targets satml_targets/credit_factorial_targets.csv \
  --low-cell eff_su2_r1_d6 \
  --high-cell eff_su2_r5_d6 \
  --blocks 5 \
  --out pets_targets/credit_defense_targets.csv

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

"${PYTHON_BIN}" pets_tools/validate_protocol.py \
  --targets pets_targets/credit_defense_training_targets.csv \
  --target-only \
  --out pets_targets/protocol_validation.json

if ! "${PYTHON_BIN}" -c 'import opacus; assert opacus.__version__' >/dev/null 2>&1; then
  echo "[ACTION] Install the audited DP accountant before training DP-QML:"
  echo "         ${PYTHON_BIN} -m pip install -r requirements-pets.txt"
fi

echo "[DONE] PETS target manifests are prepared and validated."
