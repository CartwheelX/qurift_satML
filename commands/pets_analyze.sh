#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIRMATORY_TARGETS="${QURIFT_PETS_CONFIRMATORY_TARGETS:-pets_targets/credit_defense_training_targets_confirmatory.csv}"

"${PYTHON_BIN}" pets_tools/validate_protocol.py \
  --targets pets_targets/pilot_credit_training_targets.csv \
  --run-root pets_runs \
  --result-root pets_results/defenses \
  --out pets_results/pilot_protocol_validation.json

"${PYTHON_BIN}" pets_tools/validate_protocol.py \
  --targets "${CONFIRMATORY_TARGETS}" \
  --run-root pets_runs \
  --result-root pets_results/defenses \
  --out pets_results/confirmatory_protocol_validation.json

"${PYTHON_BIN}" pets_tools/analyze_defenses.py \
  --results-dir pets_results/defenses \
  --out-dir pets_results/analysis \
  --bootstrap 10000 \
  --seed 2026 \
  --exclude-block pets_b01

echo "[DONE] PETS validation, tables, and figures are under pets_results/analysis."
