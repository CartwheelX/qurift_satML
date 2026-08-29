#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIRMATORY_TARGETS="${QURIFT_PETS_CONFIRMATORY_TARGETS:-pets_targets/credit_defense_training_targets_confirmatory.csv}"

# pets_b01 is a development-only diagnostic and contains the superseded
# pre-v3 DP checkpoint.  Do not mislabel it as passing the frozen confirmatory
# training protocol; the analysis below excludes it explicitly.
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

echo "[DONE] Confirmatory PETS validation, tables, and figures are under pets_results/analysis."
