#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p pets_v2_targets pets_v2_runs pets_v2_results pets_v2_logs

"${PYTHON_BIN}" pets_tools/build_confirmatory_targets.py \
  --source-targets satml_targets/credit_factorial_targets.csv \
  --exclude-targets pets_targets/credit_defense_targets.csv \
  --exclude-targets pets_targets/credit_defense_training_targets.csv \
  --blocks 8 \
  --data-seed-start 90261 \
  --model-seed-start 100261 \
  --selection pets_results/tuning/selection.json \
  --stress-evidence satml_results/credit_factorial/lira/lira_reference_mia_raw.csv \
  --structural-out pets_v2_targets/credit_confirmatory_structural_targets.csv \
  --out pets_v2_targets/credit_confirmatory_training_targets.csv

"${PYTHON_BIN}" pets_tools/validate_confirmatory_manifest.py \
  --targets pets_v2_targets/credit_confirmatory_training_targets.csv \
  --prior-targets satml_targets/credit_factorial_targets.csv \
  --prior-targets pets_targets/credit_defense_targets.csv \
  --prior-targets pets_targets/credit_defense_training_targets.csv \
  --out pets_v2_targets/confirmatory_manifest_validation.json

echo "[DONE] Isolated PETS v2 manifests are prepared; no model was trained."
