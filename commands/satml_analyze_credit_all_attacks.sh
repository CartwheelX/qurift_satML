#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TARGETS="satml_targets/credit_factorial_targets.csv"

"${PYTHON_BIN}" -u satml_tools/analyze_paired_factorial.py \
  --targets "${TARGETS}" \
  --metrics satml_results/credit_factorial/target_metrics/retrained_target_metrics_raw.csv \
  --attack-results satml_results/credit_factorial/threshold_mia/threshold_mia_raw.csv \
  --attack-results satml_results/credit_factorial/learned_mia/attack_summary.csv \
  --attack-results satml_results/credit_factorial/lira/lira_reference_mia_raw.csv \
  --attack-results satml_results/credit_factorial/label_only_correctness/label_only_correctness_raw.csv \
  --attack-results satml_results/credit_factorial/label_only_hsj/label_only_hsj_raw.csv \
  --out-dir satml_results/credit_factorial/paired_all_attacks \
  --bootstrap 10000 \
  --bootstrap-seed 2026
