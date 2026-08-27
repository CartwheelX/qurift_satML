#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
QURIFT_GPUS="${QURIFT_GPUS:-auto}"
QURIFT_JOBS_PER_GPU="${QURIFT_JOBS_PER_GPU:-auto}"
QURIFT_PETS_LIRA_REFS="${QURIFT_PETS_LIRA_REFS:-8}"
: "${QURIFT_PETS_STICKY_SECRET:?Set QURIFT_PETS_STICKY_SECRET to the original pilot value}"

# Move the pre-label-matching pilot result directories into a recoverable,
# timestamped archive. Target and LiRA-reference checkpoints remain in place.
"${PYTHON_BIN}" pets_tools/archive_incompatible_pilot_results.py \
  --targets pets_targets/pilot_credit_training_targets.csv \
  --result-root pets_results/defenses \
  --analysis-dir pets_results/pilot_analysis

# Recompute only partition-dependent evaluation artifacts. Existing trained
# target checkpoints are reused by --phase evaluate.
"${PYTHON_BIN}" pets_tools/run_defenses_multigpu.py \
  --targets pets_targets/pilot_credit_training_targets.csv \
  --phase evaluate \
  --gpus "${QURIFT_GPUS}" \
  --jobs-per-gpu "${QURIFT_JOBS_PER_GPU}" \
  --resume

# Refresh the nearby-query experiment with common perturbations and matched
# classical controls.
"${PYTHON_BIN}" pets_tools/run_adaptive_attacks_multigpu.py \
  --attack query_stress \
  --targets pets_targets/pilot_credit_output_defense_targets.csv \
  --defenses none,logitguard_continuous,logitguard_quantized,measurementguard_continuous,lattice_round,memgq_lattice,memgq_lattice_sticky \
  --status-label query_stress_matched_controls \
  --archive-existing-query-stress \
  --gpus "${QURIFT_GPUS}" \
  --jobs-per-gpu "${QURIFT_JOBS_PER_GPU}" \
  --resume

"${PYTHON_BIN}" pets_tools/run_adaptive_attacks_multigpu.py \
  --attack hsj \
  --targets pets_targets/pilot_credit_output_defense_targets.csv \
  --defenses none,dynanoise,lattice_round,memgq_lattice_sticky \
  --status-label hsj_output_boundary_conditions \
  --hsj-records-per-class 20 \
  --max-queries 512 \
  --gpus "${QURIFT_GPUS}" \
  --jobs-per-gpu 1 \
  --resume

"${PYTHON_BIN}" pets_tools/filter_targets.py \
  --targets pets_targets/pilot_credit_training_targets.csv \
  --training-defenses l2,hamp_train,dp_qml \
  --out pets_targets/pilot_credit_training_defense_hsj_targets.csv

"${PYTHON_BIN}" pets_tools/run_adaptive_attacks_multigpu.py \
  --attack hsj \
  --targets pets_targets/pilot_credit_training_defense_hsj_targets.csv \
  --defenses none \
  --status-label hsj_training_defenses \
  --hsj-records-per-class 20 \
  --max-queries 512 \
  --gpus "${QURIFT_GPUS}" \
  --jobs-per-gpu 1 \
  --resume

"${PYTHON_BIN}" pets_tools/filter_targets.py \
  --targets pets_targets/pilot_credit_lira_targets.csv \
  --training-defenses none \
  --out pets_targets/pilot_credit_lira_output_defense_targets.csv

# Re-score the standard target and every matched output-defense condition
# through the existing standard reference checkpoints.
"${PYTHON_BIN}" pets_tools/run_adaptive_attacks_multigpu.py \
  --attack lira \
  --targets pets_targets/pilot_credit_lira_output_defense_targets.csv \
  --reference-dir pets_results/lira_references \
  --defenses none,dynanoise,hamp_output,memguard,logitguard_continuous,logitguard_quantized,measurementguard_continuous,lattice_round,memgq_lattice,memgq_lattice_sticky \
  --status-label lira_output_defenses \
  --num-references "${QURIFT_PETS_LIRA_REFS}" \
  --gpus "${QURIFT_GPUS}" \
  --jobs-per-gpu 1 \
  --resume

"${PYTHON_BIN}" pets_tools/filter_targets.py \
  --targets pets_targets/pilot_credit_lira_targets.csv \
  --training-defenses l2 \
  --out pets_targets/pilot_credit_lira_l2_targets.csv

"${PYTHON_BIN}" pets_tools/run_adaptive_attacks_multigpu.py \
  --attack lira \
  --targets pets_targets/pilot_credit_lira_l2_targets.csv \
  --reference-dir pets_results/lira_references \
  --defenses none \
  --status-label lira_l2_training \
  --num-references "${QURIFT_PETS_LIRA_REFS}" \
  --gpus "${QURIFT_GPUS}" \
  --jobs-per-gpu 1 \
  --resume

"${PYTHON_BIN}" pets_tools/validate_protocol.py \
  --targets pets_targets/pilot_credit_training_targets.csv \
  --run-root pets_runs \
  --result-root pets_results/defenses \
  --out pets_results/pilot_protocol_validation.json

"${PYTHON_BIN}" pets_tools/analyze_defenses.py \
  --results-dir pets_results/defenses \
  --out-dir pets_results/pilot_analysis \
  --bootstrap 10000

echo "[DONE] Matched LiRA/query controls and imbalanced-utility pilot tables are ready."
