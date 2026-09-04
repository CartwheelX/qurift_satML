#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
QURIFT_GPUS="${QURIFT_GPUS:-auto}"
QURIFT_JOBS_PER_GPU="${QURIFT_JOBS_PER_GPU:-auto}"
QURIFT_PETS_LIRA_REFS="${QURIFT_PETS_LIRA_REFS:-16}"
CONFIRMATORY_TARGETS="${QURIFT_PETS_CONFIRMATORY_TARGETS:-pets_targets/credit_defense_training_targets_confirmatory.csv}"
: "${QURIFT_PETS_STICKY_SECRET:?Set QURIFT_PETS_STICKY_SECRET to the frozen pilot value}"

if [[ "${QURIFT_PETS_PILOT_FROZEN:-0}" != "1" ]]; then
  echo "Set QURIFT_PETS_PILOT_FROZEN=1 only after recording the accepted pilot settings." >&2
  exit 2
fi
if [[ ! -f "${CONFIRMATORY_TARGETS}" || ! -f pets_results/tuning/selection.json ]]; then
  echo "Run commands/pets_run_credit_tuning.sh successfully before confirmation." >&2
  exit 2
fi

"${PYTHON_BIN}" pets_tools/run_defenses_multigpu.py \
  --targets "${CONFIRMATORY_TARGETS}" \
  --phase all \
  --gpus "${QURIFT_GPUS}" \
  --jobs-per-gpu "${QURIFT_JOBS_PER_GPU}" \
  --resume

"${PYTHON_BIN}" pets_tools/filter_targets.py \
  --targets "${CONFIRMATORY_TARGETS}" \
  --training-defenses none \
  --out pets_targets/credit_output_defense_targets.csv

"${PYTHON_BIN}" pets_tools/run_adaptive_attacks_multigpu.py \
  --attack query_stress \
  --targets pets_targets/credit_output_defense_targets.csv \
  --defenses none,logitguard_continuous,logitguard_quantized,measurementguard_continuous,lattice_round,memgq_lattice,memgq_lattice_sticky \
  --status-label query_stress_matched_controls \
  --gpus "${QURIFT_GPUS}" \
  --jobs-per-gpu "${QURIFT_JOBS_PER_GPU}" \
  --resume

"${PYTHON_BIN}" pets_tools/run_adaptive_attacks_multigpu.py \
  --attack hsj \
  --targets pets_targets/credit_output_defense_targets.csv \
  --defenses none,dynanoise,lattice_round,memgq_lattice_sticky \
  --status-label hsj_output_boundary_conditions \
  --gpus "${QURIFT_GPUS}" \
  --jobs-per-gpu 1 \
  --resume

# Training-time defenses can change the genuine hard-label boundary.  Evaluate
# those trained checkpoints directly; HAMP-full is label-preserving, so its HSJ
# result is exactly the HAMP-train result and is not redundantly recomputed.
"${PYTHON_BIN}" pets_tools/filter_targets.py \
  --targets "${CONFIRMATORY_TARGETS}" \
  --training-defenses l2,hamp_train,dp_qml \
  --out pets_targets/credit_training_defense_hsj_targets.csv

"${PYTHON_BIN}" pets_tools/run_adaptive_attacks_multigpu.py \
  --attack hsj \
  --targets pets_targets/credit_training_defense_hsj_targets.csv \
  --defenses none \
  --status-label hsj_training_defenses \
  --gpus "${QURIFT_GPUS}" \
  --jobs-per-gpu 1 \
  --resume

# Legacy four-block launcher: preserve its original standard/L2 LiRA scope.
# The prospective v2 launcher evaluates matched references for every arm.
"${PYTHON_BIN}" pets_tools/filter_targets.py \
  --targets "${CONFIRMATORY_TARGETS}" \
  --training-defenses none,l2 \
  --out pets_targets/credit_lira_targets.csv

"${PYTHON_BIN}" reviewer_tools/run_lira_reference_multigpu.py \
  --targets pets_targets/credit_lira_targets.csv \
  --repo-root . \
  --run-root pets_runs \
  --out-dir pets_results/lira_references \
  --num-references "${QURIFT_PETS_LIRA_REFS}" \
  --save-reference-checkpoints \
  --phase train \
  --gpus "${QURIFT_GPUS}" \
  --jobs-per-gpu "${QURIFT_JOBS_PER_GPU}" \
  --resume

"${PYTHON_BIN}" pets_tools/filter_targets.py \
  --targets pets_targets/credit_lira_targets.csv \
  --training-defenses none \
  --out pets_targets/credit_lira_output_defense_targets.csv

"${PYTHON_BIN}" pets_tools/run_adaptive_attacks_multigpu.py \
  --attack lira \
  --targets pets_targets/credit_lira_output_defense_targets.csv \
  --reference-dir pets_results/lira_references \
  --defenses none,dynanoise,hamp_output,memguard,logitguard_continuous,logitguard_quantized,measurementguard_continuous,lattice_round,memgq_lattice,memgq_lattice_sticky \
  --status-label lira_output_defenses \
  --num-references "${QURIFT_PETS_LIRA_REFS}" \
  --gpus "${QURIFT_GPUS}" \
  --jobs-per-gpu 1 \
  --resume

"${PYTHON_BIN}" pets_tools/filter_targets.py \
  --targets pets_targets/credit_lira_targets.csv \
  --training-defenses l2 \
  --out pets_targets/credit_lira_l2_targets.csv

"${PYTHON_BIN}" pets_tools/run_adaptive_attacks_multigpu.py \
  --attack lira \
  --targets pets_targets/credit_lira_l2_targets.csv \
  --reference-dir pets_results/lira_references \
  --defenses none \
  --status-label lira_l2_training \
  --num-references "${QURIFT_PETS_LIRA_REFS}" \
  --gpus "${QURIFT_GPUS}" \
  --jobs-per-gpu 1 \
  --resume

bash commands/pets_analyze.sh
