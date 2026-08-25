#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
QURIFT_GPUS="${QURIFT_GPUS:-auto}"
QURIFT_LABEL_JOBS_PER_GPU="${QURIFT_LABEL_JOBS_PER_GPU:-1}"
QURIFT_LABEL_PILOT_MEMBERS="${QURIFT_LABEL_PILOT_MEMBERS:-20}"
QURIFT_LABEL_PILOT_NONMEMBERS="${QURIFT_LABEL_PILOT_NONMEMBERS:-20}"
QURIFT_LABEL_PILOT_DRY_RUN="${QURIFT_LABEL_PILOT_DRY_RUN:-0}"
TARGETS="satml_targets/credit_factorial_targets.csv"
OUT_ROOT="satml_results/credit_factorial/label_only_hsj_pilot"

if [[ "${QURIFT_LABEL_PILOT_DRY_RUN}" == "1" ]]; then
  OUT_ROOT="$(mktemp -d /tmp/qurift_label_hsj_pilot_dry_run.XXXXXX)"
fi

target_args=(
  --target-id CREDIT_QNN_eff_su2_r1_d2_b07
  --target-id CREDIT_QNN_z_r1_d2_b03
  --target-id CREDIT_QNN_z_r5_d6_b06
)

mkdir -p "${OUT_ROOT}" satml_logs

run_budget() {
  local budget="$1" init_queries="$2" iterations="$3" gradient_samples="$4" binary_steps="$5"
  local launcher_args=()
  if [[ "${QURIFT_LABEL_PILOT_DRY_RUN}" == "1" ]]; then
    launcher_args+=(--dry-run)
  fi
  "${PYTHON_BIN}" -u reviewer_tools/run_label_only_hsj_multigpu.py \
    --targets "${TARGETS}" --repo-root . --run-root satml_runs \
    --out-dir "${OUT_ROOT}/q${budget}" "${target_args[@]}" \
    --n-member "${QURIFT_LABEL_PILOT_MEMBERS}" \
    --n-nonmember "${QURIFT_LABEL_PILOT_NONMEMBERS}" \
    --max-queries "${budget}" --init-queries "${init_queries}" --init-batch-size 32 \
    --iterations "${iterations}" --gradient-samples "${gradient_samples}" \
    --binary-steps "${binary_steps}" --step-search-steps 10 \
    --bootstrap 1000 --seed 2026 --gpus "${QURIFT_GPUS}" \
    --jobs-per-gpu "${QURIFT_LABEL_JOBS_PER_GPU}" --cpu-threads 2 --resume \
    "${launcher_args[@]}" \
    2>&1 | tee "satml_logs/credit_label_only_hsj_pilot_q${budget}.log"
}

# Candidate identities and deterministic seed rules are common across budgets.
# Each budget-specific search schedule is declared here in advance; settings
# are not selected using target-set attack AUC.
run_budget 128 64 2 16 8
run_budget 512 128 8 32 10
run_budget 2500 256 18 96 12

if [[ "${QURIFT_LABEL_PILOT_DRY_RUN}" == "1" ]]; then
  printf '%s\n' "[DRY-RUN] Pilot commands validated under ${OUT_ROOT}; no analysis was generated."
  exit 0
fi

"${PYTHON_BIN}" -u satml_tools/analyze_label_only_hsj_pilot.py \
  --pilot-root "${OUT_ROOT}" \
  2>&1 | tee satml_logs/credit_label_only_hsj_pilot_analysis.log

printf '%s\n' "[OK] Pilot analysis: ${OUT_ROOT}/analysis"
