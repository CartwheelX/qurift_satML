#!/usr/bin/env bash
# Prospective, isolated PETS defense evaluation.  No stage writes into the
# inspected pets_runs/ or pets_results/ trees.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
QURIFT_GPUS="${QURIFT_GPUS:-auto}"
QURIFT_JOBS_PER_GPU="${QURIFT_JOBS_PER_GPU:-auto}"
QURIFT_PETS_LIRA_REFS="${QURIFT_PETS_LIRA_REFS:-16}"
QURIFT_PETS_LOW_FPR_NONMEMBER_MULTIPLIER="${QURIFT_PETS_LOW_FPR_NONMEMBER_MULTIPLIER:-10}"
TARGETS="pets_v2_targets/credit_confirmatory_training_targets.csv"
RUN_ROOT="pets_v2_runs"
RESULTS="pets_v2_results/defenses"
REFERENCES="pets_v2_results/lira_references"
LOGS="pets_v2_logs"
STAGE="${1:-all}"

case "${STAGE}" in
  all|1|2|3|4|5|6|7) ;;
  *) echo "Usage: $0 [all|1|2|3|4|5|6|7]" >&2; exit 2 ;;
esac

# Hold a nonblocking advisory lock for every stage this invocation can run.
# An `all` invocation takes all seven locks up front, so it cannot overlap a
# separately launched numbered stage (and vice versa). File locks are released
# by the kernel when this shell exits, including after an uncatchable SIGKILL;
# the human-readable owner files are diagnostics only and may safely be stale.
if ! command -v flock >/dev/null 2>&1; then
  echo "Required command 'flock' is unavailable; refusing an unlocked run." >&2
  exit 69
fi
mkdir -p "${LOGS}"
declare -a PETS_V2_LOCK_FDS=()
declare -a PETS_V2_LOCK_OWNERS=()

release_stage_locks() {
  local owner
  for owner in "${PETS_V2_LOCK_OWNERS[@]:-}"; do
    if [[ -f "${owner}" ]] && grep -qx "pid=$$" "${owner}"; then
      rm -f -- "${owner}"
    fi
  done
}
trap release_stage_locks EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

acquire_stage_lock() {
  local stage_number="$1"
  local lock_path="${LOGS}/.confirmatory_v2_stage_${stage_number}.lock"
  local owner_path="${lock_path}.owner"
  local lock_fd holder

  # Append mode avoids changing the lock inode before the ownership test. The
  # lock file intentionally stays empty; owner metadata lives beside it.
  exec {lock_fd}>>"${lock_path}"
  if ! flock -n "${lock_fd}"; then
    holder="owner metadata unavailable"
    if [[ -r "${owner_path}" ]]; then
      holder="$(tr '\n' ' ' < "${owner_path}")"
    fi
    echo "[LOCKED] PETS v2 stage ${stage_number} is already active (${holder})." >&2
    exec {lock_fd}>&-
    return 75
  fi
  printf 'pid=%s\nstarted_at=%s\nrequested_stage=%s\nlocked_stage=%s\n' \
    "$$" "$(date --iso-8601=seconds)" "${STAGE}" "${stage_number}" \
    > "${owner_path}"
  PETS_V2_LOCK_FDS+=("${lock_fd}")
  PETS_V2_LOCK_OWNERS+=("${owner_path}")
}

if [[ "${STAGE}" == "all" ]]; then
  for stage_number in 1 2 3 4 5 6 7; do
    acquire_stage_lock "${stage_number}"
  done
else
  acquire_stage_lock "${STAGE}"
fi

if [[ "${QURIFT_PETS_LIRA_REFS}" -ne 16 ]]; then
  echo "The headline protocol is frozen at 16 LiRA references; got ${QURIFT_PETS_LIRA_REFS}." >&2
  exit 2
fi
if [[ "${QURIFT_PETS_LOW_FPR_NONMEMBER_MULTIPLIER}" -ne 10 ]]; then
  echo "The headline protocol is frozen at a 10x final non-member pool; got ${QURIFT_PETS_LOW_FPR_NONMEMBER_MULTIPLIER}." >&2
  exit 2
fi

run_stage() {
  [[ "${STAGE}" == "all" || "${STAGE}" == "$1" ]]
}

if run_stage 4 || run_stage 6; then
  : "${QURIFT_PETS_STICKY_SECRET:?Set QURIFT_PETS_STICKY_SECRET to the frozen private pilot value}"
  case "${QURIFT_PETS_STICKY_SECRET,,}" in
    your_same_frozen_pilot_secret|\
    '<the same private value used for the pilot>'|\
    the-same-private-pilot-secret|\
    the-same-private-secret-used-for-the-pilot|\
    replace-with-a-private-random-value|\
    your-secret-here|placeholder|changeme|change-me|secret|test)
      echo "QURIFT_PETS_STICKY_SECRET is an example placeholder, not the frozen private value." >&2
      exit 2
      ;;
  esac
fi

if run_stage 1; then
  echo "== stage 1/7: isolated prospective manifest =="
  bash commands/pets_prepare_confirmatory_v2.sh
fi

if [[ ! -f "${TARGETS}" ]]; then
  echo "Missing ${TARGETS}; run stage 1 first." >&2
  exit 2
fi

if run_stage 2; then
  echo "== stage 2/7: train 3 roles x 8 blocks x 4 training arms =="
  "${PYTHON_BIN}" pets_tools/run_defenses_multigpu.py \
    --targets "${TARGETS}" \
    --repo-root . \
    --run-root "${RUN_ROOT}" \
    --out-dir "${RESULTS}" \
    --logs-dir "${LOGS}" \
    --phase train \
    --gpus "${QURIFT_GPUS}" \
    --jobs-per-gpu "${QURIFT_JOBS_PER_GPU}" \
    --resume
fi

if run_stage 3; then
  echo "== stage 3/7: checkpoint/manifest integrity gate =="
  "${PYTHON_BIN}" pets_tools/check_protocol_integrity.py \
    --targets "${TARGETS}" \
    --run-root "${RUN_ROOT}" \
    --results-root "${RESULTS}" \
    --reference-root "${REFERENCES}" \
    --allow-missing-evaluations \
    --out pets_v2_results/protocol_integrity_after_training.csv
fi

if run_stage 4; then
  echo "== stage 4/7: adaptive scalar/learned attacks, utility, HSJ, query stress =="
  "${PYTHON_BIN}" pets_tools/run_defenses_multigpu.py \
    --targets "${TARGETS}" \
    --repo-root . \
    --run-root "${RUN_ROOT}" \
    --out-dir "${RESULTS}" \
    --logs-dir "${LOGS}" \
    --phase evaluate \
    --evaluation-nonmember-multiplier "${QURIFT_PETS_LOW_FPR_NONMEMBER_MULTIPLIER}" \
    --gpus "${QURIFT_GPUS}" \
    --jobs-per-gpu "${QURIFT_JOBS_PER_GPU}" \
    --resume

  "${PYTHON_BIN}" pets_tools/filter_targets.py \
    --targets "${TARGETS}" \
    --training-defenses none \
    --out pets_v2_targets/credit_output_defense_targets.csv
  "${PYTHON_BIN}" pets_tools/run_adaptive_attacks_multigpu.py \
    --attack query_stress \
    --targets pets_v2_targets/credit_output_defense_targets.csv \
    --repo-root . --run-root "${RUN_ROOT}" --out-dir "${RESULTS}" --logs-dir "${LOGS}" \
    --defenses none,logitguard_continuous,logitguard_quantized,measurementguard_continuous,lattice_round,memgq_lattice,memgq_lattice_sticky \
    --status-label query_stress_matched_controls \
    --gpus "${QURIFT_GPUS}" --jobs-per-gpu "${QURIFT_JOBS_PER_GPU}" --resume
  "${PYTHON_BIN}" pets_tools/run_adaptive_attacks_multigpu.py \
    --attack hsj \
    --targets pets_v2_targets/credit_output_defense_targets.csv \
    --repo-root . --run-root "${RUN_ROOT}" --out-dir "${RESULTS}" --logs-dir "${LOGS}" \
    --defenses none,dynanoise,lattice_round,memgq_lattice_sticky \
    --status-label hsj_output_boundary_conditions \
    --gpus "${QURIFT_GPUS}" --jobs-per-gpu 1 --resume
  "${PYTHON_BIN}" pets_tools/filter_targets.py \
    --targets "${TARGETS}" \
    --training-defenses l2,hamp_train,dp_qml \
    --out pets_v2_targets/credit_training_defense_hsj_targets.csv
  "${PYTHON_BIN}" pets_tools/run_adaptive_attacks_multigpu.py \
    --attack hsj \
    --targets pets_v2_targets/credit_training_defense_hsj_targets.csv \
    --repo-root . --run-root "${RUN_ROOT}" --out-dir "${RESULTS}" --logs-dir "${LOGS}" \
    --defenses none --status-label hsj_training_defenses \
    --gpus "${QURIFT_GPUS}" --jobs-per-gpu 1 --resume
fi

if run_stage 5; then
  echo "== stage 5/7: clean 16-reference LiRA banks for every training arm =="
  "${PYTHON_BIN}" reviewer_tools/run_lira_reference_multigpu.py \
    --targets "${TARGETS}" \
    --repo-root . \
    --run-root "${RUN_ROOT}" \
    --out-dir "${REFERENCES}" \
    --num-references "${QURIFT_PETS_LIRA_REFS}" \
    --save-reference-checkpoints \
    --phase train \
    --gpus "${QURIFT_GPUS}" \
    --jobs-per-gpu "${QURIFT_JOBS_PER_GPU}" \
    --resume
fi

if run_stage 6; then
  echo "== stage 6/7: defended LiRA scoring =="
  "${PYTHON_BIN}" pets_tools/filter_targets.py \
    --targets "${TARGETS}" --training-defenses none \
    --out pets_v2_targets/credit_lira_output_targets.csv
  "${PYTHON_BIN}" pets_tools/run_adaptive_attacks_multigpu.py \
    --attack lira --targets pets_v2_targets/credit_lira_output_targets.csv \
    --repo-root . --run-root "${RUN_ROOT}" --out-dir "${RESULTS}" --logs-dir "${LOGS}" \
    --reference-dir "${REFERENCES}" --num-references "${QURIFT_PETS_LIRA_REFS}" \
    --defenses none,dynanoise,memguard,logitguard_continuous,logitguard_quantized,measurementguard_continuous,lattice_round,memgq_lattice,memgq_lattice_sticky \
    --status-label lira_output_defenses \
    --gpus "${QURIFT_GPUS}" --jobs-per-gpu 1 --resume

  "${PYTHON_BIN}" pets_tools/filter_targets.py \
    --targets "${TARGETS}" --training-defenses l2,dp_qml \
    --out pets_v2_targets/credit_lira_l2_dp_targets.csv
  "${PYTHON_BIN}" pets_tools/run_adaptive_attacks_multigpu.py \
    --attack lira --targets pets_v2_targets/credit_lira_l2_dp_targets.csv \
    --repo-root . --run-root "${RUN_ROOT}" --out-dir "${RESULTS}" --logs-dir "${LOGS}" \
    --reference-dir "${REFERENCES}" --num-references "${QURIFT_PETS_LIRA_REFS}" \
    --defenses none --status-label lira_l2_dp_training \
    --gpus "${QURIFT_GPUS}" --jobs-per-gpu 1 --resume

  "${PYTHON_BIN}" pets_tools/filter_targets.py \
    --targets "${TARGETS}" --training-defenses hamp_train \
    --out pets_v2_targets/credit_lira_hamp_targets.csv
  "${PYTHON_BIN}" pets_tools/run_adaptive_attacks_multigpu.py \
    --attack lira --targets pets_v2_targets/credit_lira_hamp_targets.csv \
    --repo-root . --run-root "${RUN_ROOT}" --out-dir "${RESULTS}" --logs-dir "${LOGS}" \
    --reference-dir "${REFERENCES}" --num-references "${QURIFT_PETS_LIRA_REFS}" \
    --defenses none,hamp_output --status-label lira_hamp_training_and_full \
    --gpus "${QURIFT_GPUS}" --jobs-per-gpu 1 --resume
fi

if run_stage 7; then
  echo "== stage 7/7: final integrity gate and paired-block analysis =="
  "${PYTHON_BIN}" pets_tools/check_protocol_integrity.py \
    --targets "${TARGETS}" \
    --run-root "${RUN_ROOT}" \
    --results-root "${RESULTS}" \
    --reference-root "${REFERENCES}" \
    --expected-references 16 \
    --out pets_v2_results/protocol_integrity_final.csv
  "${PYTHON_BIN}" pets_tools/analyze_confirmatory_v2.py \
    --results-dir "${RESULTS}" \
    --out-dir pets_v2_results/analysis \
    --bootstrap 10000 \
    --seed 2027
fi

echo "[OK] PETS v2 stage=${STAGE} complete"
