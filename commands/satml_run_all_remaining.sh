#!/usr/bin/env bash
set -Eeuo pipefail

# Unattended, resumable SaTML pipeline after the 96-target Credit factorial.
# Scientific stage definitions remain in the individual commands/*.sh files;
# this wrapper supplies ordering, early external preflight, durable stage
# markers, and one master log.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
QURIFT_GPUS="${QURIFT_GPUS:-auto}"
QURIFT_JOBS_PER_GPU="${QURIFT_JOBS_PER_GPU:-1}"
QURIFT_NOISE_JOBS_PER_GPU="${QURIFT_NOISE_JOBS_PER_GPU:-auto}"
QURIFT_NOISE_SHARDS_PER_TARGET="${QURIFT_NOISE_SHARDS_PER_TARGET:-0}"
QURIFT_TRAIN_JOBS_PER_GPU="${QURIFT_TRAIN_JOBS_PER_GPU:-auto}"
QURIFT_ATTACK_JOBS_PER_GPU="${QURIFT_ATTACK_JOBS_PER_GPU:-3}"
QURIFT_LIRA_JOBS_PER_GPU="${QURIFT_LIRA_JOBS_PER_GPU:-auto}"
QURIFT_LABEL_JOBS_PER_GPU="${QURIFT_LABEL_JOBS_PER_GPU:-auto}"
QURIFT_GPU_MONITOR_INTERVAL="${QURIFT_GPU_MONITOR_INTERVAL:-15}"
QURIFT_LEGACY_REPO="${QURIFT_LEGACY_REPO:-/home/najeeb/quarift_neurips_rebutal_2}"
QURIFT_INCLUDE_OPTIONAL_NOISY_LABEL="${QURIFT_INCLUDE_OPTIONAL_NOISY_LABEL:-0}"
QURIFT_MASTER_FORCE="${QURIFT_MASTER_FORCE:-0}"
QURIFT_MIN_FREE_GB="${QURIFT_MIN_FREE_GB:-50}"
RUN_TAG="${QURIFT_MASTER_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
STATE_DIR="satml_results/unattended_pipeline"
MARKER_DIR="${STATE_DIR}/completed_stages"
STATUS_FILE="${STATE_DIR}/events_${RUN_TAG}.tsv"
CURRENT_FILE="${STATE_DIR}/current_stage.txt"
PID_FILE="${STATE_DIR}/pipeline.pid"
SNAPSHOT_STATE_FILE="${STATE_DIR}/frozen_snapshot_path.txt"
MASTER_LOG="satml_logs/satml_all_remaining_${RUN_TAG}.log"
GPU_TELEMETRY_FILE="satml_logs/gpu_telemetry_${RUN_TAG}.csv"
GPU_MONITOR_PID=""
DRY_RUN=0

usage() {
  printf '%s\n' \
    'Usage: bash commands/satml_run_all_remaining.sh [--dry-run]' \
    '' \
    'Required unless QURIFT_NOISE_SNAPSHOT already names a valid frozen snapshot:' \
    '  QURIFT_NOISE_BACKEND and a working saved IBM account or QISKIT_IBM_TOKEN.' \
    '' \
    'Useful environment variables:' \
    '  QURIFT_GPUS=0,1,2,3,4,5,6,7' \
    '  QURIFT_JOBS_PER_GPU=1' \
    '  QURIFT_TRAIN_JOBS_PER_GPU=auto' \
    '  QURIFT_ATTACK_JOBS_PER_GPU=3' \
    '  QURIFT_LIRA_JOBS_PER_GPU=auto' \
    '  QURIFT_LABEL_JOBS_PER_GPU=auto' \
    '  QURIFT_GPU_MONITOR_INTERVAL=15  # seconds; 0 disables telemetry' \
    '  QURIFT_NOISE_JOBS_PER_GPU=auto' \
    '  QURIFT_NOISE_SHARDS_PER_TARGET=0  # auto: 4 for CPU Aer, 1 for GPU Aer' \
    '  QURIFT_LEGACY_REPO=/path/to/quarift_neurips_rebutal_2' \
    '  QURIFT_NOISE_SNAPSHOT=/path/to/frozen/snapshot' \
    '  QURIFT_INCLUDE_OPTIONAL_NOISY_LABEL=1' \
    '  QURIFT_MIN_FREE_GB=50' \
    '  QURIFT_MASTER_FORCE=1  # rerun stages already marked complete'
}

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
elif [[ $# -gt 0 ]]; then
  usage >&2
  exit 2
fi

mkdir -p "${STATE_DIR}" "${MARKER_DIR}" satml_logs

if [[ -s "${PID_FILE}" ]]; then
  EXISTING_PID="$(tr -dc '0-9' < "${PID_FILE}")"
  if [[ -n "${EXISTING_PID}" ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
    printf '[ERROR] Pipeline already running with PID %s\n' "${EXISTING_PID}" >&2
    exit 1
  fi
fi
printf '%s\n' "$$" > "${PID_FILE}"
printf 'timestamp_utc\tstage\tstatus\tdetail\n' > "${STATUS_FILE}"
printf '%s\n' "${MASTER_LOG}" > "${STATE_DIR}/latest_log_path.txt"
printf '%s\n' "${STATUS_FILE}" > "${STATE_DIR}/latest_status_path.txt"
ln -sfn "$(basename "${MASTER_LOG}")" satml_logs/satml_all_remaining_latest.log
exec > >(tee -a "${MASTER_LOG}") 2>&1

export PYTHON_BIN QURIFT_GPUS QURIFT_JOBS_PER_GPU QURIFT_NOISE_JOBS_PER_GPU
export QURIFT_NOISE_SHARDS_PER_TARGET
export QURIFT_TRAIN_JOBS_PER_GPU QURIFT_ATTACK_JOBS_PER_GPU
export QURIFT_LIRA_JOBS_PER_GPU QURIFT_LABEL_JOBS_PER_GPU
export QURIFT_LEGACY_REPO CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1

ACTIVE_STAGE="startup"
PIPELINE_SUCCEEDED=0

record_event() {
  local stage="$1" status="$2" detail="${3:-}"
  detail="${detail//$'\t'/ }"
  detail="${detail//$'\n'/ }"
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${stage}" "${status}" "${detail}" >> "${STATUS_FILE}"
}

finish_pipeline() {
  local code=$?
  if [[ ${PIPELINE_SUCCEEDED} -eq 1 && ${code} -eq 0 ]]; then
    if [[ ${DRY_RUN} -eq 1 ]]; then
      printf '%s\n' dry_run_complete > "${CURRENT_FILE}"
      record_event pipeline dry_run_complete "all required stages were planned"
      printf '[DRY DONE] All required stages were planned. Master log: %s\n' "${MASTER_LOG}"
    else
      printf '%s\n' complete > "${CURRENT_FILE}"
      record_event pipeline complete "all required stages completed"
      printf '[DONE] All required SaTML stages completed. Master log: %s\n' "${MASTER_LOG}"
    fi
  else
    printf '%s\n' "failed:${ACTIVE_STAGE}" > "${CURRENT_FILE}"
    record_event pipeline failed "stage=${ACTIVE_STAGE}; exit_code=${code}"
    printf '[FAILED] stage=%s exit_code=%s log=%s\n' "${ACTIVE_STAGE}" "${code}" "${MASTER_LOG}" >&2
  fi
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
}
trap finish_pipeline EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_gpu_monitor() {
  if [[ "${QURIFT_GPU_MONITOR_INTERVAL}" == "0" ]]; then
    printf '%s\n' '[MONITOR] GPU telemetry disabled.'
    return
  fi
  "${PYTHON_BIN}" -u satml_tools/monitor_pipeline_gpus.py \
    --pid-file "${PID_FILE}" --stage-file "${CURRENT_FILE}" \
    --out "${GPU_TELEMETRY_FILE}" --interval "${QURIFT_GPU_MONITOR_INTERVAL}" &
  GPU_MONITOR_PID=$!
  printf '[MONITOR] GPU telemetry: %s (pid=%s)\n' \
    "${GPU_TELEMETRY_FILE}" "${GPU_MONITOR_PID}"
}

run_stage() {
  local stage="$1"
  shift
  local marker="${MARKER_DIR}/${stage}.done"
  ACTIVE_STAGE="${stage}"
  printf '%s\n' "${stage}" > "${CURRENT_FILE}"

  if [[ ${DRY_RUN} -eq 1 ]]; then
    record_event "${stage}" planned "$*"
    printf '[DRY] stage=%s command=' "${stage}"
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi

  if [[ -s "${marker}" && "${QURIFT_MASTER_FORCE}" != "1" ]]; then
    record_event "${stage}" skipped "completion marker exists"
    printf '[SKIP] stage=%s already completed; marker=%s\n' "${stage}" "${marker}"
    return 0
  fi

  record_event "${stage}" running "$*"
  printf '\n[STAGE] %s started at %s\n' "${stage}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if "$@"; then
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${marker}"
    record_event "${stage}" complete "$*"
    printf '[STAGE] %s completed at %s\n' "${stage}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  else
    local code=$?
    record_event "${stage}" failed "exit_code=${code}; command=$*"
    printf '[ERROR] stage=%s exit_code=%s\n' "${stage}" "${code}" >&2
    return "${code}"
  fi
}

verify_credit_factorial() {
  local progress
  progress="$("${PYTHON_BIN}" satml_tools/progress.py \
    --targets satml_targets/credit_factorial_targets.csv --run-root satml_runs)"
  printf '%s\n' "${progress}"
  [[ "${progress}" == *"targets=96 complete=96"* ]]
  [[ "${progress}" == *"errors=0"* ]]
}

verify_snapshot() {
  "${PYTHON_BIN}" - "${QURIFT_NOISE_SNAPSHOT}" <<'PY'
from pathlib import Path
import sys
from reviewer_tools.qurift_qiskit_bridge import load_backend_noise_snapshot

path = Path(sys.argv[1]).resolve()
context = load_backend_noise_snapshot(path, require_noise=True)
print(
    f"[SNAPSHOT VERIFIED] path={path} "
    f"backend={context.metadata.resolved_backend_name} "
    f"calibration={context.metadata.calibration_timestamp}"
)
PY
}

preflight() {
  ACTIVE_STAGE="preflight"
  printf '%s\n' preflight > "${CURRENT_FILE}"
  record_event preflight running "checking environment and immutable inputs"

  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
    printf '[ERROR] Python executable not found: %s\n' "${PYTHON_BIN}" >&2
    return 1
  }
  [[ -f satml_targets/credit_factorial_targets.csv ]] || {
    printf '%s\n' '[ERROR] Run from a prepared qurift_satML repository.' >&2
    return 1
  }
  verify_credit_factorial || {
    printf '%s\n' '[ERROR] The 96-target Credit factorial is not complete.' >&2
    return 1
  }
  if [[ "${QURIFT_GPUS}" == "auto" ]]; then
    command -v nvidia-smi >/dev/null 2>&1 || {
      printf '%s\n' '[ERROR] QURIFT_GPUS=auto but nvidia-smi is unavailable.' >&2
      return 1
    }
  fi
  [[ "${QURIFT_MIN_FREE_GB}" =~ ^[0-9]+$ ]] || {
    printf '[ERROR] QURIFT_MIN_FREE_GB must be a non-negative integer, got %s\n' \
      "${QURIFT_MIN_FREE_GB}" >&2
    return 1
  }
  local free_kb required_kb
  free_kb="$(df -Pk . | awk 'NR == 2 {print $4}')"
  required_kb=$((QURIFT_MIN_FREE_GB * 1024 * 1024))
  if (( free_kb < required_kb )); then
    printf '[ERROR] Only %s GB free; unattended minimum is %s GB.\n' \
      "$((free_kb / 1024 / 1024))" "${QURIFT_MIN_FREE_GB}" >&2
    printf '%s\n' '[ERROR] Free space or explicitly lower QURIFT_MIN_FREE_GB after checking capacity.' >&2
    return 1
  fi
  printf '[PREFLIGHT] GPUs=%s jobs_per_gpu=%s noise_jobs_per_gpu=%s free_gb=%s\n' \
    "${QURIFT_GPUS}" "${QURIFT_JOBS_PER_GPU}" "${QURIFT_NOISE_JOBS_PER_GPU}" \
    "$((free_kb / 1024 / 1024))"

  if [[ ${DRY_RUN} -eq 1 ]]; then
    printf '[DRY] would import 36 retained MNIST checkpoints from %s\n' "${QURIFT_LEGACY_REPO}"
    if [[ -n "${QURIFT_NOISE_SNAPSHOT:-}" ]]; then
      printf '[DRY] would validate frozen snapshot %s\n' "${QURIFT_NOISE_SNAPSHOT}"
    else
      printf '[DRY] would capture one snapshot from backend %s\n' "${QURIFT_NOISE_BACKEND:-<unset>}"
    fi
    record_event preflight planned "mutating checks omitted in dry run"
    return 0
  fi

  [[ -d "${QURIFT_LEGACY_REPO}" ]] || {
    printf '[ERROR] Legacy NeurIPS repository not found: %s\n' "${QURIFT_LEGACY_REPO}" >&2
    return 1
  }
  bash commands/satml_import_legacy_mnist.sh

  if [[ -z "${QURIFT_NOISE_SNAPSHOT:-}" && -s "${SNAPSHOT_STATE_FILE}" ]]; then
    QURIFT_NOISE_SNAPSHOT="$(realpath "$(head -n 1 "${SNAPSHOT_STATE_FILE}")")"
    export QURIFT_NOISE_SNAPSHOT
    printf '[PREFLIGHT] Reusing pipeline-frozen snapshot: %s\n' "${QURIFT_NOISE_SNAPSHOT}"
  elif [[ -n "${QURIFT_NOISE_SNAPSHOT:-}" && -s "${SNAPSHOT_STATE_FILE}" ]]; then
    local requested_snapshot stored_snapshot
    requested_snapshot="$(realpath "${QURIFT_NOISE_SNAPSHOT}")"
    stored_snapshot="$(realpath "$(head -n 1 "${SNAPSHOT_STATE_FILE}")")"
    if [[ "${requested_snapshot}" != "${stored_snapshot}" ]]; then
      printf '[ERROR] This pipeline is frozen to snapshot %s, not %s.\n' \
        "${stored_snapshot}" "${requested_snapshot}" >&2
      printf '%s\n' '[ERROR] Do not mix snapshots across resumed N1/N2/N3 stages.' >&2
      return 1
    fi
    QURIFT_NOISE_SNAPSHOT="${stored_snapshot}"
    export QURIFT_NOISE_SNAPSHOT
  fi

  if [[ -z "${QURIFT_NOISE_SNAPSHOT:-}" ]]; then
    : "${QURIFT_NOISE_BACKEND:?Set QURIFT_NOISE_BACKEND or provide QURIFT_NOISE_SNAPSHOT}"
    local snapshot_tag="unattended_${RUN_TAG}"
    export QURIFT_NOISE_SNAPSHOT_TAG="${snapshot_tag}"
    export QURIFT_NOISE_SNAPSHOT="${REPO_ROOT}/satml_results/backend_snapshots/${snapshot_tag}"
    bash commands/satml_capture_noise_snapshot.sh
  else
    QURIFT_NOISE_SNAPSHOT="$(realpath "${QURIFT_NOISE_SNAPSHOT}")"
    export QURIFT_NOISE_SNAPSHOT
  fi
  verify_snapshot
  printf '%s\n' "${QURIFT_NOISE_SNAPSHOT}" > "${SNAPSHOT_STATE_FILE}"

  # No later stage needs live IBM credentials. Removing them from the child
  # environment prevents accidental credential propagation into long logs.
  unset QISKIT_IBM_TOKEN IBM_QUANTUM_TOKEN QISKIT_IBM_INSTANCE IBM_QUANTUM_INSTANCE
  record_event preflight complete "credit=96/96; imported_mnist=36; snapshot=${QURIFT_NOISE_SNAPSHOT}"
}

verify_repository() {
  PYTHONPATH=.:reviewer_tools "${PYTHON_BIN}" -m unittest discover \
    -s test -p 'test_satml_*.py' -q
  "${PYTHON_BIN}" -m py_compile satml_tools/*.py reviewer_tools/*.py \
    experiments/qurift_main.py
  bash -n commands/*.sh
  git diff --check
}

printf '[START] Unattended SaTML pipeline run_tag=%s repo=%s\n' "${RUN_TAG}" "${REPO_ROOT}"
printf '[MONITOR] current stage: cat %s\n' "${CURRENT_FILE}"
printf '[MONITOR] master log: tail -f %s\n' "${MASTER_LOG}"
printf '[MONITOR] events: column -ts $'"'\t'"' %s\n' "${STATUS_FILE}"

start_gpu_monitor
preflight

run_stage prepare bash commands/satml_prepare.sh
run_stage credit_factorial_analysis bash commands/satml_analyze_credit_factorial.sh
run_stage credit_geometry bash commands/satml_run_credit_geometry.sh
run_stage credit_mechanism bash commands/satml_analyze_mechanism.sh
run_stage fashion_factorial bash commands/satml_run_fashion_factorial.sh
run_stage fashion_analysis bash commands/satml_analyze_fashion.sh
run_stage wdbc_targeted bash commands/satml_run_wdbc_targeted.sh
run_stage wdbc_analysis bash commands/satml_analyze_wdbc.sh
run_stage added_geometry bash commands/satml_run_added_geometry.sh
run_stage added_mechanisms bash commands/satml_analyze_added_mechanisms.sh
run_stage credit_attacks bash commands/satml_run_credit_attacks.sh
run_stage added_attacks bash commands/satml_run_added_attacks.sh
run_stage encoding_scale bash commands/satml_run_encoding_scale.sh

# The selector decision is immutable once fresh outcomes may have been seen.
if [[ -s satml_targets/selector/selection_decision.json && \
      -s satml_targets/selector/fresh_selector_targets.csv && \
      "${QURIFT_MASTER_FORCE}" != "1" ]]; then
  printf '%s\n' selector_freeze > "${CURRENT_FILE}"
  record_event selector_freeze skipped "existing frozen selector decision retained"
  printf '%s\n' '[SKIP] Retaining the existing frozen selector decision.'
else
  run_stage selector_freeze bash commands/satml_build_selector.sh
fi
run_stage selector_fresh bash commands/satml_run_fresh_selector.sh

run_stage noise_n1_structural bash commands/satml_noise_n1_structural.sh
run_stage noise_n2_query_policy bash commands/satml_noise_n2_queries.sh
run_stage noise_n3_lira bash commands/satml_noise_n3_attacks.sh

if [[ "${QURIFT_INCLUDE_OPTIONAL_NOISY_LABEL}" == "1" ]]; then
  run_stage noise_n3_label_only_optional bash commands/satml_noise_n3_label_only_optional.sh
else
  record_event noise_n3_label_only_optional skipped \
    "optional pilot; set QURIFT_INCLUDE_OPTIONAL_NOISY_LABEL=1 to include"
  printf '%s\n' '[SKIP] Optional noisy label-only pilot was not requested.'
fi

run_stage paper_artifacts bash commands/satml_generate_artifacts.sh
run_stage repository_verification verify_repository

PIPELINE_SUCCEEDED=1
ACTIVE_STAGE="complete"
