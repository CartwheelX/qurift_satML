#!/usr/bin/env bash
# Plan/archive corrected PETS-v2 evaluation outputs and securely load the new
# sticky secret for the Stage-4/6 reruns. Nothing destructive is the default.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
ACTION="${1:-plan}"
if [[ $# -gt 0 ]]; then
  shift
fi
ARCHIVE_WORKER_LOGS=0
for option in "$@"; do
  case "${option}" in
    --archive-worker-logs) ARCHIVE_WORKER_LOGS=1 ;;
    *) echo "Unknown option: ${option}" >&2; exit 2 ;;
  esac
done
case "${ACTION}" in
  plan|archive|stage4|stage6) ;;
  *)
    echo "Usage: $0 [plan|archive|stage4|stage6] [--archive-worker-logs]" >&2
    exit 2
    ;;
esac

REASON="${QURIFT_PETS_REFRESH_REASON:-pre-analysis correction: enforce the frozen binary decision rule across target, defense, and attack evaluation}"
SECRET_FILE="${QURIFT_PETS_V2_SECRET_FILE:-pets_v2_logs/.confirmatory_v2_refresh_secret}"
ARCHIVER=(
  "${PYTHON_BIN}" pets_tools/archive_confirmatory_v2_evaluation.py
  --targets pets_v2_targets/credit_confirmatory_training_targets.csv
  --result-root pets_v2_results/defenses
  --run-root pets_v2_runs
  --reference-root pets_v2_results/lira_references
  --log-root pets_v2_logs
  --archive-root pets_v2_results/archive/evaluation_refresh
  --reason "${REASON}"
)
if [[ "${ARCHIVE_WORKER_LOGS}" -eq 1 ]]; then
  ARCHIVER+=(--archive-worker-logs)
fi

case "${ACTION}" in
  plan)
    "${ARCHIVER[@]}" --dry-run
    ;;
  archive)
    if ! command -v openssl >/dev/null 2>&1; then
      echo "openssl is required to generate the replacement sticky secret." >&2
      exit 69
    fi
    if [[ -e "${SECRET_FILE}" ]]; then
      echo "Refusing to overwrite existing secret file: ${SECRET_FILE}" >&2
      exit 2
    fi
    secret_parent="$(dirname -- "${SECRET_FILE}")"
    mkdir -p "${secret_parent}"
    umask 077
    secret_temporary="$(mktemp "${SECRET_FILE}.tmp.XXXXXX")"
    trap 'rm -f -- "${secret_temporary}"' EXIT
    openssl rand -hex 32 > "${secret_temporary}"
    chmod 600 "${secret_temporary}"
    # Prepare the secret first, but publish it only after a successful archive.
    # A failed archiver removes the temporary through the EXIT trap.
    "${ARCHIVER[@]}" --execute
    mv -- "${secret_temporary}" "${SECRET_FILE}"
    trap - EXIT
    echo "[OK] generated a new 256-bit sticky secret at ${SECRET_FILE} (mode 600; value not printed)"
    echo "[NEXT] nohup bash $0 stage4 >> pets_v2_logs/stage4_evaluation_refresh.log 2>&1 &"
    ;;
  stage4|stage6)
    if [[ ! -f "${SECRET_FILE}" ]]; then
      echo "Missing ${SECRET_FILE}; complete the archive action first." >&2
      exit 2
    fi
    secret_mode="$(stat -c '%a' "${SECRET_FILE}")"
    if (( 10#${secret_mode} % 100 != 0 )); then
      echo "Secret file must not be readable or writable by group/other: ${SECRET_FILE}" >&2
      exit 2
    fi
    IFS= read -r QURIFT_PETS_STICKY_SECRET < "${SECRET_FILE}"
    if [[ ! "${QURIFT_PETS_STICKY_SECRET}" =~ ^[[:xdigit:]]{64}$ ]]; then
      echo "Secret file does not contain the expected 256-bit hexadecimal value." >&2
      exit 2
    fi
    export QURIFT_PETS_STICKY_SECRET
    if [[ "${ACTION}" == "stage4" ]]; then
      exec bash commands/pets_run_credit_confirmatory_v2.sh 4
    else
      exec bash commands/pets_run_credit_confirmatory_v2.sh 6
    fi
    ;;
esac
