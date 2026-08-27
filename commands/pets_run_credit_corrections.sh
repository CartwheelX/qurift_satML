#!/usr/bin/env bash
set -euo pipefail

: "${QURIFT_PETS_STICKY_SECRET:?Set QURIFT_PETS_STICKY_SECRET to the original pilot value}"

# Keep independent GPU schedulers from racing for the same free-memory snapshot.
bash commands/pets_run_credit_pilot_extension.sh
bash commands/pets_run_credit_tuning.sh

echo "[DONE] Utility tuning and missing pilot controls are complete."
