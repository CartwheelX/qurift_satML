#!/usr/bin/env bash
# Backward-compatible entry point for the corrected prospective protocol.
# The earlier draft rewrote inspected pilot manifests and checkpoints; all
# finalization now lives under isolated pets_v2_* paths.
set -euo pipefail

exec bash commands/pets_run_credit_confirmatory_v2.sh "${1:-all}"
