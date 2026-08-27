#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
QURIFT_GPUS="${QURIFT_GPUS:-auto}"
: "${QURIFT_PETS_STICKY_SECRET:?Set QURIFT_PETS_STICKY_SECRET to the frozen pilot value}"

stamp="$(date +%Y%m%dT%H%M%S)"
archive="pets_results/pilot_hsj_before_common_random_numbers_${stamp}"
mkdir -p "${archive}"

mapfile -t target_ids < <(
  "${PYTHON_BIN}" - <<'PY'
import pandas as pd

paths = [
    "pets_targets/pilot_credit_output_defense_targets.csv",
    "pets_targets/pilot_credit_training_defense_hsj_targets.csv",
]
ids = []
for path in paths:
    ids.extend(pd.read_csv(path)["target_id"].astype(str).tolist())
for target_id in dict.fromkeys(ids):
    print(target_id)
PY
)

for target_id in "${target_ids[@]}"; do
  source="pets_results/defenses/${target_id}/hsj"
  if [[ -d "${source}" ]]; then
    mkdir -p "${archive}/${target_id}"
    mv "${source}" "${archive}/${target_id}/hsj"
  fi
done

echo "[OK] Previous pilot HSJ outputs archived under ${archive}"

"${PYTHON_BIN}" pets_tools/run_adaptive_attacks_multigpu.py \
  --attack hsj \
  --targets pets_targets/pilot_credit_output_defense_targets.csv \
  --defenses none,dynanoise,lattice_round,memgq_lattice,memgq_lattice_sticky \
  --hsj-records-per-class 20 \
  --max-queries 512 \
  --gpus "${QURIFT_GPUS}" \
  --jobs-per-gpu 1 \
  --resume

"${PYTHON_BIN}" pets_tools/run_adaptive_attacks_multigpu.py \
  --attack hsj \
  --targets pets_targets/pilot_credit_training_defense_hsj_targets.csv \
  --defenses none \
  --hsj-records-per-class 20 \
  --max-queries 512 \
  --gpus "${QURIFT_GPUS}" \
  --jobs-per-gpu 1 \
  --resume

"${PYTHON_BIN}" pets_tools/analyze_defenses.py \
  --results-dir pets_results/defenses \
  --out-dir pets_results/pilot_analysis \
  --bootstrap 10000

"${PYTHON_BIN}" pets_tools/validate_protocol.py \
  --targets pets_targets/pilot_credit_training_targets.csv \
  --run-root pets_runs \
  --result-root pets_results/defenses \
  --out pets_results/pilot_protocol_validation.json

echo "[DONE] Corrected common-random-number HSJ pilot results are ready."
