#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
QURIFT_GPUS="${QURIFT_GPUS:-auto}"
QURIFT_TRAIN_JOBS_PER_GPU="${QURIFT_TRAIN_JOBS_PER_GPU:-auto}"
TARGETS="satml_targets/credit_scaling_targets.csv"
RUNS="satml_runs/satml_credit_scaling"
RESULTS="satml_results/encoding_scale"

mkdir -p "${RESULTS}" satml_logs
"${PYTHON_BIN}" -u reviewer_tools/run_multiseed_factorial.py \
  --targets "${TARGETS}" --repo-root . --out satml_runs \
  --gpus "${QURIFT_GPUS}" --jobs-per-gpu "${QURIFT_TRAIN_JOBS_PER_GPU}" \
  --cpu-threads 2 --resume \
  2>&1 | tee satml_logs/encoding_scale_train.log

encoding_outputs_complete() {
  "${PYTHON_BIN}" - \
    "${TARGETS}" \
    "${RESULTS}/target_metrics/retrained_target_metrics_raw.csv" \
    "${RESULTS}/threshold_mia/threshold_mia_raw.csv" <<'PY'
from pathlib import Path
import sys

import pandas as pd

targets_path, metrics_path, attacks_path = map(Path, sys.argv[1:])
expected_attacks = {
    "confidence",
    "correctness",
    "entropy",
    "loss",
    "margin",
    "max_probability",
}
try:
    target_ids = set(pd.read_csv(targets_path)["target_id"].astype(str))
    metrics = pd.read_csv(metrics_path)
    attacks = pd.read_csv(attacks_path)
    metrics_ids = set(metrics["target_id"].astype(str))
    attack_ids = set(attacks["target_id"].astype(str))
    if metrics_ids != target_ids or len(metrics) != len(target_ids):
        raise ValueError("target-metrics coverage is incomplete")
    if "status" in metrics and not metrics["status"].astype(str).eq("ok").all():
        raise ValueError("target metrics contain a non-ok status")
    if attack_ids != target_ids:
        raise ValueError("threshold-MIA target coverage is incomplete")
    observed_attacks = set(attacks["attack"].astype(str))
    if observed_attacks != expected_attacks:
        raise ValueError("threshold-MIA attack coverage is incomplete")
    counts = attacks.groupby(["target_id", "attack"]).size()
    if len(counts) != len(target_ids) * len(expected_attacks) or not counts.eq(1).all():
        raise ValueError("threshold-MIA rows are missing or duplicated")
except (FileNotFoundError, KeyError, pd.errors.EmptyDataError, ValueError) as exc:
    print(f"[RESUME] Encoding-scale derived outputs need regeneration: {exc}")
    raise SystemExit(1)
print(
    f"[RESUME] Reusing complete encoding-scale outputs: "
    f"{len(target_ids)} targets and {len(attacks)} threshold rows."
)
PY
}

if ! encoding_outputs_complete; then
  "${PYTHON_BIN}" reviewer_tools/extract_retrained_target_metrics.py \
    --attack-data-dir "${RUNS}" --targets "${TARGETS}" --out-dir "${RESULTS}/target_metrics"
  "${PYTHON_BIN}" reviewer_tools/threshold_mia_bootstrap.py \
    --attack-data-dir "${RUNS}" --targets "${TARGETS}" --out-dir "${RESULTS}/threshold_mia" \
    --bootstrap 10000 --bootstrap-seed 2026 --fprs 0.01,0.05,0.10
fi

"${PYTHON_BIN}" -m satml_tools.analyze_encoding_scale \
  --factorial-targets satml_targets/credit_factorial_targets.csv \
  --scaling-targets "${TARGETS}" \
  --factorial-metrics satml_results/credit_factorial/target_metrics/retrained_target_metrics_raw.csv \
  --scaling-metrics "${RESULTS}/target_metrics/retrained_target_metrics_raw.csv" \
  --factorial-attacks satml_results/credit_factorial/threshold_mia/threshold_mia_raw.csv \
  --scaling-attacks "${RESULTS}/threshold_mia/threshold_mia_raw.csv" \
  --out-dir "${RESULTS}/paired_analysis" --bootstrap 10000 --bootstrap-seed 2026
