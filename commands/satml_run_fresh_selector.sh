#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
QURIFT_GPUS="${QURIFT_GPUS:-auto}"
QURIFT_TRAIN_JOBS_PER_GPU="${QURIFT_TRAIN_JOBS_PER_GPU:-auto}"
QURIFT_ATTACK_JOBS_PER_GPU="${QURIFT_ATTACK_JOBS_PER_GPU:-3}"
QURIFT_LIRA_JOBS_PER_GPU="${QURIFT_LIRA_JOBS_PER_GPU:-auto}"
QURIFT_LABEL_JOBS_PER_GPU="${QURIFT_LABEL_JOBS_PER_GPU:-auto}"
QURIFT_LABEL_MAX_QUERIES="${QURIFT_LABEL_MAX_QUERIES:-512}"
QURIFT_LABEL_INIT_QUERIES="${QURIFT_LABEL_INIT_QUERIES:-128}"
TARGETS="satml_targets/selector/fresh_selector_targets.csv"
RUNS="satml_runs/satml_selector_fresh"
RESULTS="satml_results/selector_fresh"

mkdir -p "${RESULTS}" satml_logs
"${PYTHON_BIN}" -u reviewer_tools/run_multiseed_factorial.py \
  --targets "${TARGETS}" --repo-root . --out satml_runs \
  --gpus "${QURIFT_GPUS}" --jobs-per-gpu "${QURIFT_TRAIN_JOBS_PER_GPU}" \
  --cpu-threads 2 --resume \
  2>&1 | tee satml_logs/selector_fresh_train.log

selector_outputs_complete() {
  "${PYTHON_BIN}" - \
    "${TARGETS}" \
    "${RESULTS}/target_metrics/retrained_target_metrics_raw.csv" \
    "${RESULTS}/threshold_mia/threshold_mia_raw.csv" <<'PY'
from pathlib import Path
import sys

import pandas as pd

targets_path, metrics_path, attacks_path = map(Path, sys.argv[1:])
expected_attacks = {
    "confidence", "correctness", "entropy", "loss", "margin", "max_probability"
}
try:
    target_ids = set(pd.read_csv(targets_path)["target_id"].astype(str))
    metrics = pd.read_csv(metrics_path)
    attacks = pd.read_csv(attacks_path)
    if set(metrics["target_id"].astype(str)) != target_ids or len(metrics) != len(target_ids):
        raise ValueError("target-metrics coverage is incomplete")
    if "status" in metrics and not metrics["status"].astype(str).eq("ok").all():
        raise ValueError("target metrics contain a non-ok status")
    if set(attacks["target_id"].astype(str)) != target_ids:
        raise ValueError("threshold-MIA target coverage is incomplete")
    if set(attacks["attack"].astype(str)) != expected_attacks:
        raise ValueError("threshold-MIA attack coverage is incomplete")
    counts = attacks.groupby(["target_id", "attack"]).size()
    if len(counts) != len(target_ids) * len(expected_attacks) or not counts.eq(1).all():
        raise ValueError("threshold-MIA rows are missing or duplicated")
except (FileNotFoundError, KeyError, pd.errors.EmptyDataError, ValueError) as exc:
    print(f"[RESUME] Fresh-selector derived outputs need regeneration: {exc}")
    raise SystemExit(1)
print(
    f"[RESUME] Reusing complete fresh-selector outputs: "
    f"{len(target_ids)} targets and {len(attacks)} threshold rows."
)
PY
}

if ! selector_outputs_complete; then
  "${PYTHON_BIN}" reviewer_tools/extract_retrained_target_metrics.py \
    --attack-data-dir "${RUNS}" --targets "${TARGETS}" --out-dir "${RESULTS}/target_metrics"
  "${PYTHON_BIN}" reviewer_tools/threshold_mia_bootstrap.py \
    --attack-data-dir "${RUNS}" --targets "${TARGETS}" --out-dir "${RESULTS}/threshold_mia" \
    --bootstrap 10000 --bootstrap-seed 2026 --fprs 0.01,0.05,0.10
fi

"${PYTHON_BIN}" -u experiments/gen_results/run_train_mia_attack_cvholdout_multigpu.py \
  --launcher --attack-data-dir "${RUNS}" --out "${RESULTS}/learned_mia" \
  --test-ratio 0.2 --cv-folds 5 --tune --n-trials 20 --max-epochs 150 --patience 15 \
  --device cuda --seed 2026 --cpu-threads 2 --resume \
  --jobs-per-gpu "${QURIFT_ATTACK_JOBS_PER_GPU}" --gpus "${QURIFT_GPUS}"
"${PYTHON_BIN}" -u reviewer_tools/run_lira_reference_multigpu.py \
  --targets "${TARGETS}" --repo-root . --run-root satml_runs --out-dir "${RESULTS}/lira" \
  --num-references 16 --bootstrap 10000 --seed 2026 --gpus "${QURIFT_GPUS}" \
  --jobs-per-gpu "${QURIFT_LIRA_JOBS_PER_GPU}" --cpu-threads 2 --resume
"${PYTHON_BIN}" -u reviewer_tools/label_only_correctness_attack.py \
  --attack-data-dir "${RUNS}" --targets "${TARGETS}" \
  --out-dir "${RESULTS}/label_only_correctness" --bootstrap 10000 --seed 2026
"${PYTHON_BIN}" -u reviewer_tools/run_label_only_hsj_multigpu.py \
  --targets "${TARGETS}" --repo-root . --run-root satml_runs --out-dir "${RESULTS}/label_only_hsj" \
  --n-member 200 --n-nonmember 200 --max-queries "${QURIFT_LABEL_MAX_QUERIES}" \
  --init-queries "${QURIFT_LABEL_INIT_QUERIES}" --init-batch-size 32 --iterations 8 \
  --gradient-samples 32 --binary-steps 10 --step-search-steps 10 --bootstrap 10000 \
  --seed 2026 --gpus "${QURIFT_GPUS}" --jobs-per-gpu "${QURIFT_LABEL_JOBS_PER_GPU}" \
  --cpu-threads 2 --resume

"${PYTHON_BIN}" -m satml_tools.analyze_fresh_selector \
  --targets "${TARGETS}" \
  --metrics "${RESULTS}/target_metrics/retrained_target_metrics_raw.csv" \
  --attacks "${RESULTS}/threshold_mia/threshold_mia_raw.csv" \
  --attacks "${RESULTS}/learned_mia/attack_summary.csv" \
  --attacks "${RESULTS}/lira/lira_reference_mia_raw.csv" \
  --attacks "${RESULTS}/label_only_correctness/label_only_correctness_raw.csv" \
  --attacks "${RESULTS}/label_only_hsj/label_only_hsj_raw.csv" \
  --out-dir "${RESULTS}/paired_analysis" --bootstrap 10000 --bootstrap-seed 2026
