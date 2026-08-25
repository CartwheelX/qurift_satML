#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
QURIFT_GPUS="${QURIFT_GPUS:-auto}"
QURIFT_LABEL_JOBS_PER_GPU="${QURIFT_LABEL_JOBS_PER_GPU:-1}"
QURIFT_LABEL_MAX_QUERIES="${QURIFT_LABEL_MAX_QUERIES:-512}"
QURIFT_LABEL_INIT_QUERIES="${QURIFT_LABEL_INIT_QUERIES:-128}"
mkdir -p satml_logs satml_results/fashion_factorial satml_results/wdbc_targeted

run_learned() {
  local dataset="$1" run_dir="$2" result_dir="$3"
  "${PYTHON_BIN}" -u experiments/gen_results/run_train_mia_attack_cvholdout_multigpu.py \
    --launcher --attack-data-dir "${run_dir}" --out "${result_dir}/learned_mia" \
    --test-ratio 0.2 --cv-folds 5 --tune --n-trials 20 --max-epochs 150 \
    --patience 15 --device cuda --seed 2026 --cpu-threads 2 --resume \
    --jobs-per-gpu 1 --gpus "${QURIFT_GPUS}" \
    2>&1 | tee "satml_logs/${dataset}_learned_mia.log"
}

run_label_only() {
  local dataset="$1" targets="$2" members="$3" nonmembers="$4" result_dir="$5"
  "${PYTHON_BIN}" -u reviewer_tools/run_label_only_hsj_multigpu.py \
    --targets "${targets}" --repo-root . --run-root satml_runs \
    --out-dir "${result_dir}/label_only_hsj" --n-member "${members}" \
    --n-nonmember "${nonmembers}" --max-queries "${QURIFT_LABEL_MAX_QUERIES}" \
    --init-queries "${QURIFT_LABEL_INIT_QUERIES}" --init-batch-size 32 \
    --iterations 8 --gradient-samples 32 --binary-steps 10 --step-search-steps 10 \
    --bootstrap 10000 --seed 2026 --gpus "${QURIFT_GPUS}" \
    --jobs-per-gpu "${QURIFT_LABEL_JOBS_PER_GPU}" \
    --cpu-threads 2 --resume 2>&1 | tee "satml_logs/${dataset}_label_only_hsj.log"
}

run_label_correctness() {
  local dataset="$1" targets="$2" run_dir="$3" result_dir="$4"
  "${PYTHON_BIN}" -u reviewer_tools/label_only_correctness_attack.py \
    --attack-data-dir "${run_dir}" --targets "${targets}" \
    --out-dir "${result_dir}/label_only_correctness" --bootstrap 10000 --seed 2026 \
    2>&1 | tee "satml_logs/${dataset}_label_only_correctness.log"
}

run_lira_subset() {
  local dataset="$1" targets="$2" result_dir="$3"
  "${PYTHON_BIN}" -u reviewer_tools/run_lira_reference_multigpu.py \
    --targets "${targets}" --repo-root . --run-root satml_runs \
    --out-dir "${result_dir}/lira_representative" --num-references 16 \
    --bootstrap 10000 --seed 2026 --gpus "${QURIFT_GPUS}" --jobs-per-gpu 1 \
    --cpu-threads 2 --resume 2>&1 | tee "satml_logs/${dataset}_lira_representative.log"
}

run_learned fashion satml_runs/satml_fashion_factorial satml_results/fashion_factorial
run_label_correctness fashion satml_targets/fashion_factorial_targets.csv satml_runs/satml_fashion_factorial satml_results/fashion_factorial
run_label_only fashion satml_targets/fashion_factorial_targets.csv 200 200 satml_results/fashion_factorial
run_lira_subset fashion satml_targets/fashion_lira_targets.csv satml_results/fashion_factorial

run_learned wdbc satml_runs/satml_wdbc_targeted satml_results/wdbc_targeted
run_label_correctness wdbc satml_targets/wdbc_targeted_targets.csv satml_runs/satml_wdbc_targeted satml_results/wdbc_targeted
run_label_only wdbc satml_targets/wdbc_targeted_targets.csv 160 160 satml_results/wdbc_targeted
run_lira_subset wdbc satml_targets/wdbc_lira_targets.csv satml_results/wdbc_targeted

"${PYTHON_BIN}" -u satml_tools/analyze_paired_factorial.py \
  --targets satml_targets/fashion_factorial_targets.csv \
  --metrics satml_results/fashion_factorial/target_metrics/retrained_target_metrics_raw.csv \
  --attack-results satml_results/fashion_factorial/threshold_mia/threshold_mia_raw.csv \
  --attack-results satml_results/fashion_factorial/learned_mia/attack_summary.csv \
  --attack-results satml_results/fashion_factorial/lira_representative/lira_reference_mia_raw.csv \
  --attack-results satml_results/fashion_factorial/label_only_correctness/label_only_correctness_raw.csv \
  --attack-results satml_results/fashion_factorial/label_only_hsj/label_only_hsj_raw.csv \
  --out-dir satml_results/fashion_factorial/paired_all_attacks \
  --bootstrap 10000 --bootstrap-seed 2026

"${PYTHON_BIN}" -u satml_tools/analyze_paired_factorial.py \
  --targets satml_targets/wdbc_targeted_targets.csv \
  --metrics satml_results/wdbc_targeted/target_metrics/retrained_target_metrics_raw.csv \
  --attack-results satml_results/wdbc_targeted/threshold_mia/threshold_mia_raw.csv \
  --attack-results satml_results/wdbc_targeted/learned_mia/attack_summary.csv \
  --attack-results satml_results/wdbc_targeted/lira_representative/lira_reference_mia_raw.csv \
  --attack-results satml_results/wdbc_targeted/label_only_correctness/label_only_correctness_raw.csv \
  --attack-results satml_results/wdbc_targeted/label_only_hsj/label_only_hsj_raw.csv \
  --out-dir satml_results/wdbc_targeted/paired_all_attacks \
  --bootstrap 10000 --bootstrap-seed 2026
