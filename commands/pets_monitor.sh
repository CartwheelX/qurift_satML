#!/usr/bin/env bash
set -euo pipefail

echo "GPU processes"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader || true

echo
echo "Launcher status"
find pets_results -type f -name '*status.csv' -print -exec tail -n 5 {} \; 2>/dev/null || true

echo
echo "Completed training metadata"
find pets_runs -type f -name training_metadata.json 2>/dev/null | wc -l

if [[ -f pets_targets/credit_defense_tuning_targets.csv ]]; then
  expected_tuning_targets=$(($(wc -l < pets_targets/credit_defense_tuning_targets.csv) - 1))
else
  expected_tuning_targets="manifest-missing"
fi
echo "Completed utility-tuning targets (expected ${expected_tuning_targets})"
find pets_runs/pets_credit_defense_tuning -type f -name training_metadata.json 2>/dev/null | wc -l

if [[ -f pets_results/tuning/selection.json ]]; then
  echo "Utility-only tuning selection: ready"
else
  echo "Utility-only tuning selection: pending"
fi

echo "Completed prediction-defense evaluations"
find pets_results/defenses -type f -name evaluation_metadata.json 2>/dev/null | wc -l

echo "Completed HSJ conditions"
find pets_results/defenses -type f -path '*/hsj/*_metrics.json' 2>/dev/null | wc -l

echo "Completed LiRA conditions"
find pets_results/defenses -type f -path '*/lira/*_metrics.json' 2>/dev/null | wc -l

echo "Completed nearby-query targets"
find pets_results/defenses -type f -path '*/query_stress/metrics.json' 2>/dev/null | wc -l
