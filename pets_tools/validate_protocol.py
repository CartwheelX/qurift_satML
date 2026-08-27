#!/usr/bin/env python3
"""Fail-closed validator for PETS target and result artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qurift.defenses.protocol import PARTITION_PROTOCOL


EVALUATION_PROTOCOL = "pets_defense_evaluation_v3"


def validate_targets(targets: pd.DataFrame) -> Dict[str, Any]:
    required = {
        "target_id",
        "block_id",
        "defense_structural_role",
        "structural_cell_id",
        "training_defense",
        "data_seed",
        "model_seed",
    }
    missing = required - set(targets.columns)
    if missing:
        raise ValueError(f"training target table lacks {sorted(missing)}")
    if targets.target_id.duplicated().any():
        raise ValueError("duplicate target IDs")
    expected_variants = {"none", "l2", "hamp_train", "dp_qml"}
    for (block, role), group in targets.groupby(["block_id", "defense_structural_role"]):
        if set(group.training_defense.astype(str)) != expected_variants:
            raise ValueError(f"{block}/{role} lacks a training-defense variant")
        if group.data_seed.nunique() != 1 or group.model_seed.nunique() != 1:
            raise ValueError(f"{block}/{role} variants do not share seeds")
    for block, group in targets.groupby("block_id"):
        if set(group.defense_structural_role.astype(str)) != {"low", "high"}:
            raise ValueError(f"{block} does not contain paired low/high structures")
        if group.groupby("defense_structural_role").data_seed.first().nunique() != 1:
            raise ValueError(f"{block} low/high pair does not share data seed")
        if group.groupby("defense_structural_role").model_seed.first().nunique() != 1:
            raise ValueError(f"{block} low/high pair does not share model seed")
    dp = targets[targets.training_defense.astype(str).eq("dp_qml")]
    required_dp = {
        "dp_optimizer": "rmsprop",
        "dp_batch_size": 32,
        "dp_epochs": 30,
        "dp_learning_rate": 0.05,
        "dp_scheduler": "none",
        "dp_loss": "unweighted_nll",
        "dp_max_grad_norm": 1.0,
    }
    for column, expected in required_dp.items():
        if column not in dp or not dp[column].eq(expected).all():
            raise ValueError(
                f"DP-QML manifest must set {column}={expected!r} for every DP target"
            )
    return {
        "targets": len(targets),
        "blocks": targets.block_id.nunique(),
        "structural_cells": sorted(targets.structural_cell_id.unique().tolist()),
        "training_defenses": sorted(targets.training_defense.unique().tolist()),
    }


def validate_results(targets: pd.DataFrame, run_root: Path, result_root: Path) -> Dict[str, Any]:
    missing_models = []
    missing_evaluations = []
    invalid_dp = []
    invalid_partitions = []
    task_label_mismatches = []
    invalid_utility_scope = []
    incompatible_protocols = []
    for _, row in targets.iterrows():
        target_id = str(row.target_id)
        model_dir = run_root / str(row.experiment) / target_id
        if not (model_dir / "target_model.pt").exists():
            missing_models.append(target_id)
        training_metadata = model_dir / "training_metadata.json"
        if str(row.training_defense) == "dp_qml":
            training_payload = (
                json.loads(training_metadata.read_text())
                if training_metadata.exists()
                else {}
            )
            privacy = training_payload.get("privacy")
            decision_rule = training_payload.get("decision_rule")
            target_epsilon = float(row.get("dp_target_epsilon", 4.0))
            if (
                not privacy
                or privacy.get("formal_dp_claim") is not True
                or privacy.get("sampler") != "poisson"
                or privacy.get("accountant") != "rdp"
                or float(privacy.get("epsilon", float("inf"))) > target_epsilon + 1e-3
                or privacy.get("empty_step_behavior")
                != "gaussian_update_to_zero_clipped_sum"
                or privacy.get("randomness_streams")
                != "independent_sampler_and_noise_generators"
                or training_payload.get("optimizer") != "rmsprop"
                or training_payload.get("scheduler") != "none"
                or int(training_payload.get("batch_size", -1)) != 32
                or int(training_payload.get("epochs", -1)) != 30
                or not isinstance(decision_rule, dict)
                or decision_rule.get("selection_split") != "valid"
                or int(decision_rule.get("test_records_consulted", -1)) != 0
            ):
                invalid_dp.append(target_id)
        target_result = result_root / target_id
        if not (target_result / "adaptive_attack_metrics.csv").exists() or not (
            target_result / "evaluation_metadata.json"
        ).exists() or not (target_result / "test_utility_predictions.csv").exists():
            missing_evaluations.append(target_id)
        metadata_path = target_result / "evaluation_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("protocol") != EVALUATION_PROTOCOL:
                incompatible_protocols.append(target_id)
            utility = metadata.get("utility_evaluation", {})
            if (
                utility.get("scope") != "full_held_out_test_split"
                or int(utility.get("records", -1)) != int(row.get("vector_test", -1))
                or int(utility.get("member_records_included", -1)) != 0
            ):
                invalid_utility_scope.append(target_id)
        partition = target_result / "partition_manifest.json"
        if partition.exists():
            partition_document = json.loads(partition.read_text())
            if partition_document.get("protocol") != PARTITION_PROTOCOL:
                incompatible_protocols.append(target_id)
            payload = partition_document["partitions"]
            sets = {
                name: {item["record_id"] for item in values}
                for name, values in payload.items()
            }
            names = list(sets)
            if any(sets[left] & sets[right] for i, left in enumerate(names) for right in names[i + 1 :]):
                invalid_partitions.append(target_id)
            for values in payload.values():
                member = Counter(
                    int(item["task_label"])
                    for item in values
                    if int(item["membership"]) == 1
                )
                nonmember = Counter(
                    int(item["task_label"])
                    for item in values
                    if int(item["membership"]) == 0
                )
                if member != nonmember:
                    task_label_mismatches.append(target_id)
                    break
    failures = {
        "missing_models": missing_models,
        "missing_evaluations": missing_evaluations,
        "invalid_dp_reports": invalid_dp,
        "overlapping_partitions": invalid_partitions,
        "task_label_mismatches": sorted(set(task_label_mismatches)),
        "invalid_utility_scope": invalid_utility_scope,
        "incompatible_protocols": sorted(set(incompatible_protocols)),
    }
    if any(failures.values()):
        raise RuntimeError(json.dumps(failures, indent=2))
    return {"complete_targets": len(targets), "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("pets_targets/credit_defense_training_targets.csv"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("pets_runs"))
    parser.add_argument("--result-root", type=Path, default=Path("pets_results/defenses"))
    parser.add_argument("--target-only", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("pets_results/protocol_validation.json"))
    args = parser.parse_args()
    targets = pd.read_csv(args.targets)
    payload = {"protocol": "pets_fail_closed_validation_v3", "targets": validate_targets(targets)}
    if not args.target_only:
        payload["results"] = validate_results(targets, args.run_root, args.result_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[OK] {args.out.resolve()}")


if __name__ == "__main__":
    main()
