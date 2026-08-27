#!/usr/bin/env python3
"""Build an isolated, utility-only L2/DP tuning manifest from the pilot block."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_positive_grid(text: str, label: str) -> list[float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values or any(not value > 0 for value in values):
        raise ValueError(f"{label} must contain positive comma-separated values")
    return list(dict.fromkeys(values))


def value_token(value: float) -> str:
    return f"{float(value):.8g}".replace("-", "m").replace(".", "p")


def build_tuning_targets(
    base: pd.DataFrame,
    *,
    block_id: str,
    l2_weight_decays: list[float],
    dp_epsilons: list[float],
    dp_batch_size: int = 32,
    dp_epochs: int = 30,
    dp_learning_rate: float = 0.05,
    dp_max_grad_norm: float = 1.0,
    dp_delta: float = 1e-5,
) -> pd.DataFrame:
    if (
        dp_batch_size <= 0
        or dp_epochs <= 0
        or dp_learning_rate <= 0
        or dp_max_grad_norm <= 0
        or not 0 < dp_delta < 1
    ):
        raise ValueError("invalid Watkins DP optimization or privacy parameters")
    selected = base[base.block_id.astype(str).eq(str(block_id))].copy()
    if selected.empty:
        raise ValueError(f"no base targets found for tuning block {block_id!r}")
    rows = []
    for _, source in selected.iterrows():
        source_id = str(source["target_id"])
        for weight_decay in l2_weight_decays:
            row = source.copy()
            row["source_target_id"] = source_id
            row["experiment"] = "pets_credit_defense_tuning"
            row["training_defense"] = "l2"
            row["weight_decay"] = float(weight_decay)
            row["tuning_family"] = "l2"
            row["tuning_value"] = float(weight_decay)
            row["target_id"] = f"{source_id}__tune_v2_l2_wd{value_token(weight_decay)}"
            rows.append(row)
        for epsilon in dp_epsilons:
            row = source.copy()
            row["source_target_id"] = source_id
            row["experiment"] = "pets_credit_defense_tuning"
            row["training_defense"] = "dp_qml"
            row["weight_decay"] = 0.0
            row["dp_target_epsilon"] = float(epsilon)
            row["dp_noise_multiplier"] = float("nan")
            row["dp_optimizer"] = "rmsprop"
            row["dp_batch_size"] = int(dp_batch_size)
            row["dp_epochs"] = int(dp_epochs)
            row["dp_learning_rate"] = float(dp_learning_rate)
            row["dp_max_grad_norm"] = float(dp_max_grad_norm)
            row["dp_delta"] = float(dp_delta)
            row["dp_scheduler"] = "none"
            row["dp_loss"] = "unweighted_nll"
            row["dp_protocol"] = "watkins_faithful_core_v2"
            row["tuning_family"] = "dp_qml"
            row["tuning_value"] = float(epsilon)
            row["target_id"] = f"{source_id}__tune_v2_watkins_dp_eps{value_token(epsilon)}"
            rows.append(row)
    result = pd.DataFrame(rows)
    if result.target_id.duplicated().any():
        raise AssertionError("duplicate tuning target IDs")
    expected = len(selected) * (len(l2_weight_decays) + len(dp_epsilons))
    if len(result) != expected:
        raise AssertionError("tuning grid expansion is incomplete")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-targets",
        type=Path,
        default=Path("pets_targets/credit_defense_targets.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("pets_targets/credit_defense_tuning_targets.csv"),
    )
    parser.add_argument("--block-id", default="pets_b01")
    parser.add_argument("--l2-weight-decays", default="0.001,0.0001")
    parser.add_argument("--dp-epsilons", default="8,16")
    parser.add_argument("--dp-batch-size", type=int, default=32)
    parser.add_argument("--dp-epochs", type=int, default=30)
    parser.add_argument("--dp-learning-rate", type=float, default=0.05)
    parser.add_argument("--dp-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    args = parser.parse_args()
    l2_values = parse_positive_grid(args.l2_weight_decays, "L2 grid")
    dp_values = parse_positive_grid(args.dp_epsilons, "DP epsilon grid")
    result = build_tuning_targets(
        pd.read_csv(args.base_targets),
        block_id=args.block_id,
        l2_weight_decays=l2_values,
        dp_epsilons=dp_values,
        dp_batch_size=args.dp_batch_size,
        dp_epochs=args.dp_epochs,
        dp_learning_rate=args.dp_learning_rate,
        dp_max_grad_norm=args.dp_max_grad_norm,
        dp_delta=args.dp_delta,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "protocol": "pets_utility_only_defense_tuning_v2",
                "source": str(args.base_targets.resolve()),
                "development_block": args.block_id,
                "l2_weight_decays": l2_values,
                "dp_epsilons": dp_values,
                "dp_optimizer": "rmsprop",
                "dp_batch_size": args.dp_batch_size,
                "dp_epochs": args.dp_epochs,
                "dp_learning_rate": args.dp_learning_rate,
                "dp_max_grad_norm": args.dp_max_grad_norm,
                "dp_delta": args.dp_delta,
                "dp_scheduler": "none",
                "dp_loss": "unweighted_nll",
                "decision_threshold": "validation_balanced_accuracy",
                "selection_must_not_use_attack_results": True,
                "targets": len(result),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[OK] tuning targets={len(result)} -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
