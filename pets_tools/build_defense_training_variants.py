#!/usr/bin/env python3
"""Expand fresh structural targets into predeclared training-defense variants."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def expand_variants(
    targets: pd.DataFrame,
    *,
    l2_weight_decay: float,
    hamp_gamma: float,
    hamp_alpha: float,
    dp_target_epsilon: float,
    dp_max_grad_norm: float,
    dp_delta: float,
    dp_batch_size: int = 32,
    dp_epochs: int = 30,
    dp_learning_rate: float = 0.05,
) -> pd.DataFrame:
    if l2_weight_decay <= 0:
        raise ValueError("strong-L2 weight decay must be positive")
    if dp_batch_size <= 0 or dp_epochs <= 0 or dp_learning_rate <= 0:
        raise ValueError("Watkins DP batch size, epochs, and learning rate must be positive")
    variants = []
    for _, source in targets.iterrows():
        for training_defense in ("none", "l2", "hamp_train", "dp_qml"):
            row = source.copy()
            source_id = str(source["target_id"])
            row["source_target_id"] = source_id
            row["training_defense"] = training_defense
            row["experiment"] = "pets_credit_defense_training"
            row["target_id"] = f"{source_id}__{training_defense}"
            row["weight_decay"] = l2_weight_decay if training_defense == "l2" else 0.0
            row["hamp_gamma"] = hamp_gamma
            row["hamp_alpha"] = hamp_alpha
            row["dp_target_epsilon"] = dp_target_epsilon
            row["dp_noise_multiplier"] = float("nan")
            row["dp_max_grad_norm"] = dp_max_grad_norm
            row["dp_delta"] = dp_delta
            row["dp_optimizer"] = "rmsprop"
            row["dp_batch_size"] = int(dp_batch_size)
            row["dp_epochs"] = int(dp_epochs)
            row["dp_learning_rate"] = float(dp_learning_rate)
            row["dp_scheduler"] = "none"
            row["dp_loss"] = "unweighted_nll"
            row["dp_protocol"] = "watkins_faithful_core_v2"
            variants.append(row)
    result = pd.DataFrame(variants)
    if result.target_id.duplicated().any():
        raise AssertionError("duplicate defense-training target IDs")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets", type=Path, default=Path("pets_targets/credit_defense_targets.csv")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("pets_targets/credit_defense_training_targets.csv")
    )
    parser.add_argument("--l2-weight-decay", type=float, default=0.01)
    parser.add_argument("--hamp-gamma", type=float, default=0.95)
    parser.add_argument("--hamp-alpha", type=float, default=0.001)
    parser.add_argument("--dp-target-epsilon", type=float, default=4.0)
    parser.add_argument("--dp-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--dp-batch-size", type=int, default=32)
    parser.add_argument("--dp-epochs", type=int, default=30)
    parser.add_argument("--dp-learning-rate", type=float, default=0.05)
    args = parser.parse_args()
    source = pd.read_csv(args.targets)
    expanded = expand_variants(
        source,
        l2_weight_decay=args.l2_weight_decay,
        hamp_gamma=args.hamp_gamma,
        hamp_alpha=args.hamp_alpha,
        dp_target_epsilon=args.dp_target_epsilon,
        dp_max_grad_norm=args.dp_max_grad_norm,
        dp_delta=args.dp_delta,
        dp_batch_size=args.dp_batch_size,
        dp_epochs=args.dp_epochs,
        dp_learning_rate=args.dp_learning_rate,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    expanded.to_csv(args.out, index=False)
    metadata = {
        "protocol": "pets_training_defense_variants_v2",
        "source_targets": str(args.targets.resolve()),
        "variants": ["none", "l2", "hamp_train", "dp_qml"],
        "hamp_full_definition": "hamp_train checkpoint plus HAMP output transformation",
        "l2_weight_decay": args.l2_weight_decay,
        "hamp_gamma": args.hamp_gamma,
        "hamp_alpha": args.hamp_alpha,
        "dp_target_epsilon": args.dp_target_epsilon,
        "dp_noise_multiplier": "derived by Opacus from target epsilon at training time",
        "dp_max_grad_norm": args.dp_max_grad_norm,
        "dp_delta": args.dp_delta,
        "dp_optimizer": "rmsprop",
        "dp_batch_size": args.dp_batch_size,
        "dp_epochs": args.dp_epochs,
        "dp_learning_rate": args.dp_learning_rate,
        "dp_scheduler": "none",
        "dp_loss": "unweighted_nll",
        "dp_protocol": "watkins_faithful_core_v2",
        "decision_threshold": "selected on validation balanced accuracy per checkpoint",
    }
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[OK] training targets={len(expanded)} -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
