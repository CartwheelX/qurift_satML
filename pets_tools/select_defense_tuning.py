#!/usr/bin/env python3
"""Select L2/DP settings using only predeclared task-utility constraints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


UTILITY_COLUMNS = (
    "accuracy",
    "balanced_accuracy",
    "task_roc_auc",
    "task_average_precision",
    "minority_class_recall",
    "predicted_minority_fraction",
    "prediction_collapse",
    "calibrated_accuracy",
    "calibrated_balanced_accuracy",
    "calibrated_minority_class_recall",
    "calibrated_predicted_minority_fraction",
    "calibrated_prediction_collapse",
    "calibrated_decision_threshold",
)


def read_tuning_metrics(targets: pd.DataFrame, run_root: Path) -> pd.DataFrame:
    rows = []
    for _, target in targets.iterrows():
        path = (
            run_root
            / str(target["experiment"])
            / str(target["target_id"])
            / "training_metadata.json"
        )
        if not path.exists():
            raise FileNotFoundError(f"missing tuning metadata: {path}")
        payload = json.loads(path.read_text())
        metrics = payload.get("metrics", {}).get("test", {})
        missing = [column for column in UTILITY_COLUMNS if column not in metrics]
        if missing:
            raise ValueError(f"{path} lacks tuning utility metrics: {missing}")
        rows.append(
            {
                "target_id": target["target_id"],
                "source_target_id": target["source_target_id"],
                "block_id": target["block_id"],
                "structural_role": target["defense_structural_role"],
                "tuning_family": target["tuning_family"],
                "tuning_value": float(target["tuning_value"]),
                **{column: float(metrics[column]) for column in UTILITY_COLUMNS},
                "test_records": int(metrics.get("records", 0)),
                "privacy_epsilon": (
                    None
                    if payload.get("privacy") is None
                    else float(payload["privacy"]["epsilon"])
                ),
                "privacy_delta": (
                    None
                    if payload.get("privacy") is None
                    else float(payload["privacy"]["delta"])
                ),
                "noise_multiplier": (
                    None
                    if payload.get("privacy") is None
                    else float(payload["privacy"]["config"]["noise_multiplier"])
                ),
            }
        )
    return pd.DataFrame(rows)


def eligible_settings(
    metrics: pd.DataFrame,
    *,
    minimum_roc_auc: float,
    minimum_average_precision: float,
    minimum_minority_recall: float,
    minimum_balanced_accuracy: float = 0.55,
) -> pd.DataFrame:
    rows = []
    for (family, value), group in metrics.groupby(
        ["tuning_family", "tuning_value"], dropna=False
    ):
        eligible = bool(
            group.task_roc_auc.ge(minimum_roc_auc).all()
            and group.task_average_precision.ge(minimum_average_precision).all()
            and group.calibrated_minority_class_recall.ge(
                minimum_minority_recall
            ).all()
            and group.calibrated_balanced_accuracy.ge(
                minimum_balanced_accuracy
            ).all()
            and group.calibrated_prediction_collapse.eq(0.0).all()
        )
        rows.append(
            {
                "tuning_family": family,
                "tuning_value": float(value),
                "roles": int(group.structural_role.nunique()),
                "eligible": eligible,
                "minimum_task_roc_auc": float(group.task_roc_auc.min()),
                "minimum_task_average_precision": float(
                    group.task_average_precision.min()
                ),
                "minimum_minority_class_recall": float(
                    group.calibrated_minority_class_recall.min()
                ),
                "minimum_calibrated_balanced_accuracy": float(
                    group.calibrated_balanced_accuracy.min()
                ),
                "collapsed_targets": int(group.calibrated_prediction_collapse.sum()),
                "default_threshold_collapsed_targets": int(
                    group.prediction_collapse.sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def choose_settings(summary: pd.DataFrame) -> dict[str, float]:
    chosen: dict[str, float] = {}
    for family in ("l2", "dp_qml"):
        candidates = summary[
            summary.tuning_family.eq(family) & summary.eligible
        ].copy()
        if candidates.empty:
            raise RuntimeError(
                f"no {family} setting satisfies the predeclared utility constraints; "
                "expand the utility-only grid rather than inspecting attack outcomes"
            )
        # Strongest feasible regularization: largest L2 penalty or smallest epsilon.
        chosen[family] = float(
            candidates.tuning_value.max()
            if family == "l2"
            else candidates.tuning_value.min()
        )
    return chosen


def freeze_confirmatory_manifest(
    source: pd.DataFrame,
    *,
    chosen: dict[str, float],
    development_block: str,
) -> pd.DataFrame:
    result = source[~source.block_id.astype(str).eq(str(development_block))].copy()
    if result.empty:
        raise ValueError("no confirmatory blocks remain after excluding development block")
    l2 = result.training_defense.astype(str).eq("l2")
    dp = result.training_defense.astype(str).eq("dp_qml")
    result.loc[l2, "weight_decay"] = chosen["l2"]
    result.loc[dp, "dp_target_epsilon"] = chosen["dp_qml"]
    result["tuning_protocol"] = "utility_only_calibrated_threshold_v2"
    result["development_block_excluded"] = str(development_block)
    expected_variants = {"none", "l2", "hamp_train", "dp_qml"}
    for (_, _), group in result.groupby(["block_id", "structural_cell_id"]):
        if set(group.training_defense.astype(str)) != expected_variants:
            raise AssertionError("confirmatory manifest lost a training-defense variant")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tuning-targets",
        type=Path,
        default=Path("pets_targets/credit_defense_tuning_targets.csv"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("pets_runs"))
    parser.add_argument(
        "--source-training-targets",
        type=Path,
        default=Path("pets_targets/credit_defense_training_targets.csv"),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("pets_results/tuning")
    )
    parser.add_argument(
        "--frozen-targets",
        type=Path,
        default=Path(
            "pets_targets/credit_defense_training_targets_confirmatory.csv"
        ),
    )
    parser.add_argument("--development-block", default="pets_b01")
    parser.add_argument("--minimum-roc-auc", type=float, default=0.65)
    parser.add_argument("--minimum-average-precision", type=float, default=0.30)
    parser.add_argument("--minimum-minority-recall", type=float, default=0.02)
    parser.add_argument("--minimum-balanced-accuracy", type=float, default=0.55)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_tuning_metrics(pd.read_csv(args.tuning_targets), args.run_root)
    summary = eligible_settings(
        metrics,
        minimum_roc_auc=args.minimum_roc_auc,
        minimum_average_precision=args.minimum_average_precision,
        minimum_minority_recall=args.minimum_minority_recall,
        minimum_balanced_accuracy=args.minimum_balanced_accuracy,
    )
    metrics.to_csv(args.out_dir / "utility_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "eligibility_summary.csv", index=False)
    chosen = choose_settings(summary)
    frozen = freeze_confirmatory_manifest(
        pd.read_csv(args.source_training_targets),
        chosen=chosen,
        development_block=args.development_block,
    )
    args.frozen_targets.parent.mkdir(parents=True, exist_ok=True)
    frozen.to_csv(args.frozen_targets, index=False)
    selection: dict[str, Any] = {
        "protocol": "pets_utility_only_tuning_selection_v2",
        "attack_metrics_consulted": False,
        "development_block": args.development_block,
        "selection_constraints": {
            "minimum_task_roc_auc": args.minimum_roc_auc,
            "minimum_task_average_precision": args.minimum_average_precision,
            "minimum_calibrated_minority_class_recall": args.minimum_minority_recall,
            "minimum_calibrated_balanced_accuracy": args.minimum_balanced_accuracy,
            "calibrated_prediction_collapse_required": 0,
        },
        "decision_threshold_rule": (
            "selected independently per checkpoint on validation balanced accuracy; "
            "test and attack outcomes are not consulted"
        ),
        "selection_rule": {
            "l2": "largest eligible weight decay",
            "dp_qml": "smallest eligible epsilon",
        },
        "selected_l2_weight_decay": chosen["l2"],
        "selected_dp_epsilon": chosen["dp_qml"],
        "confirmatory_targets": len(frozen),
        "confirmatory_blocks": sorted(frozen.block_id.astype(str).unique()),
        "frozen_manifest": str(args.frozen_targets.resolve()),
    }
    (args.out_dir / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    args.frozen_targets.with_suffix(".json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    print(
        f"[OK] selected L2={chosen['l2']:.8g} DP-epsilon={chosen['dp_qml']:.8g} "
        f"confirmatory_targets={len(frozen)}"
    )


if __name__ == "__main__":
    main()
