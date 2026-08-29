#!/usr/bin/env python3
"""Aggregate PETS privacy/utility results with paired structural contrasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from qurift.defenses.utility import classification_utility_from_arrays
from qurift.defenses.protocol import PARTITION_PROTOCOL


TASK_UTILITY_OUTCOMES = (
    "accuracy",
    "balanced_accuracy",
    "task_roc_auc",
    "task_average_precision",
    "minority_class_recall",
    "minimum_class_recall",
    "predicted_minority_fraction",
    "prediction_collapse",
)
EVALUATION_PROTOCOL = "pets_defense_evaluation_v3"


def effective_defense(training_defense: str, output_defense: str) -> str:
    training = str(training_defense).strip().lower()
    output = str(output_defense).strip().lower()
    if training in {"", "none", "nan"}:
        return output
    if training == "hamp_train" and output == "hamp_output":
        return "hamp_full"
    if output == "none":
        return training
    return f"{training}+{output}"


def bootstrap_mean(values: Sequence[float], *, draws: int, seed: int):
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    samples = rng.choice(array, size=(int(draws), len(array)), replace=True).mean(axis=1)
    return (
        float(array.mean()),
        float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def paired_structural_effects(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    outcome: str,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(list(group_columns), dropna=False):
        pivot = group.pivot_table(
            index="block_id", columns="structural_role", values=outcome, aggfunc="first"
        )
        if not {"low", "high"}.issubset(pivot.columns):
            continue
        differences = (pivot["high"] - pivot["low"]).dropna()
        mean, std, low, high = bootstrap_mean(
            differences.to_numpy(), draws=draws, seed=seed + len(rows)
        )
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "outcome": outcome,
                "contrast": "high-low",
                "mean_difference": mean,
                "sd_across_paired_blocks": std,
                "ci95_low": low,
                "ci95_high": high,
                "paired_blocks": len(differences),
            }
        )
    return pd.DataFrame(rows)


def difference_in_differences(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    outcome: str,
    baseline: str,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    extra = [column for column in group_columns if column != "effective_defense"]
    grouped = [((), frame)] if not extra else frame.groupby(extra, dropna=False)
    for keys, group in grouped:
        pivot = group.pivot_table(
            index="block_id",
            columns=["effective_defense", "structural_role"],
            values=outcome,
            aggfunc="first",
        )
        if (baseline, "low") not in pivot or (baseline, "high") not in pivot:
            continue
        baseline_effect = pivot[(baseline, "high")] - pivot[(baseline, "low")]
        defenses = sorted(set(pivot.columns.get_level_values(0)))
        for defense in defenses:
            if defense == baseline or (defense, "low") not in pivot or (defense, "high") not in pivot:
                continue
            defense_effect = pivot[(defense, "high")] - pivot[(defense, "low")]
            values = (defense_effect - baseline_effect).dropna()
            mean, std, low, high = bootstrap_mean(
                values.to_numpy(), draws=draws, seed=seed + len(rows)
            )
            keys = keys if isinstance(keys, tuple) else (keys,)
            rows.append(
                {
                    **dict(zip(extra, keys)),
                    "effective_defense": defense,
                    "baseline_defense": baseline,
                    "outcome": outcome,
                    "contrast": "(high-low)_defense-(high-low)_none",
                    "mean_difference_in_differences": mean,
                    "sd_across_paired_blocks": std,
                    "ci95_low": low,
                    "ci95_high": high,
                    "paired_blocks": len(values),
                }
            )
    return pd.DataFrame(rows)


def load_results(root: Path, *, exclude_blocks: Iterable[str] = ()):
    excluded = {str(value) for value in exclude_blocks}
    metric_files = sorted(root.glob("*/adaptive_attack_metrics.csv"))
    metadata_files = sorted(root.glob("*/evaluation_metadata.json"))
    if not metric_files or not metadata_files:
        raise FileNotFoundError(f"no complete defense results found below {root}")
    privacy_frames = [pd.read_csv(path) for path in metric_files]
    hsj_rows = []
    for path in sorted(root.glob("*/hsj/*_metrics.json")):
        payload = json.loads(path.read_text())
        if str(payload.get("block_id")) in excluded:
            continue
        if payload.get("protocol") != "pets_defended_label_only_hsj_v3":
            raise ValueError(f"stale HSJ result must be archived: {path}")
        if payload.get("partition_protocol") != PARTITION_PROTOCOL:
            raise ValueError(f"HSJ result lacks label-matched partitions: {path}")
        hsj_rows.append(
            {
                key: payload.get(key)
                for key in (
                    "target_id",
                    "block_id",
                    "structural_cell_id",
                    "structural_role",
                    "training_defense",
                    "defense",
                    "attack",
                    "attack_fit",
                )
            }
            | payload["metrics"]
        )
    if hsj_rows:
        privacy_frames.append(pd.DataFrame(hsj_rows))
    lira_rows = []
    for path in sorted(root.glob("*/lira/*_metrics.json")):
        payload = json.loads(path.read_text())
        if str(payload.get("block_id")) in excluded:
            continue
        if payload.get("protocol") != "pets_adaptive_defended_lira_v3":
            raise ValueError(f"stale LiRA result must be archived: {path}")
        if payload.get("partition_protocol") != PARTITION_PROTOCOL:
            raise ValueError(f"LiRA result lacks label-matched partitions: {path}")
        lira_rows.extend(payload.get("rows", []))
    if lira_rows:
        privacy_frames.append(pd.DataFrame(lira_rows))
    query_rows = []
    for path in sorted(root.glob("*/query_stress/metrics.json")):
        payload = json.loads(path.read_text())
        if str(payload.get("block_id")) in excluded:
            continue
        if payload.get("protocol") != "pets_nearby_query_stress_v3":
            raise ValueError(f"stale nearby-query result must be archived: {path}")
        if payload.get("partition_protocol") != PARTITION_PROTOCOL:
            raise ValueError(f"nearby-query result lacks label-matched partitions: {path}")
        query_rows.extend(payload.get("rows", []))
    if query_rows:
        privacy_frames.append(pd.DataFrame(query_rows))
    privacy = pd.concat(privacy_frames, ignore_index=True)
    if "training_defense" not in privacy:
        privacy["training_defense"] = "none"
    privacy["effective_defense"] = [
        effective_defense(training, output)
        for training, output in zip(privacy.training_defense, privacy.defense)
    ]
    utility_rows = []
    for path in metadata_files:
        payload = json.loads(path.read_text())
        target = payload.get("target", {})
        if str(target.get("block_id")) in excluded:
            continue
        if payload.get("protocol") != EVALUATION_PROTOCOL:
            raise ValueError(
                f"{path} uses incompatible protocol {payload.get('protocol')!r}; "
                "archive and rerun the label-matched evaluation"
            )
        if payload.get("partition_protocol") != PARTITION_PROTOCOL:
            raise ValueError(f"{path} does not use label-matched MIA partitions")
        if payload.get("utility_evaluation", {}).get("scope") != "full_held_out_test_split":
            raise ValueError(f"{path} does not report utility on the held-out test split")
        target = payload["target"]
        training = str(target.get("training_defense", "none"))
        prediction_path = path.with_name("test_utility_predictions.csv")
        predictions = (
            pd.read_csv(prediction_path) if prediction_path.exists() else pd.DataFrame()
        )
        if predictions.empty:
            raise FileNotFoundError(f"missing full-test utility predictions: {prediction_path}")
        for output, values in payload["conditions"].items():
            task_utility = dict(values["utility"])
            if not predictions.empty:
                condition = predictions[predictions.defense.astype(str).eq(str(output))]
                probability_columns = sorted(
                    [column for column in condition if column.startswith("probability_")],
                    key=lambda value: int(value.rsplit("_", 1)[1]),
                )
                if not condition.empty and probability_columns:
                    task_utility.update(
                        classification_utility_from_arrays(
                            condition[probability_columns].to_numpy(),
                            condition["true_label"].to_numpy(),
                            predictions=condition["predicted_label"].to_numpy(),
                        )
                    )
            utility_rows.append(
                {
                    "target_id": target["target_id"],
                    "block_id": target.get("block_id"),
                    "structural_cell_id": target.get("structural_cell_id"),
                    "structural_role": target.get(
                        "defense_structural_role", target.get("role")
                    ),
                    "training_defense": training,
                    "defense": output,
                    "effective_defense": effective_defense(training, output),
                    **task_utility,
                    "runtime_seconds": values["runtime_seconds"],
                    "attack_runtime_seconds": values.get("attack_runtime_seconds"),
                    "utility_runtime_seconds": values.get("utility_runtime_seconds"),
                    "records_per_second": values["records_per_second"],
                    "mean_l1_probability_distortion": values[
                        "mean_l1_probability_distortion"
                    ],
                }
            )
    return privacy, pd.DataFrame(utility_rows)


def group_summary(frame: pd.DataFrame, groups: Sequence[str], outcomes: Sequence[str]):
    rows = []
    for keys, group in frame.groupby(list(groups), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        fixed = dict(zip(groups, keys))
        for outcome in outcomes:
            values = group[outcome].dropna().astype(float)
            rows.append(
                {
                    **fixed,
                    "outcome": outcome,
                    "count": len(values),
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                }
            )
    return pd.DataFrame(rows)


def make_plots(privacy_summary: pd.DataFrame, structural: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = privacy_summary[
        privacy_summary.attack.eq("loss") & privacy_summary.outcome.eq("auc")
    ].copy()
    if not selected.empty:
        pivot = selected.pivot(
            index="effective_defense", columns="structural_role", values="mean"
        )
        pivot = pivot[[column for column in ("low", "high") if column in pivot]]
        if pivot.empty or len(pivot.columns) == 0:
            pivot = None
        if pivot is not None:
            axis = pivot.plot(kind="bar", figsize=(10, 4.8), ylim=(0.45, 1.0))
            axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
            axis.set_ylabel("Adaptive loss-MIA AUC")
            axis.set_xlabel("Defense")
            axis.set_title("Privacy leakage by structural role and defense")
            axis.legend(title="Structural role")
            plt.tight_layout()
            for suffix in ("png", "pdf"):
                plt.savefig(out_dir / f"defense_structure_auc.{suffix}", dpi=200)
            plt.close()

    chosen = structural[
        structural.attack.eq("loss") if "attack" in structural else np.ones(len(structural), bool)
    ].copy()
    if not chosen.empty:
        chosen = chosen.sort_values("mean_difference")
        figure, axis = plt.subplots(figsize=(9, max(4, 0.35 * len(chosen))))
        errors = np.vstack(
            [
                chosen.mean_difference - chosen.ci95_low,
                chosen.ci95_high - chosen.mean_difference,
            ]
        )
        axis.errorbar(
            chosen.mean_difference,
            np.arange(len(chosen)),
            xerr=errors,
            fmt="o",
            capsize=3,
        )
        axis.set_yticks(np.arange(len(chosen)), chosen.effective_defense)
        axis.axvline(0, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("Paired high-low loss-MIA AUC")
        axis.set_title("Residual structural leakage under defense")
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            figure.savefig(out_dir / f"defense_high_low_auc.{suffix}", dpi=200)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("pets_results/defenses"))
    parser.add_argument("--out-dir", type=Path, default=Path("pets_results/analysis"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--exclude-block",
        action="append",
        default=[],
        help="Exclude development/pilot blocks from confirmatory summaries.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    privacy, utility = load_results(
        args.results_dir, exclude_blocks=args.exclude_block
    )
    if args.exclude_block:
        privacy = privacy[~privacy.block_id.astype(str).isin(args.exclude_block)].copy()
        utility = utility[~utility.block_id.astype(str).isin(args.exclude_block)].copy()
        if privacy.empty or utility.empty:
            raise ValueError("block exclusions removed all analyzable results")
    adaptive = privacy[
        privacy.attack_fit.astype(str).str.startswith("adaptive_defended")
    ].copy()
    appendix = privacy[
        privacy.attack_fit.eq("dynanoise_artifact_fixed_threshold_appendix")
    ].copy()
    privacy_summary = group_summary(
        adaptive,
        ["effective_defense", "structural_role", "attack"],
        ["auc", "balanced_accuracy", "tpr_at_5_fpr", "tpr_at_10_fpr"],
    )
    utility_outcomes = [
        outcome
        for outcome in (
            *TASK_UTILITY_OUTCOMES,
            "nll",
            "mean_entropy",
            "runtime_seconds",
            "attack_runtime_seconds",
            "utility_runtime_seconds",
            "records_per_second",
            "mean_l1_probability_distortion",
        )
        if outcome in utility.columns
    ]
    utility_summary = group_summary(
        utility,
        ["effective_defense", "structural_role"],
        utility_outcomes,
    )
    privacy_structural = paired_structural_effects(
        adaptive,
        group_columns=["effective_defense", "attack"],
        outcome="auc",
        draws=args.bootstrap,
        seed=args.seed,
    )
    privacy_did = difference_in_differences(
        adaptive,
        group_columns=["effective_defense", "attack"],
        outcome="auc",
        baseline="none",
        draws=args.bootstrap,
        seed=args.seed + 1000,
    )
    structural_utility_outcomes = [
        outcome for outcome in TASK_UTILITY_OUTCOMES if outcome in utility.columns
    ]
    utility_structural = pd.concat(
        [
            paired_structural_effects(
                utility,
                group_columns=["effective_defense"],
                outcome=outcome,
                draws=args.bootstrap,
                seed=args.seed + 2000 + index * 100,
            )
            for index, outcome in enumerate(structural_utility_outcomes)
        ],
        ignore_index=True,
    )
    utility_did = pd.concat(
        [
            difference_in_differences(
                utility,
                group_columns=["effective_defense"],
                outcome=outcome,
                baseline="none",
                draws=args.bootstrap,
                seed=args.seed + 3000 + index * 100,
            )
            for index, outcome in enumerate(structural_utility_outcomes)
        ],
        ignore_index=True,
    )
    outputs = {
        "adaptive_attack_raw.csv": adaptive,
        "artifact_faithful_dynanoise_appendix.csv": appendix,
        "privacy_summary.csv": privacy_summary,
        "utility_summary.csv": utility_summary,
        "privacy_high_low.csv": privacy_structural,
        "privacy_difference_in_differences.csv": privacy_did,
        "utility_high_low.csv": utility_structural,
        "utility_difference_in_differences.csv": utility_did,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.out_dir / name, index=False)
    metadata = {
        "protocol": "pets_defense_analysis_v2",
        "bootstrap_draws": args.bootstrap,
        "bootstrap_seed": args.seed,
        "independent_unit": "fresh paired target-model block",
        "primary_contrast": "high-low within defense",
        "interaction": "(high-low)_defense-(high-low)_none",
        "adaptive_attacks": True,
        "artifact_faithful_dynanoise_is_appendix_only": True,
        "imbalanced_task_utility": list(TASK_UTILITY_OUTCOMES),
        "targets": int(adaptive.target_id.nunique()),
        "blocks": int(adaptive.block_id.nunique()),
        "excluded_development_blocks": list(args.exclude_block),
    }
    (args.out_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    make_plots(privacy_summary, privacy_structural, args.out_dir)
    print(f"[DONE] analysis -> {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
