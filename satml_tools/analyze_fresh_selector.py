#!/usr/bin/env python3
"""Paired fresh-block evaluation of the three frozen selector policies."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from satml_tools.analyze_paired_factorial import bootstrap_mean_interval


POLICY_CONTRASTS = (
    ("privacy_aware", "utility_only"),
    ("privacy_aware", "utility_regularized"),
    ("utility_regularized", "utility_only"),
)


def analyze_selector(
    targets: pd.DataFrame,
    metrics: pd.DataFrame,
    attacks: list[pd.DataFrame],
    *,
    bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    design = targets[["target_id", "block_id", "selector_policy"]]
    long_frames = []
    for outcome in ("valid_acc", "test_acc", "gap"):
        if outcome in metrics:
            frame = design.merge(metrics[["target_id", outcome]], on="target_id", how="inner")
            frame = frame.rename(columns={outcome: "value"})
            frame["outcome"] = outcome
            frame["attack"] = "target_model"
            long_frames.append(frame)
    for attack_table in attacks:
        attack_table = attack_table.copy()
        if "auc" not in attack_table and "attack_auc" in attack_table:
            attack_table = attack_table.rename(columns={"attack_auc": "auc"})
        if "attack" not in attack_table and "auc" in attack_table:
            attack_table["attack"] = "learned_prediction_vector_stats"
        for attack, group in attack_table.groupby("attack"):
            frame = design.merge(group[["target_id", "auc"]], on="target_id", how="inner")
            frame = frame.rename(columns={"auc": "value"})
            frame["outcome"] = "auc"
            frame["attack"] = str(attack)
            long_frames.append(frame)
    if not long_frames:
        raise ValueError("No selector outcomes were available")
    long = pd.concat(long_frames, ignore_index=True)
    summary = (
        long.groupby(["outcome", "attack", "selector_policy"])["value"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"count": "n_blocks", "mean": "mean_value", "std": "sd_across_blocks"})
    )
    rows = []
    for group_index, ((outcome, attack), group) in enumerate(long.groupby(["outcome", "attack"])):
        pivot = group.pivot(index="block_id", columns="selector_policy", values="value")
        for high, low in POLICY_CONTRASTS:
            if high not in pivot or low not in pivot:
                continue
            differences = (pivot[high] - pivot[low]).dropna().to_numpy(float)
            ci_low, ci_high = bootstrap_mean_interval(differences, bootstrap, seed + group_index * 17 + len(rows))
            rows.append(
                {
                    "outcome": outcome,
                    "attack": attack,
                    "contrast": f"{high} - {low}",
                    "mean_difference": float(differences.mean()),
                    "sd_across_blocks": float(differences.std(ddof=1)) if len(differences) > 1 else np.nan,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "n_independent_fresh_blocks": len(differences),
                    "inference_unit": "fresh paired split/init block",
                }
            )
    return summary, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--attacks", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("satml_results/selector_fresh"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    summary, contrasts = analyze_selector(
        pd.read_csv(args.targets), pd.read_csv(args.metrics), [pd.read_csv(path) for path in args.attacks],
        bootstrap=args.bootstrap, seed=args.bootstrap_seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "selector_policy_summary.csv", index=False)
    contrasts.to_csv(args.out_dir / "selector_paired_contrasts.csv", index=False)
    print(f"[OK] summaries={len(summary)} contrasts={len(contrasts)}")


if __name__ == "__main__":
    main()
