#!/usr/bin/env python3
"""Targeted paired robustness analysis for the encoder angle scale alpha."""
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


def analyze_scale(
    targets: pd.DataFrame,
    metrics: pd.DataFrame,
    attacks: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    design = targets[["target_id", "block_id", "fm_kind", "reps", "depth", "feature_angle_scale"]].copy()
    design = design[design.depth.astype(int) == 2]
    frames = []
    for outcome in ("valid_acc", "test_acc", "gap"):
        if outcome in metrics:
            merged = design.merge(metrics[["target_id", outcome]], on="target_id", how="inner")
            merged = merged.rename(columns={outcome: "value"})
            merged["outcome"], merged["attack"] = outcome, "target_model"
            frames.append(merged)
    loss = attacks[attacks.attack.astype(str).str.lower().eq("loss")]
    merged = design.merge(loss[["target_id", "auc"]], on="target_id", how="inner")
    merged = merged.rename(columns={"auc": "value"})
    merged["outcome"], merged["attack"] = "auc", "loss"
    frames.append(merged)
    long = pd.concat(frames, ignore_index=True)
    block_rows = []
    summary_rows = []
    for group_index, ((outcome, attack, fm, reps), group) in enumerate(
        long.groupby(["outcome", "attack", "fm_kind", "reps"])
    ):
        pivot = group.pivot(index="block_id", columns="feature_angle_scale", values="value")
        for scale in (0.5, 2.0):
            if scale not in pivot or 1.0 not in pivot:
                continue
            differences = (pivot[scale] - pivot[1.0]).dropna()
            for block_id, value in differences.items():
                block_rows.append(
                    {"outcome": outcome, "attack": attack, "fm_kind": fm, "reps": reps,
                     "contrast": f"alpha {scale} - alpha 1.0", "block_id": block_id, "block_effect": value}
                )
            values = differences.to_numpy(float)
            low, high = bootstrap_mean_interval(values, bootstrap, seed + group_index * 19 + len(summary_rows))
            summary_rows.append(
                {"outcome": outcome, "attack": attack, "fm_kind": fm, "reps": reps,
                 "contrast": f"alpha {scale} - alpha 1.0", "mean_difference": values.mean(),
                 "sd_across_blocks": values.std(ddof=1) if len(values) > 1 else np.nan,
                 "ci95_low": low, "ci95_high": high, "n_independent_blocks": len(values),
                 "inference_unit": "paired split/init block"}
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(block_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factorial-targets", type=Path, required=True)
    parser.add_argument("--scaling-targets", type=Path, required=True)
    parser.add_argument("--factorial-metrics", type=Path, required=True)
    parser.add_argument("--scaling-metrics", type=Path, required=True)
    parser.add_argument("--factorial-attacks", type=Path, required=True)
    parser.add_argument("--scaling-attacks", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("satml_results/encoding_scale"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    targets = pd.concat([pd.read_csv(args.factorial_targets), pd.read_csv(args.scaling_targets)], ignore_index=True)
    metrics = pd.concat([pd.read_csv(args.factorial_metrics), pd.read_csv(args.scaling_metrics)], ignore_index=True)
    attacks = pd.concat([pd.read_csv(args.factorial_attacks), pd.read_csv(args.scaling_attacks)], ignore_index=True)
    summary, blocks = analyze_scale(targets, metrics, attacks, bootstrap=args.bootstrap, seed=args.bootstrap_seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "encoding_scale_contrasts.csv", index=False)
    blocks.to_csv(args.out_dir / "encoding_scale_block_effects.csv", index=False)
    print(f"[OK] contrasts={len(summary)} block_effects={len(blocks)}")


if __name__ == "__main__":
    main()
