#!/usr/bin/env python3
"""Freeze fresh-seed high/low structural target manifests for PETS defenses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def build_targets(
    source: pd.DataFrame,
    *,
    low_cell: str,
    high_cell: str,
    blocks: int,
    data_seed_start: int,
    model_seed_start: int,
) -> pd.DataFrame:
    if low_cell == high_cell:
        raise ValueError("low and high structural cells must differ")
    rows = []
    for role, cell in (("low", low_cell), ("high", high_cell)):
        matches = source[source["structural_cell_id"].astype(str) == cell]
        if matches.empty:
            raise ValueError(f"structural cell {cell!r} is absent from source target table")
        template = matches.iloc[0].copy()
        for block in range(1, int(blocks) + 1):
            row = template.copy()
            data_seed = int(data_seed_start) + block - 1
            model_seed = int(model_seed_start) + block - 1
            row["experiment"] = "pets_credit_defense_targets"
            row["block_id"] = f"pets_b{block:02d}"
            row["data_seed"] = data_seed
            row["split_seed"] = data_seed
            row["model_seed"] = model_seed
            row["init_seed"] = model_seed
            row["seed"] = model_seed
            row["role"] = role
            row["defense_structural_role"] = role
            row["target_id"] = f"PETS_CREDIT_{role}_{safe(cell)}_b{block:02d}"
            rows.append(row)
    result = pd.DataFrame(rows)
    if result["target_id"].duplicated().any():
        raise AssertionError("generated duplicate PETS target IDs")
    source_data_seeds = set(pd.to_numeric(source["data_seed"], errors="coerce").dropna().astype(int))
    source_model_seeds = set(pd.to_numeric(source["model_seed"], errors="coerce").dropna().astype(int))
    if source_data_seeds & set(result["data_seed"].astype(int)):
        raise ValueError("fresh data seeds overlap the discovery target table")
    if source_model_seeds & set(result["model_seed"].astype(int)):
        raise ValueError("fresh model seeds overlap the discovery target table")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-targets",
        type=Path,
        default=Path("satml_targets/credit_factorial_targets.csv"),
    )
    parser.add_argument("--low-cell", default="eff_su2_r1_d6")
    parser.add_argument("--high-cell", default="eff_su2_r5_d6")
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--data-seed-start", type=int, default=70261)
    parser.add_argument("--model-seed-start", type=int, default=80261)
    parser.add_argument(
        "--out", type=Path, default=Path("pets_targets/credit_defense_targets.csv")
    )
    args = parser.parse_args()
    if args.blocks < 1:
        parser.error("--blocks must be positive")
    source = pd.read_csv(args.source_targets)
    required = {"structural_cell_id", "data_seed", "model_seed"}
    missing = required - set(source.columns)
    if missing:
        parser.error(f"source target table lacks columns: {sorted(missing)}")
    targets = build_targets(
        source,
        low_cell=args.low_cell,
        high_cell=args.high_cell,
        blocks=args.blocks,
        data_seed_start=args.data_seed_start,
        model_seed_start=args.model_seed_start,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(args.out, index=False)
    metadata = {
        "protocol": "fresh_seed_paired_high_low_v1",
        "source_targets": str(args.source_targets.resolve()),
        "source_role": "discovery_only",
        "low_cell": args.low_cell,
        "high_cell": args.high_cell,
        "blocks": args.blocks,
        "fresh_data_seeds": sorted(targets.data_seed.astype(int).unique().tolist()),
        "fresh_model_seeds": sorted(targets.model_seed.astype(int).unique().tolist()),
        "pairing": "same fresh data and model seed within each block",
    }
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[OK] targets={len(targets)} -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
