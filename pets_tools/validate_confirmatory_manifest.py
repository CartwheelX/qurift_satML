#!/usr/bin/env python3
"""Fail-closed validation for the prospective PETS v2 target manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROLES = {"low", "repetition", "stress"}
ROLE_CELLS = {
    "low": "eff_su2_r1_d6",
    "repetition": "eff_su2_r5_d6",
    "stress": "zz_r5_d6",
}
TRAINING_DEFENSES = {"none", "l2", "hamp_train", "dp_qml"}


def number(value: Any) -> float:
    return float(value)


def validate(targets: pd.DataFrame, prior: list[pd.DataFrame]) -> dict[str, Any]:
    required = {
        "target_id",
        "experiment",
        "block_id",
        "defense_structural_role",
        "structural_cell_id",
        "training_defense",
        "data_seed",
        "model_seed",
        "confirmatory_protocol",
        "dp_batch_size",
        "dp_epochs",
        "dp_learning_rate",
        "dp_protocol",
        "dp_delta",
        "dp_max_grad_norm",
        "dp_optimizer",
        "dp_scheduler",
    }
    missing = required - set(targets)
    if missing:
        raise ValueError(f"manifest lacks required columns: {sorted(missing)}")
    if targets.empty or targets.target_id.duplicated().any():
        raise ValueError("manifest is empty or contains duplicate target IDs")
    if set(targets.defense_structural_role.astype(str)) != ROLES:
        raise ValueError("manifest must contain exactly low, repetition, and stress roles")
    if set(targets.training_defense.astype(str)) != TRAINING_DEFENSES:
        raise ValueError("manifest must contain the four frozen training arms")
    if set(targets.confirmatory_protocol.astype(str)) != {"pets_credit_three_regime_v2"}:
        raise ValueError("manifest mixes confirmatory protocol versions")
    if set(targets.experiment.astype(str)) != {"petsv2_credit_confirmatory_training"}:
        raise ValueError("manifest is not isolated in the PETS v2 experiment namespace")

    blocks = sorted(targets.block_id.astype(str).unique())
    if len(blocks) != 8:
        raise ValueError(
            f"confirmatory v2 requires exactly 8 paired blocks, found {len(blocks)}"
        )
    expected_per_block = len(ROLES) * len(TRAINING_DEFENSES)
    for block_id, block in targets.groupby("block_id"):
        if len(block) != expected_per_block:
            raise ValueError(f"{block_id} has {len(block)} rows, expected {expected_per_block}")
        if block.data_seed.nunique() != 1 or block.model_seed.nunique() != 1:
            raise ValueError(f"{block_id} is not seed-paired across all conditions")
        combinations = set(
            zip(
                block.defense_structural_role.astype(str),
                block.training_defense.astype(str),
            )
        )
        expected = {(role, defense) for role in ROLES for defense in TRAINING_DEFENSES}
        if combinations != expected:
            raise ValueError(f"{block_id} lacks a role-by-defense condition")

    for role, role_frame in targets.groupby("defense_structural_role"):
        if role_frame.structural_cell_id.nunique() != 1:
            raise ValueError(f"role {role!r} maps to multiple structural cells")
        observed_cell = str(role_frame.structural_cell_id.iloc[0])
        if observed_cell != ROLE_CELLS[str(role)]:
            raise ValueError(
                f"role {role!r} maps to {observed_cell!r}, expected "
                f"{ROLE_CELLS[str(role)]!r}"
            )

    dp = targets[targets.training_defense.astype(str).eq("dp_qml")]
    if not dp.dp_batch_size.astype(int).eq(32).all():
        raise ValueError("primary DP-QML arm must retain Watkins batch size 32")
    if not dp.dp_epochs.astype(int).eq(30).all():
        raise ValueError("primary DP-QML arm must retain Watkins 30 epochs")
    if not dp.dp_learning_rate.astype(float).eq(0.05).all():
        raise ValueError("primary DP-QML arm must retain Watkins learning rate 0.05")
    if not dp.dp_protocol.astype(str).eq("watkins_faithful_core_v2").all():
        raise ValueError("primary DP-QML arm is not the literature-schedule protocol")
    if not dp.dp_delta.astype(float).eq(1e-5).all():
        raise ValueError("primary DP-QML arm must retain delta=1e-5")
    if not dp.dp_max_grad_norm.astype(float).eq(1.0).all():
        raise ValueError("primary DP-QML arm must retain clipping norm 1")
    if not dp.dp_optimizer.astype(str).str.lower().eq("rmsprop").all():
        raise ValueError("primary DP-QML arm must retain RMSprop")
    if not dp.dp_scheduler.astype(str).str.lower().eq("none").all():
        raise ValueError("primary DP-QML arm must not use a scheduler")

    prior_data: set[int] = set()
    prior_model: set[int] = set()
    for frame in prior:
        if "data_seed" in frame:
            prior_data.update(
                pd.to_numeric(frame.data_seed, errors="coerce").dropna().astype(int)
            )
        if "model_seed" in frame:
            prior_model.update(
                pd.to_numeric(frame.model_seed, errors="coerce").dropna().astype(int)
            )
    current_data = set(targets.data_seed.astype(int))
    current_model = set(targets.model_seed.astype(int))
    if current_data & prior_data or current_model & prior_model:
        raise ValueError(
            "prospective seed overlap: "
            f"data={sorted(current_data & prior_data)}, "
            f"model={sorted(current_model & prior_model)}"
        )

    return {
        "protocol": "pets_credit_three_regime_v2_manifest_validation",
        "valid": True,
        "targets": int(len(targets)),
        "blocks": len(blocks),
        "roles": sorted(ROLES),
        "role_cells": ROLE_CELLS,
        "training_defenses": sorted(TRAINING_DEFENSES),
        "conditions_per_block": expected_per_block,
        "fresh_data_seeds": sorted(current_data),
        "fresh_model_seeds": sorted(current_model),
        "dp_primary_schedule": "Watkins-style RMSprop/batch32/30 epochs",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("pets_v2_targets/credit_confirmatory_training_targets.csv"),
    )
    parser.add_argument("--prior-targets", action="append", type=Path, default=[])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("pets_v2_targets/confirmatory_manifest_validation.json"),
    )
    args = parser.parse_args()
    targets = pd.read_csv(args.targets)
    prior = [pd.read_csv(path) for path in args.prior_targets if path.exists()]
    try:
        payload = validate(targets, prior)
    except Exception as error:
        payload = {
            "protocol": "pets_credit_three_regime_v2_manifest_validation",
            "valid": False,
            "error": f"{type(error).__name__}: {error}",
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[FAIL] {payload['error']}", file=sys.stderr)
        raise SystemExit(1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[OK] validated {payload['targets']} targets across {payload['blocks']} blocks")


if __name__ == "__main__":
    main()
