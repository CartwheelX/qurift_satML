#!/usr/bin/env python3
"""Build an isolated, prospective three-regime PETS confirmatory manifest."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pets_tools.build_defense_training_variants import expand_variants
from reviewer_tools.qurift_lira_attack import LIRA_SCORE_PROTOCOL


DEFAULT_ROLES = {
    "low": "eff_su2_r1_d6",
    "repetition": "eff_su2_r5_d6",
    "stress": "zz_r5_d6",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def parse_role(values: Sequence[str]) -> dict[str, str]:
    if not values:
        return dict(DEFAULT_ROLES)
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--role must use NAME=STRUCTURAL_CELL")
        name, cell = (part.strip() for part in value.split("=", 1))
        if not name or not cell or name in result:
            raise ValueError(f"invalid or duplicate structural role {value!r}")
        result[name] = cell
    if set(result) != set(DEFAULT_ROLES):
        raise ValueError(
            "confirmatory roles must be exactly low, repetition, and stress"
        )
    return result


def used_seeds(frames: Sequence[pd.DataFrame]) -> tuple[set[int], set[int]]:
    data: set[int] = set()
    model: set[int] = set()
    for frame in frames:
        if "data_seed" in frame:
            data.update(
                pd.to_numeric(frame["data_seed"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )
        if "model_seed" in frame:
            model.update(
                pd.to_numeric(frame["model_seed"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )
    return data, model


def exact_sign_flip_p(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    observed = abs(float(array.mean()))
    statistics = [
        abs(float(np.mean(array * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(array))
    ]
    return float(np.mean(np.asarray(statistics) >= observed - 1e-15))


def discovery_stress_evidence(
    frame: pd.DataFrame,
    *,
    roles: Mapping[str, str],
    attack: str = "lira_online_fixed_variance",
) -> dict[str, object]:
    """Validate the inspected Credit evidence used only to choose v2 roles."""

    required = {"attack", "auc", "data_seed"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"stress-evidence table lacks columns: {sorted(missing)}")
    cells = set(roles.values())
    cell_column = None
    for candidate in ("role", "structural_cell_id"):
        if candidate in frame and cells.issubset(
            set(frame[candidate].dropna().astype(str))
        ):
            cell_column = candidate
            break
    if cell_column is None:
        raise ValueError(
            "stress-evidence table has no role/cell column containing all selected cells"
        )
    selected = frame[
        frame.attack.astype(str).eq(attack)
        & frame[cell_column].astype(str).isin(cells)
    ].copy()
    duplicates = selected.duplicated(["data_seed", cell_column], keep=False)
    if duplicates.any():
        raise ValueError("stress-evidence table has duplicate seed-by-cell endpoints")
    pivot = selected.pivot(
        index="data_seed", columns=cell_column, values="auc"
    ).dropna(subset=list(cells))
    if len(pivot) < 8:
        raise ValueError(
            f"stress selection requires at least 8 complete discovery blocks, got {len(pivot)}"
        )
    low = str(roles["low"])
    repetition = str(roles["repetition"])
    stress = str(roles["stress"])
    stress_low = (pivot[stress] - pivot[low]).to_numpy(dtype=float)
    stress_repetition = (pivot[stress] - pivot[repetition]).to_numpy(dtype=float)
    if float(stress_low.mean()) <= 0 or float(stress_repetition.mean()) <= 0:
        raise ValueError(
            "the selected stress cell is not higher than both comparison cells "
            f"for discovery {attack}"
        )
    return {
        "status": "discovery_selected_then_frozen_before_fresh_confirmatory_runs",
        "attack": attack,
        "independent_blocks": int(len(pivot)),
        "cell_mean_auc": {
            role: float(pivot[str(cell)].mean()) for role, cell in roles.items()
        },
        "stress_minus_low": {
            "mean_auc_difference": float(stress_low.mean()),
            "sd_across_blocks": float(stress_low.std(ddof=1)),
            "exact_two_sided_sign_flip_p": exact_sign_flip_p(stress_low),
            "positive_blocks": int(np.sum(stress_low > 0)),
        },
        "stress_minus_repetition": {
            "mean_auc_difference": float(stress_repetition.mean()),
            "sd_across_blocks": float(stress_repetition.std(ddof=1)),
            "exact_two_sided_sign_flip_p": exact_sign_flip_p(stress_repetition),
            "positive_blocks": int(np.sum(stress_repetition > 0)),
        },
        "interpretation": (
            "selection evidence only; fresh v2 blocks provide confirmation. The stress "
            "role is not a one-factor causal contrast with low or repetition."
        ),
    }


def build_structural_targets(
    source: pd.DataFrame,
    *,
    roles: Mapping[str, str],
    blocks: int,
    data_seed_start: int,
    model_seed_start: int,
    forbidden_data_seeds: set[int],
    forbidden_model_seeds: set[int],
) -> pd.DataFrame:
    if blocks < 2:
        raise ValueError("confirmatory protocol requires at least two paired blocks")
    rows = []
    for role, cell in roles.items():
        matches = source[source["structural_cell_id"].astype(str).eq(str(cell))]
        if matches.empty:
            raise ValueError(f"structural cell {cell!r} is absent from the source table")
        template = matches.iloc[0].copy()
        for offset in range(blocks):
            block = offset + 1
            data_seed = int(data_seed_start) + offset
            model_seed = int(model_seed_start) + offset
            row = template.copy()
            row["experiment"] = "petsv2_credit_confirmatory_base"
            row["block_id"] = f"petsv2_b{block:02d}"
            row["data_seed"] = data_seed
            row["split_seed"] = data_seed
            row["model_seed"] = model_seed
            row["init_seed"] = model_seed
            row["seed"] = model_seed
            row["role"] = role
            row["defense_structural_role"] = role
            row["confirmatory_protocol"] = "pets_credit_three_regime_v2"
            row["target_id"] = (
                f"PETSV2_CREDIT_{safe(role)}_{safe(cell)}_b{block:02d}"
            )
            rows.append(row)
    result = pd.DataFrame(rows)
    data_seeds = set(result.data_seed.astype(int))
    model_seeds = set(result.model_seed.astype(int))
    overlap_data = sorted(data_seeds & forbidden_data_seeds)
    overlap_model = sorted(model_seeds & forbidden_model_seeds)
    if overlap_data or overlap_model:
        raise ValueError(
            "prospective seeds overlap prior/discovery manifests: "
            f"data={overlap_data}, model={overlap_model}"
        )
    if result.target_id.duplicated().any():
        raise AssertionError("generated duplicate confirmatory target IDs")
    for _, block in result.groupby("block_id"):
        if set(block.defense_structural_role) != set(roles):
            raise AssertionError("a confirmatory block is missing a structural role")
        if block.data_seed.nunique() != 1 or block.model_seed.nunique() != 1:
            raise AssertionError("structural roles are not seed-paired within a block")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-targets",
        type=Path,
        default=Path("satml_targets/credit_factorial_targets.csv"),
    )
    parser.add_argument(
        "--exclude-targets",
        action="append",
        type=Path,
        default=[],
        help="Additional prior manifests whose seeds must not be reused.",
    )
    parser.add_argument("--role", action="append", default=[])
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--data-seed-start", type=int, default=90261)
    parser.add_argument("--model-seed-start", type=int, default=100261)
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("pets_results/tuning/selection.json"),
    )
    parser.add_argument(
        "--stress-evidence",
        type=Path,
        default=Path(
            "satml_results/credit_factorial/lira/lira_reference_mia_raw.csv"
        ),
    )
    parser.add_argument(
        "--structural-out",
        type=Path,
        default=Path("pets_v2_targets/credit_confirmatory_structural_targets.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("pets_v2_targets/credit_confirmatory_training_targets.csv"),
    )
    args = parser.parse_args()

    roles = parse_role(args.role)
    source = pd.read_csv(args.source_targets)
    stress_evidence = discovery_stress_evidence(
        pd.read_csv(args.stress_evidence), roles=roles
    )
    required = {"structural_cell_id", "data_seed", "model_seed"}
    missing = required - set(source)
    if missing:
        parser.error(f"source target table lacks columns: {sorted(missing)}")
    prior_frames = [source]
    for path in args.exclude_targets:
        if path.exists() and path.resolve() not in {
            args.structural_out.resolve(),
            args.out.resolve(),
        }:
            prior_frames.append(pd.read_csv(path))
    forbidden_data, forbidden_model = used_seeds(prior_frames)
    structural = build_structural_targets(
        source,
        roles=roles,
        blocks=args.blocks,
        data_seed_start=args.data_seed_start,
        model_seed_start=args.model_seed_start,
        forbidden_data_seeds=forbidden_data,
        forbidden_model_seeds=forbidden_model,
    )

    selection = json.loads(args.selection.read_text())
    l2_weight_decay = float(selection["selected_l2_weight_decay"])
    dp_epsilon = float(selection["selected_dp_epsilon"])
    training = expand_variants(
        structural,
        l2_weight_decay=l2_weight_decay,
        hamp_gamma=0.95,
        hamp_alpha=0.001,
        dp_target_epsilon=dp_epsilon,
        dp_max_grad_norm=1.0,
        dp_delta=1e-5,
        dp_batch_size=32,
        dp_epochs=30,
        dp_learning_rate=0.05,
    )
    training["experiment"] = "petsv2_credit_confirmatory_training"
    training["confirmatory_protocol"] = "pets_credit_three_regime_v2"
    training["dp_comparison_role"] = "literature_schedule_primary"

    args.structural_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    structural.to_csv(args.structural_out, index=False)
    training.to_csv(args.out, index=False)
    metadata = {
        "protocol": "pets_credit_three_regime_v2",
        "status": "prospective_uninspected",
        "roles": roles,
        "blocks": int(args.blocks),
        "independent_unit": "paired target-model block",
        "data_seeds": sorted(structural.data_seed.astype(int).unique().tolist()),
        "model_seeds": sorted(structural.model_seed.astype(int).unique().tolist()),
        "training_defenses": ["none", "l2", "hamp_train", "dp_qml"],
        "literature_defenses": [
            "l2",
            "dp_qml",
            "hamp_full",
            "memguard",
            "dynanoise",
        ],
        "dp_protocol": "watkins_faithful_core_v2",
        "dp_schedule": {"optimizer": "rmsprop", "batch_size": 32, "epochs": 30},
        "selected_l2_weight_decay": l2_weight_decay,
        "selected_dp_epsilon": dp_epsilon,
        "primary_attack": "lira_online_fixed_variance",
        "primary_role": "stress",
        "primary_estimand": "paired AUC(defense)-AUC(none) in the stress role",
        "structural_role_selection": stress_evidence,
        "lira_references": 16,
        "lira_score_protocol": LIRA_SCORE_PROTOCOL,
        "evaluation_nonmember_multiplier": 10,
        "secondary_lira_scores": [
            "lira_online",
            "lira_offline",
            "lira_offline_fixed_variance",
            "lira_offline_density_surprise",
            "lira_offline_density_surprise_fixed_variance",
            "lira_offline_one_sided_z",
            "lira_offline_one_sided_z_fixed_variance",
        ],
        "one_sided_lira_status": (
            "paper-defined log-CDF is canonical offline LiRA; z scores are retained "
            "as rank-equivalent compatibility aliases; released-code OUT-density "
            "scores are retained under explicit auxiliary names"
        ),
        "secondary_structural_contrasts": [
            "repetition-low",
            "stress-low",
            "stress-repetition",
        ],
        "old_pilot_reused": False,
        "inputs": {
            "source_targets": {
                "path": str(args.source_targets.resolve()),
                "sha256": file_sha256(args.source_targets),
            },
            "selection": {
                "path": str(args.selection.resolve()),
                "sha256": file_sha256(args.selection),
            },
            "stress_evidence": {
                "path": str(args.stress_evidence.resolve()),
                "sha256": file_sha256(args.stress_evidence),
            },
            "excluded_prior_manifests": [
                {
                    "path": str(path.resolve()),
                    "sha256": file_sha256(path),
                }
                for path in args.exclude_targets
                if path.exists()
            ],
        },
    }
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"[OK] structural={len(structural)} training={len(training)} "
        f"-> {args.out.resolve()}"
    )


if __name__ == "__main__":
    main()
