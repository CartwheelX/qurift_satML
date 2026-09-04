#!/usr/bin/env python3
"""Analyze the prospective PETS v2 defense study at the paired-block level."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pets_tools.analyze_defenses import effective_defense
from pets_tools.check_protocol_integrity import (
    V2_EVALUATION_PROTOCOL,
    V2_HSJ_PROTOCOL,
    V2_LIRA_PROTOCOL,
    V2_QUERY_PROTOCOL,
)
from qurift.defenses.protocol import (
    CONFIRMATORY_PARTITION_PROTOCOL,
    PARTITION_PROTOCOL,
)
from qurift.defenses.protocol_pooled import (
    POOLED_CONFIRMATORY_PARTITION_PROTOCOL,
    POOLED_PARTITION_PROTOCOL,
)
from reviewer_tools.qurift_lira_attack import LIRA_SCORE_PROTOCOL


# Taken from the integrity checker rather than restated here. When a writer's
# protocol version is bumped, a copy kept in this file silently rejects every
# freshly produced artifact as "stale"; importing makes that impossible.
EVALUATION_PROTOCOL = V2_EVALUATION_PROTOCOL
HSJ_PROTOCOL = V2_HSJ_PROTOCOL
LIRA_PROTOCOL = V2_LIRA_PROTOCOL
QUERY_PROTOCOL = V2_QUERY_PROTOCOL

# Evaluation runs on the widened pooled partition; HSJ, LiRA, and query stress
# run on the common-quota base partition. The two pre-confirmatory contracts stay
# accepted so earlier pilot artifacts remain readable.
SUPPORTED_PARTITION_PROTOCOLS = frozenset(
    {
        PARTITION_PROTOCOL,
        POOLED_PARTITION_PROTOCOL,
        CONFIRMATORY_PARTITION_PROTOCOL,
        POOLED_CONFIRMATORY_PARTITION_PROTOCOL,
    }
)
PRIMARY_ATTACK = "lira_online_fixed_variance"
LIRA_SCORE_FAMILY = (
    "lira_online",
    "lira_online_fixed_variance",
    "lira_offline",
    "lira_offline_fixed_variance",
    "lira_offline_density_surprise",
    "lira_offline_density_surprise_fixed_variance",
    "lira_offline_one_sided_z",
    "lira_offline_one_sided_z_fixed_variance",
)
LIRA_RANK_EQUIVALENT_ALIASES = {
    "lira_offline_one_sided_z": "lira_offline",
    "lira_offline_one_sided_z_fixed_variance": (
        "lira_offline_fixed_variance"
    ),
}

# The canonical offline score is log Phi(z), which is strictly increasing in z,
# so the alias and the canonical score rank identically in exact arithmetic.
# In float64 they cannot: log Phi(z) reaches the smallest representable
# magnitude near z = 38 and underflows to -0.0 above it, so every record further
# out in the upper tail collapses to one tied value while the alias keeps them
# ordered. The tie moves the affected records within the ranking and perturbs
# the reported metrics slightly.
#
# The tolerance bounds that perturbation rather than hiding it. Every reported
# metric is a proportion over the evaluation members, so the smallest non-zero
# difference any single record can produce is 1 / members; at the confirmatory
# 100 members that is 0.01. Allowing twice that admits the underflow ties and
# still rejects a genuine divergence, which would be far larger. The observed
# maximum is written to the output so the artifact records what actually
# happened instead of only that a threshold was met.
LIRA_ALIAS_METRIC_TOLERANCE = 0.02
LITERATURE_DEFENSES = ("l2", "dp_qml", "hamp_full", "memguard", "dynanoise")
STRUCTURAL_CONTRASTS = (
    ("repetition", "low"),
    ("stress", "low"),
    ("stress", "repetition"),
)


def bootstrap_mean(
    values: Sequence[float], *, draws: int, seed: int
) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return (float("nan"),) * 4
    rng = np.random.default_rng(int(seed))
    sampled = rng.choice(array, size=(int(draws), len(array)), replace=True).mean(1)
    return (
        float(array.mean()),
        float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        float(np.quantile(sampled, 0.025)),
        float(np.quantile(sampled, 0.975)),
    )


def exact_sign_flip_p(values: Sequence[float]) -> float:
    """Two-sided paired randomization p-value under sign exchangeability."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan")
    observed = abs(float(array.mean()))
    if len(array) <= 20:
        statistics = []
        for signs in itertools.product((-1.0, 1.0), repeat=len(array)):
            statistics.append(abs(float(np.mean(array * np.asarray(signs)))))
        return float(np.mean(np.asarray(statistics) >= observed - 1e-15))
    rng = np.random.default_rng(2027)
    signs = rng.choice((-1.0, 1.0), size=(100_000, len(array)))
    statistics = np.abs((signs * array).mean(1))
    return float((1 + np.sum(statistics >= observed - 1e-15)) / (len(statistics) + 1))


def holm_adjust(values: Iterable[float]) -> np.ndarray:
    pvalues = np.asarray(list(values), dtype=float)
    adjusted = np.full(len(pvalues), np.nan)
    finite = np.flatnonzero(np.isfinite(pvalues))
    if not len(finite):
        return adjusted
    order = finite[np.argsort(pvalues[finite])]
    running = 0.0
    count = len(order)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def check_partition_protocol(value: Any, path: Path) -> None:
    if value not in SUPPORTED_PARTITION_PROTOCOLS:
        raise ValueError(f"{path} uses unsupported partition protocol {value!r}")


def load_privacy(results: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(results.glob("*/adaptive_attack_metrics.csv")):
        metadata_path = path.with_name("evaluation_metadata.json")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("protocol") != EVALUATION_PROTOCOL:
            raise ValueError(f"stale evaluation artifact: {metadata_path}")
        check_partition_protocol(metadata.get("partition_protocol"), metadata_path)
        frame = pd.read_csv(path)
        frame["source_protocol"] = EVALUATION_PROTOCOL
        frames.append(frame)
    for path in sorted(results.glob("*/hsj/*_metrics.json")):
        payload = json.loads(path.read_text())
        if payload.get("protocol") != HSJ_PROTOCOL:
            raise ValueError(f"stale HSJ artifact: {path}")
        check_partition_protocol(payload.get("partition_protocol"), path)
        fixed = {
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
        frames.append(pd.DataFrame([{**fixed, **payload["metrics"], "source_protocol": HSJ_PROTOCOL}]))
    for path in sorted(results.glob("*/lira/*_metrics.json")):
        payload = json.loads(path.read_text())
        if payload.get("protocol") != LIRA_PROTOCOL:
            raise ValueError(f"stale LiRA artifact: {path}")
        if payload.get("lira_score_protocol") != LIRA_SCORE_PROTOCOL:
            raise ValueError(f"stale LiRA score semantics: {path}")
        check_partition_protocol(payload.get("partition_protocol"), path)
        rows = payload.get("rows", [])
        if rows:
            frame = pd.DataFrame(rows)
            frame["source_protocol"] = LIRA_PROTOCOL
            frames.append(frame)
    for path in sorted(results.glob("*/query_stress/metrics.json")):
        payload = json.loads(path.read_text())
        if payload.get("protocol") != QUERY_PROTOCOL:
            raise ValueError(f"stale query-stress artifact: {path}")
        check_partition_protocol(payload.get("partition_protocol"), path)
        rows = payload.get("rows", [])
        if rows:
            frame = pd.DataFrame(rows)
            frame["source_protocol"] = QUERY_PROTOCOL
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no PETS v2 privacy artifacts under {results}")
    result = pd.concat(frames, ignore_index=True)
    result["effective_defense"] = [
        effective_defense(training, output)
        for training, output in zip(result.training_defense, result.defense)
    ]
    return result


def load_utility(results: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(results.glob("*/evaluation_metadata.json")):
        payload = json.loads(path.read_text())
        if payload.get("protocol") != EVALUATION_PROTOCOL:
            raise ValueError(f"stale evaluation artifact: {path}")
        check_partition_protocol(payload.get("partition_protocol"), path)
        target = payload["target"]
        training = str(target.get("training_defense", "none"))
        for output, condition in payload["conditions"].items():
            rows.append(
                {
                    "target_id": target["target_id"],
                    "block_id": target.get("block_id"),
                    "structural_role": target.get(
                        "defense_structural_role", target.get("role")
                    ),
                    "training_defense": training,
                    "defense": output,
                    "effective_defense": effective_defense(training, output),
                    **condition["utility"],
                }
            )
    if not rows:
        raise FileNotFoundError(f"no PETS v2 utility artifacts under {results}")
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, group: Sequence[str], outcomes: Sequence[str]) -> pd.DataFrame:
    rows = []
    for keys, values in frame.groupby(list(group), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        fixed = dict(zip(group, keys))
        for outcome in outcomes:
            if outcome not in values:
                continue
            array = values[outcome].dropna().astype(float)
            rows.append(
                {
                    **fixed,
                    "outcome": outcome,
                    "blocks": len(array),
                    "mean": array.mean(),
                    "sd_across_blocks": array.std(ddof=1),
                }
            )
    return pd.DataFrame(rows)


def validate_primary_matrix(privacy: pd.DataFrame) -> None:
    """Fail rather than silently analyze an incomplete primary family."""

    expected_defenses = {"none", *LITERATURE_DEFENSES}
    selected = privacy[
        privacy.attack.eq(PRIMARY_ATTACK)
        & privacy.structural_role.eq("stress")
        & privacy.effective_defense.isin(expected_defenses)
    ]
    duplicates = selected.duplicated(["block_id", "effective_defense"], keep=False)
    if duplicates.any():
        pairs = selected.loc[duplicates, ["block_id", "effective_defense"]]
        raise ValueError(f"duplicate primary endpoints:\n{pairs.to_string(index=False)}")
    observed = set(
        zip(selected.block_id.astype(str), selected.effective_defense.astype(str))
    )
    blocks = sorted(privacy.block_id.astype(str).unique())
    expected = {(block, defense) for block in blocks for defense in expected_defenses}
    if observed != expected:
        raise ValueError(
            "incomplete primary endpoint matrix: "
            f"missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )


def validate_lira_alias_equivalence(privacy: pd.DataFrame) -> pd.DataFrame:
    """Verify and document that z aliases preserve the paper-score ranking."""

    lira = privacy[privacy.source_protocol.eq(LIRA_PROTOCOL)].copy()
    keys = ["target_id", "effective_defense", "structural_role"]
    metrics = [
        column
        for column in (
            "auc",
            "balanced_accuracy",
            "tpr_at_0_1_fpr",
            "tpr_at_1_fpr",
            "tpr_at_5_fpr",
            "tpr_at_10_fpr",
        )
        if column in lira
    ]
    rows = []
    for alias, canonical in LIRA_RANK_EQUIVALENT_ALIASES.items():
        left = lira[lira.attack.eq(canonical)][keys + metrics].copy()
        right = lira[lira.attack.eq(alias)][keys + metrics].copy()
        merged = left.merge(
            right,
            on=keys,
            how="outer",
            suffixes=("_canonical", "_alias"),
            indicator=True,
        )
        if not merged._merge.eq("both").all():
            raise ValueError(
                f"LiRA compatibility alias {alias!r} is incomplete relative to "
                f"{canonical!r}"
            )
        maximum = 0.0
        for metric in metrics:
            difference = (
                pd.to_numeric(merged[f"{metric}_canonical"], errors="coerce")
                - pd.to_numeric(merged[f"{metric}_alias"], errors="coerce")
            ).abs()
            finite = difference[np.isfinite(difference)]
            if len(finite):
                maximum = max(maximum, float(finite.max()))
        if maximum > LIRA_ALIAS_METRIC_TOLERANCE:
            raise ValueError(
                f"LiRA alias {alias!r} is not rank-equivalent to {canonical!r}; "
                f"maximum metric difference={maximum} exceeds the float-underflow "
                f"allowance of {LIRA_ALIAS_METRIC_TOLERANCE}"
            )
        rows.append(
            {
                "canonical_attack": canonical,
                "compatibility_alias": alias,
                "paired_artifacts": int(len(merged)),
                "maximum_reported_metric_difference": maximum,
                "included_in_inferential_multiplicity": False,
            }
        )
    return pd.DataFrame(rows)


def low_fpr_resolution_table(privacy: pd.DataFrame) -> pd.DataFrame:
    """Expose what low-FPR operating points each attack can actually resolve."""

    required = {"n_evaluation_nonmember", "empirical_fpr_resolution"}
    if not required.issubset(privacy):
        return pd.DataFrame()
    rows = []
    grouping = ["source_protocol", "attack", "effective_defense"]
    for keys, frame in privacy.groupby(grouping, dropna=False):
        row = dict(zip(grouping, keys))
        nonmembers = pd.to_numeric(frame.n_evaluation_nonmember, errors="coerce")
        resolution = pd.to_numeric(frame.empirical_fpr_resolution, errors="coerce")
        row.update(
            {
                "blocks_or_targets": int(len(frame)),
                "minimum_nonmembers": int(nonmembers.min()) if nonmembers.notna().any() else 0,
                "maximum_nonmembers": int(nonmembers.max()) if nonmembers.notna().any() else 0,
                "coarsest_empirical_fpr_resolution": (
                    float(resolution.max()) if resolution.notna().any() else float("nan")
                ),
            }
        )
        for label in ("0_1", "1", "5", "10"):
            column = f"target_{label}_fpr_resolvable"
            if column in frame:
                values = frame[column].astype(str).str.lower().map(
                    {"true": True, "false": False}
                )
                row[f"fraction_target_{label}_fpr_resolvable"] = float(values.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_defense_effects(
    privacy: pd.DataFrame, *, draws: int, seed: int
) -> pd.DataFrame:
    rows = []
    for (attack, role), group in privacy.groupby(["attack", "structural_role"], dropna=False):
        pivot = group.pivot_table(
            index="block_id", columns="effective_defense", values="auc", aggfunc="first"
        )
        if "none" not in pivot:
            continue
        for defense in sorted(set(pivot) - {"none"}):
            values = (pivot[defense] - pivot["none"]).dropna().to_numpy(dtype=float)
            mean, sd, low, high = bootstrap_mean(
                values, draws=draws, seed=seed + len(rows)
            )
            rows.append(
                {
                    "attack": attack,
                    "structural_role": role,
                    "effective_defense": defense,
                    "contrast": "defense-none",
                    "mean_auc_difference": mean,
                    "sd_across_paired_blocks": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                    "paired_blocks": len(values),
                    "p_exact_sign_flip": exact_sign_flip_p(values),
                }
            )
    result = pd.DataFrame(rows)
    if len(result):
        result["p_holm_all_secondary"] = holm_adjust(result.p_exact_sign_flip)
    return result


def structural_effects(
    privacy: pd.DataFrame, *, draws: int, seed: int
) -> pd.DataFrame:
    rows = []
    for (defense, attack), group in privacy.groupby(
        ["effective_defense", "attack"], dropna=False
    ):
        pivot = group.pivot_table(
            index="block_id", columns="structural_role", values="auc", aggfunc="first"
        )
        for upper, lower in STRUCTURAL_CONTRASTS:
            if upper not in pivot or lower not in pivot:
                continue
            values = (pivot[upper] - pivot[lower]).dropna().to_numpy(dtype=float)
            mean, sd, low, high = bootstrap_mean(
                values, draws=draws, seed=seed + len(rows)
            )
            rows.append(
                {
                    "effective_defense": defense,
                    "attack": attack,
                    "contrast": f"{upper}-{lower}",
                    "mean_auc_difference": mean,
                    "sd_across_paired_blocks": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                    "paired_blocks": len(values),
                    "p_exact_sign_flip": exact_sign_flip_p(values),
                }
            )
    result = pd.DataFrame(rows)
    if len(result):
        result["p_holm_all_secondary"] = holm_adjust(result.p_exact_sign_flip)
    return result


def structural_did(
    privacy: pd.DataFrame, *, draws: int, seed: int
) -> pd.DataFrame:
    rows = []
    for attack, group in privacy.groupby("attack", dropna=False):
        pivot = group.pivot_table(
            index="block_id",
            columns=["effective_defense", "structural_role"],
            values="auc",
            aggfunc="first",
        )
        defenses = sorted(set(pivot.columns.get_level_values(0)))
        for upper, lower in STRUCTURAL_CONTRASTS:
            if ("none", upper) not in pivot or ("none", lower) not in pivot:
                continue
            baseline = pivot[("none", upper)] - pivot[("none", lower)]
            for defense in defenses:
                if defense == "none" or (defense, upper) not in pivot or (defense, lower) not in pivot:
                    continue
                defended = pivot[(defense, upper)] - pivot[(defense, lower)]
                values = (defended - baseline).dropna().to_numpy(dtype=float)
                mean, sd, low, high = bootstrap_mean(
                    values, draws=draws, seed=seed + len(rows)
                )
                rows.append(
                    {
                        "attack": attack,
                        "effective_defense": defense,
                        "structural_contrast": f"{upper}-{lower}",
                        "contrast": "structural_difference_in_differences",
                        "mean_auc_difference": mean,
                        "sd_across_paired_blocks": sd,
                        "ci95_low": low,
                        "ci95_high": high,
                        "paired_blocks": len(values),
                        "p_exact_sign_flip": exact_sign_flip_p(values),
                    }
                )
    result = pd.DataFrame(rows)
    if len(result):
        result["p_holm_all_secondary"] = holm_adjust(result.p_exact_sign_flip)
    return result


def primary_table(efficacy: pd.DataFrame) -> pd.DataFrame:
    result = efficacy[
        efficacy.attack.eq(PRIMARY_ATTACK)
        & efficacy.structural_role.eq("stress")
        & efficacy.effective_defense.isin(LITERATURE_DEFENSES)
    ].copy()
    order = {name: index for index, name in enumerate(LITERATURE_DEFENSES)}
    result["predeclared_order"] = result.effective_defense.map(order)
    result = result.sort_values("predeclared_order").drop(columns="predeclared_order")
    result["p_holm_primary_family"] = holm_adjust(result.p_exact_sign_flip)
    result["holm_reject_0_05"] = result.p_holm_primary_family.le(0.05)
    result["interpretation"] = np.where(
        result.holm_reject_0_05 & result.mean_auc_difference.lt(0),
        "family-wise evidence of lower AUC than undefended",
        np.where(
            result.holm_reject_0_05 & result.mean_auc_difference.gt(0),
            "family-wise evidence of higher AUC than undefended",
            "no family-wise resolved change; report interval and resolution limit",
        ),
    )
    return result


def make_primary_plot(primary: pd.DataFrame, out_dir: Path) -> None:
    if primary.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    view = primary.iloc[::-1].reset_index(drop=True)
    errors = np.vstack(
        [
            view.mean_auc_difference - view.ci95_low,
            view.ci95_high - view.mean_auc_difference,
        ]
    )
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    axis.errorbar(
        view.mean_auc_difference,
        np.arange(len(view)),
        xerr=errors,
        fmt="o",
        capsize=3,
    )
    axis.set_yticks(np.arange(len(view)), view.effective_defense)
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Paired online-LiRA AUC difference (defense − none)")
    axis.set_title("Defense efficacy in the predeclared stress regime")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(out_dir / f"primary_defense_efficacy.{suffix}", dpi=240)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("pets_v2_results/defenses"))
    parser.add_argument("--out-dir", type=Path, default=Path("pets_v2_results/analysis"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    privacy = load_privacy(args.results_dir)
    utility = load_utility(args.results_dir)
    blocks = sorted(privacy.block_id.astype(str).unique())
    roles = set(privacy.structural_role.astype(str))
    if roles != {"low", "repetition", "stress"}:
        raise ValueError(f"analysis requires all three structural roles, got {sorted(roles)}")
    if len(blocks) != 8:
        raise ValueError(f"confirmatory analysis requires exactly 8 complete blocks, got {len(blocks)}")
    validate_primary_matrix(privacy)
    lira_alias_equivalence = validate_lira_alias_equivalence(privacy)
    inferential_privacy = privacy[
        ~privacy.attack.isin(LIRA_RANK_EQUIVALENT_ALIASES)
    ].copy()

    privacy_summary = summarize(
        privacy,
        ["effective_defense", "structural_role", "attack"],
        [
            "auc",
            "balanced_accuracy",
            "tpr_at_0_1_fpr",
            "tpr_at_1_fpr",
            "tpr_at_5_fpr",
            "tpr_at_10_fpr",
        ],
    )
    utility_summary = summarize(
        utility,
        ["effective_defense", "structural_role"],
        [
            "accuracy",
            "balanced_accuracy",
            "task_roc_auc",
            "task_average_precision",
            "minority_class_recall",
            "prediction_collapse",
            "nll",
        ],
    )
    efficacy = paired_defense_effects(
        inferential_privacy, draws=args.bootstrap, seed=args.seed
    )
    primary = primary_table(efficacy)
    structural = structural_effects(
        inferential_privacy, draws=args.bootstrap, seed=args.seed + 10_000
    )
    did = structural_did(
        inferential_privacy, draws=args.bootstrap, seed=args.seed + 20_000
    )
    low_fpr = low_fpr_resolution_table(privacy)
    lira_score_family = privacy[
        privacy.attack.isin(LIRA_SCORE_FAMILY)
    ].copy()
    lira_score_family = summarize(
        lira_score_family,
        ["effective_defense", "structural_role", "attack"],
        ["auc", "tpr_at_1_fpr", "tpr_at_5_fpr", "tpr_at_10_fpr"],
    )

    outputs = {
        "privacy_raw.csv": privacy,
        "utility_raw.csv": utility,
        "privacy_summary.csv": privacy_summary,
        "utility_summary.csv": utility_summary,
        "primary_defense_efficacy.csv": primary,
        "secondary_defense_efficacy.csv": efficacy,
        "secondary_structural_contrasts.csv": structural,
        "secondary_structural_did.csv": did,
        "lira_score_family_summary.csv": lira_score_family,
        "lira_alias_equivalence.csv": lira_alias_equivalence,
        "low_fpr_resolution.csv": low_fpr,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.out_dir / name, index=False)
    metadata = {
        "protocol": "pets_credit_confirmatory_analysis_v2",
        "blocks": blocks,
        "independent_unit": "paired target-model block",
        "primary_attack": PRIMARY_ATTACK,
        "primary_role": "stress",
        "primary_defenses": list(LITERATURE_DEFENSES),
        "reported_lira_score_family": list(LIRA_SCORE_FAMILY),
        "lira_score_protocol": LIRA_SCORE_PROTOCOL,
        "one_sided_lira_status": (
            "paper-defined log-CDF is canonical; explicit z names are rank-equivalent "
            "compatibility aliases and are excluded from multiplicity; the released-"
            "artifact OUT-density score is retained under an explicit auxiliary name"
        ),
        "primary_multiplicity": "Holm correction over five literature defenses",
        "secondary_multiplicity": "Holm correction over each exported secondary table",
        "confidence_interval": "percentile bootstrap over paired blocks",
        "hypothesis_test": "exact two-sided paired sign-flip randomization",
        "bootstrap_draws": int(args.bootstrap),
        "bootstrap_seed": int(args.seed),
        "low_fpr_rule": (
            "report attained FPR and empirical resolution; 0.1% is not claimed "
            "when the non-member pool cannot resolve it"
        ),
    }
    (args.out_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    make_primary_plot(primary, args.out_dir)
    print(f"[DONE] PETS v2 analysis -> {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
