#!/usr/bin/env python3
"""Summarize the prespecified Credit hard-label HSJ query-budget pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


BUDGET_PATTERN = re.compile(r"^q(?P<budget>[0-9]+)$")


def load_target_results(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for directory in sorted(root.iterdir() if root.exists() else []):
        match = BUDGET_PATTERN.match(directory.name)
        path = directory / "label_only_hsj_raw.csv"
        if not match or not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "query_budget", int(match.group("budget")))
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No q*/label_only_hsj_raw.csv files under {root}")
    return pd.concat(frames, ignore_index=True)


def load_sample_results(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for directory in sorted(root.iterdir() if root.exists() else []):
        match = BUDGET_PATTERN.match(directory.name)
        score_dir = directory / "sample_scores"
        if not match or not score_dir.exists():
            continue
        for path in sorted(score_dir.glob("*.csv")):
            frame = pd.read_csv(path)
            frame.insert(0, "query_budget", int(match.group("budget")))
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No q*/sample_scores/*.csv files under {root}")
    return pd.concat(frames, ignore_index=True)


def validate_common_candidates(samples: pd.DataFrame) -> None:
    identity = ["target_id", "sample_id", "membership", "true_label", "source_split", "source_index"]
    budgets = sorted(samples.query_budget.unique())
    for budget in budgets:
        observed = samples[samples.query_budget.eq(budget)]
        if observed.duplicated(identity).any():
            raise ValueError(f"Pilot contains duplicate candidate identities at query budget {budget}")
    reference = samples[samples.query_budget.eq(budgets[0])][identity].sort_values(identity).reset_index(drop=True)
    for budget in budgets[1:]:
        observed = samples[samples.query_budget.eq(budget)][identity].sort_values(identity).reset_index(drop=True)
        if not reference.equals(observed):
            raise ValueError(f"Pilot candidate identities differ at query budget {budget}")


def budget_summary(targets: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "auc",
        "mean_label_queries",
        "max_observed_label_queries",
        "initial_accuracy",
        "boundary_found_fraction_among_correct",
        "search_censored_fraction",
        "search_censored_member_fraction",
        "search_censored_nonmember_fraction",
        "declared_iterations_completed_fraction_among_initialized_correct",
        "query_budget_exhausted_fraction",
    ]
    summary = targets.groupby("query_budget", sort=True)[metrics].agg(["mean", "std", "min", "max"]).reset_index()
    summary.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary["n_pilot_targets"] = targets.groupby("query_budget").size().to_numpy()
    summary["interpretation"] = (
        "pilot-only query/censoring diagnostic; do not select the final budget by target AUC"
    )
    return summary


def convergence_summary(samples: pd.DataFrame) -> pd.DataFrame:
    budgets = sorted(int(value) for value in samples.query_budget.unique())
    reference_budget = max(budgets)
    keys = ["target_id", "sample_id", "membership", "true_label"]
    reference = samples[samples.query_budget.eq(reference_budget)].copy()
    rows: list[dict[str, object]] = []
    for budget in budgets:
        if budget == reference_budget:
            continue
        current = samples[samples.query_budget.eq(budget)].copy()
        merged = current.merge(
            reference,
            on=keys,
            how="inner",
            suffixes=("_current", "_reference"),
            validate="one_to_one",
        )
        comparable = merged[
            merged["initially_correct_current"].astype(bool)
            & merged["initially_correct_reference"].astype(bool)
            & ~merged["search_censored_current"].astype(bool)
            & ~merged["search_censored_reference"].astype(bool)
        ].copy()
        if len(comparable) >= 2:
            rho = float(
                spearmanr(
                    comparable["boundary_distance_current"],
                    comparable["boundary_distance_reference"],
                ).statistic
            )
            absolute = (
                comparable["boundary_distance_current"]
                - comparable["boundary_distance_reference"]
            ).abs()
            mean_absolute = float(absolute.mean())
            median_absolute = float(absolute.median())
        else:
            rho = mean_absolute = median_absolute = float("nan")
        rows.append(
            {
                "query_budget": budget,
                "reference_query_budget": reference_budget,
                "n_common_candidates": int(len(merged)),
                "n_uncensored_initially_correct_pairs": int(len(comparable)),
                "boundary_distance_spearman": rho,
                "boundary_distance_mean_absolute_difference": mean_absolute,
                "boundary_distance_median_absolute_difference": median_absolute,
                "censored_fraction": float(merged["search_censored_current"].mean()),
                "reference_censored_fraction": float(merged["search_censored_reference"].mean()),
                "mean_queries": float(merged["total_label_queries_current"].mean()),
                "reference_mean_queries": float(merged["total_label_queries_reference"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-root",
        type=Path,
        default=Path("satml_results/credit_factorial/label_only_hsj_pilot"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.pilot_root.resolve()
    out = (args.out_dir or (root / "analysis")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    targets = load_target_results(root)
    samples = load_sample_results(root)
    validate_common_candidates(samples)
    targets.to_csv(out / "pilot_target_results.csv", index=False)
    budget_summary(targets).to_csv(out / "pilot_budget_summary.csv", index=False)
    convergence_summary(samples).to_csv(out / "pilot_score_convergence.csv", index=False)
    metadata = {
        "pilot_root": str(root),
        "query_budgets": sorted(int(value) for value in targets.query_budget.unique()),
        "target_ids": sorted(targets.target_id.astype(str).unique()),
        "candidate_policy": "fixed deterministic member/nonmember subset shared across budgets",
        "query_schedule_policy": (
            "budget-specific search schedules were prespecified; candidate selection and "
            "deterministic seed derivation are common across budgets"
        ),
        "selection_guard": (
            "The pilot assesses runtime, initialization/censoring, and score convergence. "
            "The final budget must not be selected by maximizing target-set MIA AUC."
        ),
        "default_full_budget": 512,
        "high_budget_role": "approximately-2500-query sensitivity analysis",
    }
    (out / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[OK] pilot targets={len(targets)} samples={len(samples)} -> {out}")


if __name__ == "__main__":
    main()
