#!/usr/bin/env python3
"""Hard-label HopSkipJump-style boundary-distance MIA for QuRiFT targets.

The attacker observes only predicted class labels and knows each candidate's
true label.  Initially misclassified candidates receive distance zero, as in
Choquette-Choo et al. (ICML 2021).  Correctly classified candidates receive a
uniform, bounded decision-based search:

1. sample random points inside the declared input domain until an untargeted
   adversarial initialization is found;
2. project the initialization to the decision boundary by binary search;
3. estimate a boundary normal from hard-label queries; and
4. take geometric search steps followed by repeated boundary projection.

This is an independent PyTorch/TorchQuantum implementation of the core
HopSkipJump procedure.  It is not a copy or bitwise reproduction of the old
TensorFlow/CleverHans reference code.  Every eligible record receives the
same nominal algorithm and maximum query budget.  If no adversarial
initialization is found, the operational score is capped at the predeclared L2
diameter of the input box and explicitly marked search-censored rather than
NaN.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from qurift_target_loader import (
    build_config,
    build_dataset,
    import_qurift_main,
    instantiate_model,
    load_saved_model,
    preprocess_like_train,
    read_target_row,
    resolve_target_paths,
    sample_dataset_split,
    select_member_nonmember_samples,
)
from reviewer_common import (
    CI_RECORD,
    atomic_write_csv,
    atomic_write_json,
    cross_fitted_threshold_metrics,
    stable_seed,
    stratified_bootstrap_auc,
    tpr_at_resolvable_fpr,
    write_analysis_metadata,
)


REFERENCE_REPOSITORY = "cchoquette/membership-inference"
REFERENCE_COMMIT = "ce12e12139b61b8d042ec38bba2eeac56b55b357"
REFERENCE_PAPER = "Choquette-Choo et al., Label-Only Membership Inference Attacks, ICML 2021"
PROTOCOL_VERSION = "qurift_label_only_hsj_v1"


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()) or "unnamed"


@torch.no_grad()
def query_class_labels(
    model: torch.nn.Module,
    raw_inputs: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Return hard labels; score-valued model outputs are immediately discarded."""
    if len(raw_inputs) == 0:
        return torch.empty(0, dtype=torch.long)
    model.eval()
    predictions: list[torch.Tensor] = []
    for start in range(0, len(raw_inputs), int(batch_size)):
        inputs = preprocess_like_train(raw_inputs[start : start + batch_size], device)
        predictions.append(model(inputs).argmax(dim=1).detach().cpu())
    return torch.cat(predictions)


def choose_evaluation_indices(
    membership: np.ndarray,
    n_member: int | None,
    n_nonmember: int | None,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    selected: list[np.ndarray] = []
    for value, requested in ((1, n_member), (0, n_nonmember)):
        available = np.flatnonzero(membership == value)
        if requested is None or requested <= 0 or requested >= len(available):
            chosen = available
        else:
            chosen = np.sort(rng.choice(available, size=requested, replace=False))
        selected.append(chosen)
    return np.sort(np.concatenate(selected))


@dataclass(frozen=True)
class InputBounds:
    lower: float
    upper: float
    source: str

    def tensors_like(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.full_like(value, self.lower), torch.full_like(value, self.upper)


def input_bounds_for_dataset(
    dataset_name: str,
    *,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> InputBounds:
    """Return the predeclared raw-input domain used by the target loader."""
    if (clip_min is None) != (clip_max is None):
        raise ValueError("--clip-min and --clip-max must be supplied together")
    if clip_min is not None and clip_max is not None:
        if not float(clip_min) < float(clip_max):
            raise ValueError("clip_min must be smaller than clip_max")
        return InputBounds(float(clip_min), float(clip_max), "explicit_cli_override")

    name = str(dataset_name).strip().lower()
    if name in {"credit_default", "breast_cancer_wdbc"}:
        return InputBounds(
            -1.0,
            1.0,
            "train-fitted PCA followed by MinMaxScaler(feature_range=(-1,1), clip=True)",
        )
    if name == "fashion_mnist":
        mean, std = 0.2860, 0.3530
        return InputBounds(
            -mean / std,
            (1.0 - mean) / std,
            "ToTensor pixel domain [0,1] followed by Fashion-MNIST normalization",
        )
    if name == "mnist":
        mean, std = 0.1307, 0.3081
        return InputBounds(
            -mean / std,
            (1.0 - mean) / std,
            "ToTensor pixel domain [0,1] followed by MNIST normalization",
        )
    raise NotImplementedError(
        f"No predeclared HSJ input domain for dataset {dataset_name!r}; "
        "provide both --clip-min and --clip-max"
    )


def validate_inputs_in_bounds(
    inputs: torch.Tensor,
    bounds: InputBounds,
    *,
    tolerance: float = 1e-5,
) -> None:
    observed_min = float(inputs.min().item())
    observed_max = float(inputs.max().item())
    if observed_min < bounds.lower - tolerance or observed_max > bounds.upper + tolerance:
        raise ValueError(
            "Candidate inputs fall outside the declared search domain: "
            f"observed=[{observed_min}, {observed_max}] "
            f"declared=[{bounds.lower}, {bounds.upper}]"
        )


def l2_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.reshape(-1), ord=2).item())


def box_l2_diameter(
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> float:
    """Return one predeclared, record-independent L2 censoring cap."""
    return l2_norm(upper - lower)


class QueryBudgetExhausted(RuntimeError):
    """Raised before a query batch would exceed the per-record budget."""


class BudgetedLabelOracle:
    """Hard-label query adapter with exact per-record accounting."""

    def __init__(
        self,
        query_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        max_queries: int,
    ) -> None:
        if max_queries < 0:
            raise ValueError("max_queries cannot be negative")
        self.query_fn = query_fn
        self.max_queries = int(max_queries)
        self.queries = 0

    @property
    def remaining(self) -> int:
        return self.max_queries - self.queries

    def labels(self, points: torch.Tensor) -> torch.Tensor:
        requested = int(len(points))
        if requested <= 0:
            return torch.empty(0, dtype=torch.long)
        if requested > self.remaining:
            raise QueryBudgetExhausted(
                f"requested {requested} hard-label queries with {self.remaining} remaining"
            )
        labels = self.query_fn(points).detach().cpu().long().reshape(-1)
        if len(labels) != requested:
            raise RuntimeError(
                f"label oracle returned {len(labels)} predictions for {requested} inputs"
            )
        self.queries += requested
        return labels


def _random_uniform_points(
    count: int,
    shape: torch.Size,
    lower: torch.Tensor,
    upper: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    unit = torch.rand((int(count), *shape), generator=generator, dtype=torch.float32)
    return lower.unsqueeze(0) + unit * (upper - lower).unsqueeze(0)


def _binary_project_to_boundary(
    origin: torch.Tensor,
    adversarial: torch.Tensor,
    *,
    true_label: int,
    oracle: BudgetedLabelOracle,
    steps: int,
) -> tuple[torch.Tensor, int]:
    """Project an adversarial endpoint toward a correctly classified origin."""
    low = 0.0
    high = 1.0
    before = oracle.queries
    direction = adversarial - origin
    for _ in range(int(steps)):
        if oracle.remaining < 1:
            break
        middle = (low + high) / 2.0
        point = origin + middle * direction
        changed = int(oracle.labels(point.unsqueeze(0))[0].item()) != int(true_label)
        if changed:
            high = middle
        else:
            low = middle
    return origin + high * direction, oracle.queries - before


def _estimate_boundary_normal(
    boundary: torch.Tensor,
    *,
    true_label: int,
    oracle: BudgetedLabelOracle,
    count: int,
    delta: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor | None, int]:
    before = oracle.queries
    requested = min(int(count), oracle.remaining)
    if requested < 2:
        return None, 0
    directions = torch.randn(
        (requested, *boundary.shape), generator=generator, dtype=torch.float32
    )
    flat = directions.reshape(requested, -1)
    flat_norm = torch.linalg.vector_norm(flat, ord=2, dim=1).clamp_min(1e-12)
    directions = directions / flat_norm.reshape((requested,) + (1,) * boundary.ndim)
    probes = torch.clamp(
        boundary.unsqueeze(0) + float(delta) * directions,
        min=lower.unsqueeze(0),
        max=upper.unsqueeze(0),
    )
    effective = probes - boundary.unsqueeze(0)
    effective_flat = effective.reshape(requested, -1)
    effective_norm = torch.linalg.vector_norm(effective_flat, ord=2, dim=1)
    usable = effective_norm > 1e-12
    if int(usable.sum().item()) < 2:
        return None, 0
    probes = probes[usable]
    effective = effective[usable]
    effective_norm = effective_norm[usable]
    effective = effective / effective_norm.reshape(
        (len(effective_norm),) + (1,) * boundary.ndim
    )
    decisions = oracle.labels(probes) != int(true_label)
    signed = decisions.to(torch.float32) * 2.0 - 1.0
    if bool((decisions == decisions[0]).all()):
        weights = signed
    else:
        weights = signed - signed.mean()
    gradient = (weights.reshape((-1,) + (1,) * boundary.ndim) * effective).mean(dim=0)
    norm = torch.linalg.vector_norm(gradient.reshape(-1), ord=2)
    if not torch.isfinite(norm) or float(norm.item()) <= 1e-12:
        return None, oracle.queries - before
    return gradient / norm, oracle.queries - before


def hsj_boundary_distance(
    *,
    origin: torch.Tensor,
    true_label: int,
    original_prediction: int,
    query_fn: Callable[[torch.Tensor], torch.Tensor],
    lower: torch.Tensor,
    upper: torch.Tensor,
    max_queries: int,
    init_queries: int,
    init_batch_size: int,
    iterations: int,
    gradient_samples: int,
    binary_steps: int,
    step_search_steps: int,
    gradient_delta_ratio: float,
    min_gradient_delta: float,
    seed: int,
) -> dict[str, Any]:
    """Estimate a hard-label L2 boundary distance under a fixed nominal budget.

    ``max_queries`` includes the already-observed label of ``origin``.  The
    returned ``boundary_queries`` therefore never exceeds ``max_queries - 1``.
    """
    if max_queries < 1:
        raise ValueError("max_queries must include at least the initial label query")
    if int(original_prediction) != int(true_label):
        return {
            "boundary_distance": 0.0,
            "initially_correct": False,
            "adversarial_initialization_found": False,
            "search_censored": False,
            "cap_distance": box_l2_diameter(lower, upper),
            "initialization_queries": 0,
            "gradient_queries": 0,
            "step_queries": 0,
            "projection_queries": 0,
            "boundary_queries": 0,
            "iterations_completed": 0,
            "query_budget_exhausted": False,
            "stopping_reason": "initially_misclassified",
        }

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    oracle = BudgetedLabelOracle(query_fn, max_queries=max(0, int(max_queries) - 1))
    cap_distance = box_l2_diameter(lower, upper)
    initialization_queries = 0
    gradient_queries = 0
    step_queries = 0
    projection_queries = 0

    # Reserve enough budget to project a found initialization to the boundary.
    initialization: torch.Tensor | None = None
    init_limit = min(int(init_queries), oracle.max_queries)
    while initialization_queries < init_limit:
        reservable = oracle.remaining - int(binary_steps)
        if reservable <= 0:
            break
        count = min(int(init_batch_size), init_limit - initialization_queries, reservable)
        probes = _random_uniform_points(count, origin.shape, lower, upper, generator)
        labels = oracle.labels(probes)
        initialization_queries += count
        adversarial = labels != int(true_label)
        if bool(adversarial.any()):
            candidates = probes[adversarial]
            distances = torch.linalg.vector_norm(
                (candidates - origin.unsqueeze(0)).reshape(len(candidates), -1),
                ord=2,
                dim=1,
            )
            initialization = candidates[int(distances.argmin().item())]
            break

    if initialization is None:
        return {
            "boundary_distance": cap_distance,
            "initially_correct": True,
            "adversarial_initialization_found": False,
            "search_censored": True,
            "cap_distance": cap_distance,
            "initialization_queries": initialization_queries,
            "gradient_queries": 0,
            "step_queries": 0,
            "projection_queries": 0,
            "boundary_queries": oracle.queries,
            "iterations_completed": 0,
            "query_budget_exhausted": oracle.remaining == 0,
            "stopping_reason": "no_adversarial_initialization_within_budget",
        }

    boundary, used = _binary_project_to_boundary(
        origin,
        initialization,
        true_label=true_label,
        oracle=oracle,
        steps=binary_steps,
    )
    projection_queries += used
    best_distance = l2_norm(boundary - origin)
    iterations_completed = 0
    stopping_reason = "iterations_completed"

    for iteration in range(1, int(iterations) + 1):
        # One geometric-step query and a boundary projection must remain after
        # the gradient estimate; otherwise stop without exceeding the budget.
        reserved = 1 + int(binary_steps)
        available_gradient = oracle.remaining - reserved
        if available_gradient < 2:
            stopping_reason = "query_budget_exhausted"
            break
        count = min(int(gradient_samples), available_gradient)
        delta = max(
            float(min_gradient_delta),
            float(gradient_delta_ratio) * max(best_distance, float(min_gradient_delta)),
        )
        gradient, used = _estimate_boundary_normal(
            boundary,
            true_label=true_label,
            oracle=oracle,
            count=count,
            delta=delta,
            lower=lower,
            upper=upper,
            generator=generator,
        )
        gradient_queries += used
        if gradient is None:
            stopping_reason = "boundary_normal_estimation_failed"
            break

        candidate: torch.Tensor | None = None
        step_size = best_distance / math.sqrt(float(iteration))
        for _ in range(int(step_search_steps)):
            if oracle.remaining <= int(binary_steps):
                break
            proposal = torch.clamp(boundary + step_size * gradient, min=lower, max=upper)
            changed = int(oracle.labels(proposal.unsqueeze(0))[0].item()) != int(true_label)
            step_queries += 1
            if changed:
                candidate = proposal
                break
            step_size /= 2.0
        if candidate is None:
            stopping_reason = (
                "query_budget_exhausted"
                if oracle.remaining <= int(binary_steps)
                else "geometric_step_search_failed"
            )
            break

        projected, used = _binary_project_to_boundary(
            origin,
            candidate,
            true_label=true_label,
            oracle=oracle,
            steps=binary_steps,
        )
        projection_queries += used
        projected_distance = l2_norm(projected - origin)
        if projected_distance <= best_distance + 1e-8:
            boundary = projected
            best_distance = projected_distance
        iterations_completed += 1

    return {
        "boundary_distance": best_distance,
        "initially_correct": True,
        "adversarial_initialization_found": True,
        "search_censored": False,
        "cap_distance": cap_distance,
        "initialization_queries": initialization_queries,
        "gradient_queries": gradient_queries,
        "step_queries": step_queries,
        "projection_queries": projection_queries,
        "boundary_queries": oracle.queries,
        "iterations_completed": iterations_completed,
        "query_budget_exhausted": oracle.remaining == 0 or stopping_reason == "query_budget_exhausted",
        "stopping_reason": stopping_reason,
    }


def load_target(args: argparse.Namespace):
    repo_root = args.repo_root.resolve()
    qmain = import_qurift_main(repo_root)
    row = read_target_row(args.targets, args.target_id)
    architecture = str(row.get("architecture", "qnn")).strip().lower()
    if architecture not in {"qnn", "hqnn", "qcnn"}:
        raise NotImplementedError(
            f"Label-only HSJ evaluation does not support {architecture!r}"
        )
    device = torch.device(args.device)
    dataset, feature_dim = build_dataset(qmain, row, repo_root)
    config = build_config(qmain, row, feature_dim, device)
    samples = select_member_nonmember_samples(
        dataset,
        n_member=None,
        n_nonmember=None,
        selection_seed=int(float(row.get("data_seed", 43))),
    )
    model, _ = instantiate_model(qmain, row, config, device)
    model_path, _ = resolve_target_paths(row, args.run_root)
    load_saved_model(model, model_path, device)
    validation_inputs, validation_labels = sample_dataset_split(
        dataset["valid"], list(range(len(dataset["valid"])))
    )
    return row, model, samples, validation_inputs, validation_labels, device


def _histogram(values: torch.Tensor) -> dict[str, int]:
    unique, counts = torch.unique(values.detach().cpu().long(), return_counts=True)
    return {str(int(key.item())): int(value.item()) for key, value in zip(unique, counts)}


def _fraction(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
    selected = frame.loc[mask, column]
    return float(selected.mean()) if len(selected) else float("nan")


def score_target(args: argparse.Namespace) -> None:
    metrics_path = args.out_dir / "target_scores" / f"{safe_name(args.target_id)}.csv"
    if args.resume and metrics_path.exists():
        print(f"[SKIP] target score exists: {metrics_path.resolve()}")
        return

    row, model, samples, validation_inputs, validation_labels, device = load_target(args)
    dataset_name = str(row.get("dataset", "")).strip().lower()
    bounds = input_bounds_for_dataset(
        dataset_name,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )
    validate_inputs_in_bounds(samples.inputs, bounds)
    validate_inputs_in_bounds(validation_inputs, bounds)
    membership = samples.membership.numpy().astype(int)
    labels = samples.labels.numpy().astype(int)
    evaluation_indices = choose_evaluation_indices(
        membership,
        args.n_member,
        args.n_nonmember,
        stable_seed(args.seed, row.get("data_seed", 0), "label_only_hsj_selection"),
    )
    evaluation_tensor = torch.as_tensor(evaluation_indices, dtype=torch.long)
    selected_predictions = query_class_labels(
        model,
        samples.inputs[evaluation_tensor],
        device=device,
        batch_size=args.query_batch_size,
    )
    # Validation labels are queried only for an explicitly reported per-model
    # collapse diagnostic. They are never used to initialize or score the HSJ
    # attack, and therefore cannot change its per-record search opportunity.
    validation_predictions = query_class_labels(
        model, validation_inputs, device=device, batch_size=args.query_batch_size
    )
    lower_scalar = float(bounds.lower)
    upper_scalar = float(bounds.upper)
    rows: list[dict[str, Any]] = []

    def model_query(points: torch.Tensor) -> torch.Tensor:
        return query_class_labels(
            model, points, device=device, batch_size=args.query_batch_size
        )

    for position, sample_index in enumerate(evaluation_indices):
        index = int(sample_index)
        origin = samples.inputs[index].detach().cpu().float()
        lower = torch.full_like(origin, lower_scalar)
        upper = torch.full_like(origin, upper_scalar)
        sample_seed = stable_seed(
            args.seed,
            row.get("data_seed", 0),
            samples.sample_ids[index],
            PROTOCOL_VERSION,
        )
        result = hsj_boundary_distance(
            origin=origin,
            true_label=int(labels[index]),
            original_prediction=int(selected_predictions[position].item()),
            query_fn=model_query,
            lower=lower,
            upper=upper,
            max_queries=args.max_queries,
            init_queries=args.init_queries,
            init_batch_size=args.init_batch_size,
            iterations=args.iterations,
            gradient_samples=args.gradient_samples,
            binary_steps=args.binary_steps,
            step_search_steps=args.step_search_steps,
            gradient_delta_ratio=args.gradient_delta_ratio,
            min_gradient_delta=args.min_gradient_delta,
            seed=sample_seed,
        )
        rows.append(
            {
                "target_id": args.target_id,
                "sample_id": samples.sample_ids[index],
                "sample_index": index,
                "source_split": samples.split_names[index],
                "source_index": samples.source_indices[index],
                "membership": int(membership[index]),
                "true_label": int(labels[index]),
                "predicted_label": int(selected_predictions[position]),
                "search_seed": int(sample_seed),
                **result,
                "initial_label_query": 1,
                "total_label_queries": 1 + int(result["boundary_queries"]),
                "max_label_queries": int(args.max_queries),
                "norm": "l2",
                "clip_min": lower_scalar,
                "clip_max": upper_scalar,
                "protocol_version": PROTOCOL_VERSION,
            }
        )
        if (
            position == 0
            or position + 1 == len(evaluation_indices)
            or (position + 1) % max(len(evaluation_indices) // 20, 1) == 0
        ):
            print(
                f"[{args.target_id}] {position + 1}/{len(evaluation_indices)}",
                flush=True,
            )

    sample_frame = pd.DataFrame(rows)
    if sample_frame.empty:
        raise RuntimeError("No member/non-member candidates were selected")
    sample_frame["declared_iterations_completed"] = sample_frame[
        "iterations_completed"
    ].eq(int(args.iterations))
    if not np.isfinite(sample_frame["boundary_distance"].to_numpy(float)).all():
        raise RuntimeError("HSJ operational scores must be finite")
    if (sample_frame["total_label_queries"] > int(args.max_queries)).any():
        raise AssertionError("Per-record label-query budget was exceeded")

    score = sample_frame["boundary_distance"].to_numpy(float)
    y = sample_frame["membership"].to_numpy(int)
    auc = float(roc_auc_score(y, score))
    low, high, valid = stratified_bootstrap_auc(
        y,
        score,
        args.bootstrap,
        stable_seed(args.seed, args.target_id, "label_only_hsj_bootstrap"),
    )
    tpr5, attained5 = tpr_at_resolvable_fpr(y, score, 0.05)
    tpr10, attained10 = tpr_at_resolvable_fpr(y, score, 0.10)
    member_mask = sample_frame["membership"].eq(1)
    nonmember_mask = sample_frame["membership"].eq(0)
    initially_correct_mask = sample_frame["initially_correct"].astype(bool)
    initialized_correct_mask = (
        initially_correct_mask
        & sample_frame["adversarial_initialization_found"].astype(bool)
    )
    record = {
        "target_id": args.target_id,
        "experiment": row.get("experiment", ""),
        "dataset": row.get("dataset", ""),
        "architecture": row.get("architecture", ""),
        "role": row.get("role", ""),
        "seed": int(float(row.get("seed", row.get("model_seed", 0)))),
        "model_seed": int(float(row.get("model_seed", row.get("seed", 0)))),
        "data_seed": int(float(row.get("data_seed", 0))),
        "block_id": row.get("block_id", ""),
        "structural_cell_id": row.get("structural_cell_id", ""),
        "fm_kind": row.get("fm_kind", ""),
        "reps": int(float(row.get("reps", 0))),
        "depth": int(float(row.get("depth", 0))),
        "attack": "label_only_hsj_l2",
        "access": "predicted class labels only; candidate true labels; bounded random probes",
        "auc": auc,
        "auc_record_ci95_low": low,
        "auc_record_ci95_high": high,
        "valid_record_bootstrap_replicates": valid,
        "tpr_at_0_05_fpr": tpr5,
        "attained_fpr_for_0_05": attained5,
        "tpr_at_0_10_fpr": tpr10,
        "attained_fpr_for_0_10": attained10,
        "n_member": int((y == 1).sum()),
        "n_nonmember": int((y == 0).sum()),
        "max_label_queries_per_record": int(args.max_queries),
        "init_query_cap": int(args.init_queries),
        "init_batch_size": int(args.init_batch_size),
        "iterations": int(args.iterations),
        "gradient_samples_per_iteration": int(args.gradient_samples),
        "binary_steps": int(args.binary_steps),
        "step_search_steps": int(args.step_search_steps),
        "gradient_delta_ratio": float(args.gradient_delta_ratio),
        "min_gradient_delta": float(args.min_gradient_delta),
        "mean_label_queries": float(sample_frame["total_label_queries"].mean()),
        "median_label_queries": float(sample_frame["total_label_queries"].median()),
        "max_observed_label_queries": int(sample_frame["total_label_queries"].max()),
        "mean_member_label_queries": _fraction(
            sample_frame, member_mask, "total_label_queries"
        ),
        "mean_nonmember_label_queries": _fraction(
            sample_frame, nonmember_mask, "total_label_queries"
        ),
        "initial_accuracy": float(sample_frame["initially_correct"].mean()),
        "boundary_found_fraction_among_correct": _fraction(
            sample_frame,
            initially_correct_mask,
            "adversarial_initialization_found",
        ),
        "search_censored_fraction": float(sample_frame["search_censored"].mean()),
        "search_censored_member_fraction": _fraction(
            sample_frame, member_mask, "search_censored"
        ),
        "search_censored_nonmember_fraction": _fraction(
            sample_frame, nonmember_mask, "search_censored"
        ),
        "mean_iterations_completed_among_initialized_correct": _fraction(
            sample_frame, initialized_correct_mask, "iterations_completed"
        ),
        "declared_iterations_completed_fraction_among_initialized_correct": _fraction(
            sample_frame,
            initialized_correct_mask,
            "declared_iterations_completed",
        ),
        "query_budget_exhausted_fraction": float(
            sample_frame["query_budget_exhausted"].mean()
        ),
        "stopping_reason_histogram": json.dumps(
            sample_frame["stopping_reason"].value_counts().sort_index().to_dict(),
            sort_keys=True,
        ),
        "candidate_prediction_histogram": json.dumps(
            _histogram(selected_predictions), sort_keys=True
        ),
        "candidate_true_label_histogram": json.dumps(
            _histogram(samples.labels[evaluation_tensor]), sort_keys=True
        ),
        "validation_prediction_histogram": json.dumps(
            _histogram(validation_predictions), sort_keys=True
        ),
        "validation_true_label_histogram": json.dumps(
            _histogram(validation_labels), sort_keys=True
        ),
        "validation_prediction_support": int(validation_predictions.unique().numel()),
        "diagnostic_validation_label_queries": int(len(validation_predictions)),
        "clip_min": lower_scalar,
        "clip_max": upper_scalar,
        "bounds_source": bounds.source,
        "input_interface_scope": (
            "continuous train-preprocessed PCA feature interface; not a semantic "
            "raw-record perturbation certificate"
            if dataset_name in {"credit_default", "breast_cancer_wdbc"}
            else "normalized pixel interface with probes clipped to valid pixel-derived bounds"
        ),
        "norm": "l2",
        "protocol_version": PROTOCOL_VERSION,
        "query_seed_policy": (
            "common random numbers within data_seed and sample_id across matched structural targets"
        ),
        "score_definition": (
            "estimated L2 distance to an untargeted changed-label boundary under a "
            "fixed-budget hard-label HopSkipJump-style search; initially misclassified "
            "records receive zero; initialization failures receive a declared capped "
            "operational score and are marked search-censored"
        ),
        "scope_note": (
            "independent hard-label HSJ-style implementation; not a certified global "
            "minimum and not a bitwise reproduction of CleverHans"
        ),
        "threshold_calibration_scope": (
            "ROC-AUC and TPR-at-FPR are primary threshold-free evaluation summaries; "
            "cross-fitted threshold metrics are descriptive and not shadow-calibrated"
        ),
        "ci_method": CI_RECORD,
    }
    record.update(
        cross_fitted_threshold_metrics(
            y,
            score,
            5,
            stable_seed(args.seed, args.target_id, "label_only_hsj_crossfit"),
        )
    )
    sample_path = args.out_dir / "sample_scores" / f"{safe_name(args.target_id)}.csv"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(sample_frame, sample_path)
    record["sample_score_file"] = str(sample_path.resolve())
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(pd.DataFrame([record]), metrics_path)
    print(f"[OK] label-only HSJ -> {metrics_path.resolve()}")


def aggregate(args: argparse.Namespace) -> None:
    paths = sorted((args.out_dir / "target_scores").glob("*.csv"))
    if not paths:
        raise SystemExit(f"No target scores found under {args.out_dir / 'target_scores'}")
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    raw_path = args.out_dir / "label_only_hsj_raw.csv"
    atomic_write_csv(raw, raw_path)
    groups = [
        column
        for column in ("attack", "dataset", "architecture", "fm_kind", "reps", "depth")
        if column in raw.columns
    ]
    metrics = [
        column
        for column in (
            "auc",
            "tpr_at_0_05_fpr",
            "tpr_at_0_10_fpr",
            "balanced_accuracy_crossfit",
            "membership_advantage_crossfit",
            "mean_label_queries",
            "search_censored_fraction",
            "boundary_found_fraction_among_correct",
            "declared_iterations_completed_fraction_among_initialized_correct",
            "query_budget_exhausted_fraction",
        )
        if column in raw.columns
    ]
    summary = raw.groupby(groups, dropna=False)[metrics].agg(["count", "mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary["error_bar_type"] = "mean ± sample SD across target-model seeds/blocks"
    summary_path = args.out_dir / "label_only_hsj_summary.csv"
    atomic_write_csv(summary, summary_path)
    write_analysis_metadata(
        args.out_dir / "analysis_metadata.json",
        script=Path(__file__).name,
        inputs=[str(args.targets), str(args.run_root)],
        outputs=[str(raw_path), str(summary_path)],
        ci_method=CI_RECORD,
        bootstrap_unit="member/non-member records within target; target block for summary SD",
        bootstrap_replicates=args.bootstrap,
        error_bar_type="mean ± sample SD across target-model seeds/blocks",
        notes=(
            "Only predicted class labels are consumed by the attack. Every initially "
            "correct record receives the same nominal HSJ-style protocol and maximum "
            "query budget. Initialization failures are retained as finite capped "
            "operational scores and explicitly marked search-censored."
        ),
    )
    atomic_write_json(
        {
            "protocol_version": PROTOCOL_VERSION,
            "reference_paper": REFERENCE_PAPER,
            "reference_repository_studied": REFERENCE_REPOSITORY,
            "reference_commit": REFERENCE_COMMIT,
            "implementation_note": (
                "Independent PyTorch/TorchQuantum implementation; no source code was copied."
            ),
            "target_files": len(paths),
            "all_target_auc_values_finite": bool(
                np.isfinite(raw["auc"].to_numpy(float)).all()
            ),
        },
        args.out_dir / "reference_provenance.json",
    )
    print(f"[OK] raw={len(raw)} summary={len(summary)} -> {args.out_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--repo-root", type=Path, default=Path("."))
        subparser.add_argument("--targets", type=Path, required=True)
        subparser.add_argument("--run-root", type=Path, default=Path("reviewer_runs"))
        subparser.add_argument(
            "--out-dir",
            type=Path,
            default=Path("reviewer_results/label_only_hsj"),
        )
        subparser.add_argument("--bootstrap", type=int, default=5000)
        subparser.add_argument("--seed", type=int, default=2026)
        subparser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
        subparser.add_argument("--resume", action="store_true")

    score = subparsers.add_parser("score-target")
    common(score)
    score.add_argument("--target-id", required=True)
    score.add_argument("--n-member", type=int, default=200)
    score.add_argument("--n-nonmember", type=int, default=200)
    score.add_argument("--max-queries", type=int, default=512)
    score.add_argument("--init-queries", type=int, default=128)
    score.add_argument("--init-batch-size", type=int, default=32)
    score.add_argument("--iterations", type=int, default=8)
    score.add_argument("--gradient-samples", type=int, default=32)
    score.add_argument("--binary-steps", type=int, default=10)
    score.add_argument("--step-search-steps", type=int, default=10)
    score.add_argument("--gradient-delta-ratio", type=float, default=0.1)
    score.add_argument("--min-gradient-delta", type=float, default=1e-4)
    score.add_argument("--query-batch-size", type=int, default=64)
    score.add_argument("--clip-min", type=float, default=None)
    score.add_argument("--clip-max", type=float, default=None)
    score.set_defaults(function=score_target)

    collect = subparsers.add_parser("aggregate")
    common(collect)
    collect.set_defaults(function=aggregate)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.bootstrap < 1:
        raise SystemExit("--bootstrap must be positive")
    if hasattr(args, "max_queries"):
        positive = (
            "max_queries",
            "init_queries",
            "init_batch_size",
            "gradient_samples",
            "binary_steps",
            "step_search_steps",
            "query_batch_size",
        )
        for name in positive:
            if int(getattr(args, name)) < 1:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if int(args.iterations) < 0:
            raise SystemExit("--iterations cannot be negative")
        if int(args.max_queries) <= int(args.binary_steps) + 1:
            raise SystemExit("--max-queries must leave room for initialization and projection")
        if float(args.gradient_delta_ratio) <= 0 or float(args.min_gradient_delta) <= 0:
            raise SystemExit("gradient delta settings must be positive")
        if (args.clip_min is None) != (args.clip_max is None):
            raise SystemExit("--clip-min and --clip-max must be supplied together")
        if args.clip_min is not None and not float(args.clip_min) < float(args.clip_max):
            raise SystemExit("--clip-min must be smaller than --clip-max")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    args.function(args)


if __name__ == "__main__":
    main()
