"""Adaptive attacks and metrics over a common PredictionOracle."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .base import PredictionBatch


def attack_signals(output: PredictionBatch, true_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    labels = true_labels.to(output.probabilities.device).long()
    true_probability = output.probabilities.gather(1, labels[:, None]).squeeze(1)
    loss = F.nll_loss(output.log_probabilities, labels, reduction="none")
    entropy = -(output.probabilities * output.log_probabilities).sum(dim=1)
    top_two = output.probabilities.topk(min(2, output.probabilities.shape[1]), dim=1).values
    if output.probabilities.shape[1] == 1:
        margin = top_two[:, 0]
    else:
        margin = top_two[:, 0] - top_two[:, 1]
    return {
        "loss": -loss,
        "confidence": true_probability,
        "maximum_probability": output.probabilities.max(dim=1).values,
        "entropy": -entropy,
        "margin": margin,
        "correctness": (output.labels == labels).float(),
    }


def _best_threshold(scores: np.ndarray, membership: np.ndarray) -> float:
    candidates = np.unique(scores)
    if len(candidates) == 1:
        return float(candidates[0])
    midpoints = (candidates[:-1] + candidates[1:]) / 2.0
    thresholds = np.concatenate(([-np.inf], midpoints, [np.inf]))
    accuracies = [
        balanced_accuracy_score(membership, (scores >= threshold).astype(int))
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(accuracies))])


def operating_point_at_fpr(
    membership: np.ndarray, scores: np.ndarray, target_fpr: float
) -> Tuple[float, float]:
    """Return best attainable TPR at or below a requested FPR and its FPR.

    Reporting the attained point matters when the non-member pool is small.  For
    example, with 100 non-members, 0.1% FPR is not empirically resolvable except
    as the zero-false-positive operating point.
    """

    fpr, tpr, _ = roc_curve(membership, scores)
    eligible = np.flatnonzero(fpr <= float(target_fpr) + 1e-12)
    if not len(eligible):
        return 0.0, 0.0
    eligible_tpr = tpr[eligible]
    best_tpr = float(eligible_tpr.max())
    tied = eligible[np.isclose(eligible_tpr, best_tpr)]
    chosen = tied[int(np.argmax(fpr[tied]))]
    return best_tpr, float(fpr[chosen])


def tpr_at_fpr(membership: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    return operating_point_at_fpr(membership, scores, target_fpr)[0]


def _roc_operating_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    nonmembers = int((np.asarray(labels) == 0).sum())
    result: Dict[str, Any] = {
        "n_evaluation_nonmember": nonmembers,
        "empirical_fpr_resolution": 1.0 / nonmembers if nonmembers else float("nan"),
    }
    for label, target in (("0_1", 0.001), ("1", 0.01), ("5", 0.05), ("10", 0.10)):
        tpr, attained = operating_point_at_fpr(labels, scores, target)
        result[f"tpr_at_{label}_fpr"] = tpr
        result[f"attained_fpr_at_{label}_fpr"] = attained
        result[f"target_{label}_fpr_resolvable"] = bool(
            nonmembers and target >= 1.0 / nonmembers - 1e-12
        )
    return result


def adaptive_threshold_metrics(
    calibration_scores: torch.Tensor,
    calibration_membership: torch.Tensor,
    evaluation_scores: torch.Tensor,
    evaluation_membership: torch.Tensor,
    *,
    orientation: str = "calibrated",
) -> Dict[str, Any]:
    """Fit a membership threshold on calibration data and score the evaluation set.

    ``orientation`` controls whether the sign of the score is learned:

    ``calibrated``
        Learn it from the calibration split. Correct for heuristic signals whose
        polarity is genuinely unknown ahead of time.
    ``fixed``
        Trust the score as defined. Required for likelihood-ratio attacks, where
        higher always means "more likely a member" by construction. Learning a
        known sign from a small calibration split flips it whenever noise wins,
        and a flipped block reports ``1 - AUC``; averaging such blocks drags the
        pooled estimate toward and below chance instead of measuring leakage.
    """

    if orientation not in {"calibrated", "fixed"}:
        raise ValueError(f"unsupported score orientation {orientation!r}")
    calibration = calibration_scores.detach().cpu().double().numpy()
    calibration_labels = calibration_membership.detach().cpu().long().numpy()
    evaluation = evaluation_scores.detach().cpu().double().numpy()
    labels = evaluation_membership.detach().cpu().long().numpy()
    if set(np.unique(calibration_labels)) != {0, 1} or set(np.unique(labels)) != {0, 1}:
        raise ValueError("adaptive attack calibration and evaluation need both membership classes")
    if orientation == "fixed":
        direction = 1.0
    else:
        calibration_auc = float(roc_auc_score(calibration_labels, calibration))
        direction = 1.0 if calibration_auc >= 0.5 else -1.0
    calibration = direction * calibration
    evaluation = direction * evaluation
    threshold = _best_threshold(calibration, calibration_labels)
    predictions = (evaluation >= threshold).astype(int)
    if orientation == "fixed":
        score_direction = "as_defined_fixed"
    else:
        score_direction = "as_defined" if direction > 0 else "inverted_from_calibration"
    return {
        "auc": float(roc_auc_score(labels, evaluation)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "threshold": threshold,
        "score_direction": score_direction,
        **_roc_operating_metrics(labels, evaluation),
    }


def fixed_threshold_metrics(
    evaluation_scores: torch.Tensor,
    evaluation_membership: torch.Tensor,
    *,
    threshold: float,
) -> Dict[str, Any]:
    """Evaluate a predeclared greater-than threshold without adaptation."""

    scores = evaluation_scores.detach().cpu().double().numpy()
    labels = evaluation_membership.detach().cpu().long().numpy()
    predictions = (scores >= float(threshold)).astype(int)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "threshold": float(threshold),
        "score_direction": "artifact_fixed",
        **_roc_operating_metrics(labels, scores),
    }


def learned_features(output: PredictionBatch, true_labels: torch.Tensor) -> np.ndarray:
    signals = attack_signals(output, true_labels)
    matrix = torch.cat(
        [
            output.probabilities,
            signals["loss"][:, None],
            signals["entropy"][:, None],
            signals["margin"][:, None],
            signals["correctness"][:, None],
        ],
        dim=1,
    )
    return matrix.detach().cpu().numpy()


def adaptive_learned_metrics(
    calibration_output: PredictionBatch,
    calibration_labels: torch.Tensor,
    calibration_membership: torch.Tensor,
    evaluation_output: PredictionBatch,
    evaluation_labels: torch.Tensor,
    evaluation_membership: torch.Tensor,
    *,
    seed: int,
) -> Tuple[Dict[str, float], Any]:
    x_train = learned_features(calibration_output, calibration_labels)
    y_train = calibration_membership.detach().cpu().numpy().astype(int)
    x_test = learned_features(evaluation_output, evaluation_labels)
    y_test = evaluation_membership.detach().cpu().numpy().astype(int)
    attacker = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=int(seed),
        ),
    )
    attacker.fit(x_train, y_train)
    scores = attacker.predict_proba(x_test)[:, 1]
    threshold = _best_threshold(attacker.predict_proba(x_train)[:, 1], y_train)
    predictions = (scores >= threshold).astype(int)
    metrics = {
        "auc": float(roc_auc_score(y_test, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "threshold": float(threshold),
        **_roc_operating_metrics(y_test, scores),
    }
    return metrics, attacker


def adaptive_feature_attack_metrics(
    calibration_features: np.ndarray,
    calibration_membership: torch.Tensor,
    evaluation_features: np.ndarray,
    evaluation_membership: torch.Tensor,
    *,
    seed: int,
) -> Tuple[Dict[str, float], Any]:
    """Fit a fresh logistic attacker to arbitrary defended-query features."""

    x_train = np.asarray(calibration_features, dtype=float)
    x_test = np.asarray(evaluation_features, dtype=float)
    y_train = calibration_membership.detach().cpu().numpy().astype(int)
    y_test = evaluation_membership.detach().cpu().numpy().astype(int)
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("adaptive attack features contain non-finite values")
    attacker = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=int(seed),
        ),
    )
    attacker.fit(x_train, y_train)
    calibration_scores = attacker.predict_proba(x_train)[:, 1]
    scores = attacker.predict_proba(x_test)[:, 1]
    threshold = _best_threshold(calibration_scores, y_train)
    metrics = {
        "auc": float(roc_auc_score(y_test, scores)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, (scores >= threshold).astype(int))
        ),
        "threshold": float(threshold),
        **_roc_operating_metrics(y_test, scores),
    }
    return metrics, attacker
