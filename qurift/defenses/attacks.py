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


def tpr_at_fpr(membership: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(membership, scores)
    eligible = np.flatnonzero(fpr <= float(target_fpr) + 1e-12)
    return float(tpr[eligible].max()) if len(eligible) else 0.0


def adaptive_threshold_metrics(
    calibration_scores: torch.Tensor,
    calibration_membership: torch.Tensor,
    evaluation_scores: torch.Tensor,
    evaluation_membership: torch.Tensor,
) -> Dict[str, Any]:
    calibration = calibration_scores.detach().cpu().double().numpy()
    calibration_labels = calibration_membership.detach().cpu().long().numpy()
    evaluation = evaluation_scores.detach().cpu().double().numpy()
    labels = evaluation_membership.detach().cpu().long().numpy()
    if set(np.unique(calibration_labels)) != {0, 1} or set(np.unique(labels)) != {0, 1}:
        raise ValueError("adaptive attack calibration and evaluation need both membership classes")
    calibration_auc = float(roc_auc_score(calibration_labels, calibration))
    direction = 1.0 if calibration_auc >= 0.5 else -1.0
    calibration = direction * calibration
    evaluation = direction * evaluation
    threshold = _best_threshold(calibration, calibration_labels)
    predictions = (evaluation >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(labels, evaluation)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "tpr_at_5_fpr": tpr_at_fpr(labels, evaluation, 0.05),
        "tpr_at_10_fpr": tpr_at_fpr(labels, evaluation, 0.10),
        "threshold": threshold,
        "score_direction": "as_defined" if direction > 0 else "inverted_from_calibration",
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
        "tpr_at_5_fpr": tpr_at_fpr(labels, scores, 0.05),
        "tpr_at_10_fpr": tpr_at_fpr(labels, scores, 0.10),
        "threshold": float(threshold),
        "score_direction": "artifact_fixed",
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
        "tpr_at_5_fpr": tpr_at_fpr(y_test, scores, 0.05),
        "tpr_at_10_fpr": tpr_at_fpr(y_test, scores, 0.10),
        "threshold": float(threshold),
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
        "tpr_at_5_fpr": tpr_at_fpr(y_test, scores, 0.05),
        "tpr_at_10_fpr": tpr_at_fpr(y_test, scores, 0.10),
        "threshold": float(threshold),
    }
    return metrics, attacker
