"""Task-utility metrics suitable for imbalanced defense evaluations."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def classification_utility_from_arrays(
    probabilities,
    labels,
    *,
    predictions=None,
) -> Dict[str, float]:
    """Return thresholded and threshold-independent classification utility."""

    probability = np.asarray(probabilities, dtype=float)
    truth = np.asarray(labels, dtype=int).reshape(-1)
    if probability.ndim != 2 or len(probability) != len(truth):
        raise ValueError("probabilities must be [records, classes] and align with labels")
    if probability.shape[1] < 2 or not np.isfinite(probability).all():
        raise ValueError("classification utility requires finite multi-class probabilities")
    predicted = (
        probability.argmax(axis=1)
        if predictions is None
        else np.asarray(predictions, dtype=int).reshape(-1)
    )
    if len(predicted) != len(truth):
        raise ValueError("predictions and labels do not align")
    classes, counts = np.unique(truth, return_counts=True)
    if len(classes) < 2:
        raise ValueError("classification utility requires at least two true classes")
    recalls = {
        int(class_id): float(np.mean(predicted[truth == class_id] == class_id))
        for class_id in classes
    }
    minority_class = int(classes[int(np.argmin(counts))])
    result: Dict[str, float] = {
        "accuracy": float(np.mean(predicted == truth)),
        "balanced_accuracy": float(np.mean(list(recalls.values()))),
        "minimum_class_recall": float(min(recalls.values())),
        "minority_class_label": float(minority_class),
        "minority_class_recall": recalls[minority_class],
        "predicted_minority_fraction": float(np.mean(predicted == minority_class)),
        "minority_prevalence": float(np.mean(truth == minority_class)),
        "prediction_collapse": float(len(np.unique(predicted)) < len(classes)),
    }
    for class_id in range(probability.shape[1]):
        result[f"predicted_class_{class_id}_fraction"] = float(
            np.mean(predicted == class_id)
        )
    try:
        if probability.shape[1] == 2:
            result["task_roc_auc"] = float(roc_auc_score(truth, probability[:, 1]))
            result["task_average_precision"] = float(
                average_precision_score(truth, probability[:, 1])
            )
        else:
            binary = label_binarize(truth, classes=np.arange(probability.shape[1]))
            result["task_roc_auc"] = float(
                roc_auc_score(binary, probability, average="macro", multi_class="ovr")
            )
            result["task_average_precision"] = float(
                average_precision_score(binary, probability, average="macro")
            )
    except ValueError:
        result["task_roc_auc"] = float("nan")
        result["task_average_precision"] = float("nan")
    return result


def binary_threshold_predictions(probabilities, threshold: float) -> np.ndarray:
    """Apply an explicit class-1 operating threshold to binary probabilities."""

    probability = np.asarray(probabilities, dtype=float)
    if probability.ndim != 2 or probability.shape[1] != 2:
        raise ValueError("binary thresholding requires probabilities with two columns")
    if not np.isfinite(probability).all() or not np.isfinite(float(threshold)):
        raise ValueError("binary thresholding requires finite inputs")
    return (probability[:, 1] >= float(threshold)).astype(np.int64)


def select_binary_decision_threshold(probabilities, labels) -> Dict[str, Any]:
    """Select a binary operating point from a labeled validation split.

    Balanced accuracy is the predeclared primary objective.  Exact ties are
    resolved by ordinary accuracy and then by proximity to the conventional
    0.5 threshold.  This keeps the rule deterministic without consulting the
    held-out test split or membership-attack outcomes.
    """

    probability = np.asarray(probabilities, dtype=float)
    truth = np.asarray(labels, dtype=int).reshape(-1)
    if probability.ndim != 2 or probability.shape != (len(truth), 2):
        raise ValueError("threshold selection requires aligned binary probabilities")
    if set(np.unique(truth).tolist()) != {0, 1}:
        raise ValueError("threshold selection requires both binary classes")
    scores = probability[:, 1]
    unique = np.unique(scores)
    if not len(unique):
        raise ValueError("threshold selection requires at least one score")
    if len(unique) == 1:
        candidates = np.array(
            [np.nextafter(unique[0], -np.inf), np.nextafter(unique[0], np.inf)]
        )
    else:
        candidates = np.concatenate(
            (
                [np.nextafter(unique[0], -np.inf)],
                (unique[:-1] + unique[1:]) / 2.0,
                [np.nextafter(unique[-1], np.inf)],
            )
        )
    candidates = np.unique(
        np.clip(
            candidates,
            np.nextafter(0.0, 1.0),
            np.nextafter(1.0, 0.0),
        )
    )
    rows = []
    for threshold in candidates:
        predicted = (scores >= threshold).astype(np.int64)
        rows.append(
            (
                float(balanced_accuracy_score(truth, predicted)),
                float(np.mean(predicted == truth)),
                -abs(float(threshold) - 0.5),
                float(threshold),
            )
        )
    balanced, accuracy, _, threshold = max(rows, key=lambda row: row[:3])
    predicted = binary_threshold_predictions(probability, threshold)
    return {
        "threshold": float(threshold),
        "validation_balanced_accuracy": float(balanced),
        "validation_accuracy": float(accuracy),
        "validation_predicted_class_1_fraction": float(np.mean(predicted == 1)),
        "selection_records": int(len(truth)),
        "selection_positive_records": int(np.sum(truth == 1)),
        "selection_objective": "maximum_balanced_accuracy",
        "tie_break": "maximum_accuracy_then_closest_to_0.5",
    }


def calibrated_binary_utility(probabilities, labels, threshold: float) -> Dict[str, float]:
    """Return utility fields prefixed for a frozen binary decision threshold."""

    metrics = classification_utility_from_arrays(
        probabilities,
        labels,
        predictions=binary_threshold_predictions(probabilities, threshold),
    )
    return {f"calibrated_{name}": value for name, value in metrics.items()}


def classification_utility(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    predictions: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    return classification_utility_from_arrays(
        probabilities.detach().cpu().numpy(),
        labels.detach().cpu().numpy(),
        predictions=(
            None if predictions is None else predictions.detach().cpu().numpy()
        ),
    )
