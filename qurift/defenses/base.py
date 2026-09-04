"""Shared, fail-closed interfaces for prediction defenses."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F


ARGMAX_LABEL_RULE = "argmax"
BINARY_THRESHOLD_LABEL_RULE = "binary_probability_threshold"


def labels_from_probabilities(
    probabilities: torch.Tensor,
    metadata: Mapping[str, Any],
) -> torch.Tensor:
    """Apply the prediction batch's declared deployment decision rule.

    PETS binary targets freeze a class-1 probability threshold on validation
    data.  Keeping this operation in one place prevents output defenses and
    hard-label attacks from silently falling back to argmax/0.5 semantics.
    """

    if probabilities.ndim != 2:
        raise ValueError("prediction labels require [batch, classes] probabilities")
    label_rule = str(metadata.get("label_rule", ARGMAX_LABEL_RULE))
    if label_rule == BINARY_THRESHOLD_LABEL_RULE:
        if probabilities.shape[1] != 2:
            raise ValueError("binary threshold labels require exactly two classes")
        threshold = float(metadata.get("decision_threshold", float("nan")))
        if not torch.isfinite(torch.tensor(threshold)) or not 0.0 < threshold < 1.0:
            raise ValueError(
                "binary threshold labels require a finite threshold strictly between zero and one"
            )
        return (probabilities[:, 1] >= threshold).long()
    if label_rule == ARGMAX_LABEL_RULE:
        return probabilities.argmax(dim=1)
    raise ValueError(f"unknown prediction label rule {label_rule!r}")


def label_preservation_mask(
    probabilities: torch.Tensor,
    raw: "PredictionBatch",
) -> torch.Tensor:
    """Return whether probabilities preserve each raw deployed label."""

    return labels_from_probabilities(probabilities, raw.metadata) == raw.labels


def label_preservation_hinge(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    metadata: Mapping[str, Any],
    *,
    margin: float,
) -> torch.Tensor:
    """Differentiable violation of the declared label-preservation rule."""

    label_rule = str(metadata.get("label_rule", ARGMAX_LABEL_RULE))
    labels = labels.long()
    if label_rule == BINARY_THRESHOLD_LABEL_RULE:
        if probabilities.shape[1] != 2:
            raise ValueError("binary threshold labels require exactly two classes")
        threshold = float(metadata.get("decision_threshold", float("nan")))
        if not torch.isfinite(torch.tensor(threshold)) or not 0.0 < threshold < 1.0:
            raise ValueError(
                "binary threshold labels require a finite threshold strictly between zero and one"
            )
        positive_violation = torch.relu(
            float(threshold) + float(margin) - probabilities[:, 1]
        )
        negative_violation = torch.relu(
            probabilities[:, 1] - float(threshold) + float(margin)
        )
        return torch.where(labels == 1, positive_violation, negative_violation)
    if label_rule == ARGMAX_LABEL_RULE:
        selected = probabilities.gather(1, labels[:, None]).squeeze(1)
        mask = F.one_hot(labels, probabilities.shape[1]).bool()
        alternatives = probabilities.masked_fill(mask, -torch.inf).max(dim=1).values
        return torch.relu(alternatives - selected + float(margin))
    raise ValueError(f"unknown prediction label rule {label_rule!r}")


@dataclass
class PredictionBatch:
    """Outputs exposed by a prediction oracle for one input batch.

    ``model_output`` is the exact tensor returned by the wrapped model for a
    raw oracle, or the defended log-probability tensor for a defended oracle.
    ``measurement`` is optional because classical controls need not expose a
    quantum measurement vector.
    """

    model_output: torch.Tensor
    logits: torch.Tensor
    log_probabilities: torch.Tensor
    probabilities: torch.Tensor
    labels: torch.Tensor
    measurement: Optional[torch.Tensor] = None
    diagnostics: Dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self, *, atol: float = 1e-5) -> "PredictionBatch":
        tensors = {
            "model_output": self.model_output,
            "logits": self.logits,
            "log_probabilities": self.log_probabilities,
            "probabilities": self.probabilities,
        }
        shape = self.probabilities.shape
        if len(shape) != 2:
            raise ValueError(f"probabilities must have shape [batch, classes], got {shape}")
        for name, tensor in tensors.items():
            if tensor.shape != shape:
                raise ValueError(f"{name} has shape {tensor.shape}, expected {shape}")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} contains a non-finite value")
        if self.labels.shape != (shape[0],):
            raise ValueError(f"labels must have shape [{shape[0]}], got {self.labels.shape}")
        if self.measurement is not None and self.measurement.shape[0] != shape[0]:
            raise ValueError("measurement batch dimension does not match probabilities")
        sums = self.probabilities.sum(dim=1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=atol, rtol=atol):
            raise ValueError("probabilities do not sum to one")
        if bool((self.probabilities < -atol).any()):
            raise ValueError("probabilities contain negative values")
        expected_labels = labels_from_probabilities(self.probabilities, self.metadata)
        if not torch.equal(self.labels, expected_labels):
            raise ValueError("labels are inconsistent with the declared probability rule")
        return self


class PredictionOracle(ABC):
    """The only prediction interface used by PETS defense evaluations."""

    @abstractmethod
    def predict(
        self,
        inputs: torch.Tensor,
        *,
        query_ids: Optional[Sequence[str]] = None,
    ) -> PredictionBatch:
        """Return predictions for ``inputs``.

        ``query_ids`` are optional stable record identifiers.  Sticky defenses
        may use them to make repeated identical queries deterministic; they
        must never use membership labels or evaluation split names.
        """

    @property
    @abstractmethod
    def config(self) -> Mapping[str, Any]:
        """Serializable defense configuration and provenance."""

    def predict_proba(
        self, inputs: torch.Tensor, *, query_ids: Optional[Sequence[str]] = None
    ) -> torch.Tensor:
        return self.predict(inputs, query_ids=query_ids).probabilities

    def predict_label(
        self, inputs: torch.Tensor, *, query_ids: Optional[Sequence[str]] = None
    ) -> torch.Tensor:
        return self.predict(inputs, query_ids=query_ids).labels


def defended_batch(
    raw: PredictionBatch,
    *,
    logits: torch.Tensor,
    probabilities: Optional[torch.Tensor] = None,
    measurement: Optional[torch.Tensor] = None,
    diagnostics: Optional[Dict[str, torch.Tensor]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> PredictionBatch:
    """Construct and validate a defended prediction batch."""

    if probabilities is None:
        probabilities = torch.softmax(logits, dim=1)
    probabilities = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
    log_probabilities = probabilities.log()
    result_metadata = {**raw.metadata, **({} if metadata is None else metadata)}
    for key in ("label_rule", "decision_threshold"):
        if key in raw.metadata and result_metadata.get(key) != raw.metadata[key]:
            raise ValueError(f"a defense cannot replace the deployed {key}")
    result = PredictionBatch(
        model_output=log_probabilities,
        logits=logits,
        log_probabilities=log_probabilities,
        probabilities=probabilities,
        labels=labels_from_probabilities(probabilities, result_metadata),
        measurement=raw.measurement if measurement is None else measurement,
        diagnostics={} if diagnostics is None else diagnostics,
        metadata=result_metadata,
    )
    return result.validate()
