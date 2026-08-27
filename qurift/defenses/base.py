"""Shared, fail-closed interfaces for prediction defenses."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import torch


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
        label_rule = str(self.metadata.get("label_rule", "argmax"))
        if label_rule == "binary_probability_threshold":
            if shape[1] != 2:
                raise ValueError("binary threshold labels require exactly two classes")
            threshold = float(self.metadata.get("decision_threshold", float("nan")))
            if not torch.isfinite(torch.tensor(threshold)):
                raise ValueError("binary threshold labels require a finite threshold")
            expected_labels = (self.probabilities[:, 1] >= threshold).long()
        elif label_rule == "argmax":
            expected_labels = self.probabilities.argmax(dim=1)
        else:
            raise ValueError(f"unknown prediction label rule {label_rule!r}")
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
    result = PredictionBatch(
        model_output=log_probabilities,
        logits=logits,
        log_probabilities=log_probabilities,
        probabilities=probabilities,
        labels=probabilities.argmax(dim=1),
        measurement=raw.measurement if measurement is None else measurement,
        diagnostics={} if diagnostics is None else diagnostics,
        metadata={
            **raw.metadata,
            "label_rule": "argmax",
            **({} if metadata is None else metadata),
        },
    )
    return result.validate()
