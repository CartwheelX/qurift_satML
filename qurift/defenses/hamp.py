"""HAMP training and output components with disjoint calibration support."""
from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .base import PredictionBatch, PredictionOracle, defended_batch


def hamp_true_probability_from_gamma(gamma: float, num_classes: int) -> float:
    """Solve the HAMP entropy constraint for the soft-label true class."""

    if num_classes < 2 or not (0.0 <= gamma <= 1.0):
        raise ValueError("HAMP requires K>=2 and gamma in [0,1]")
    if gamma == 0:
        return 1.0
    if gamma == 1:
        return 1.0 / num_classes
    target = float(gamma) * math.log(float(num_classes))

    def entropy(probability: float) -> float:
        other = (1.0 - probability) / (num_classes - 1)
        values = [probability] + [other] * (num_classes - 1)
        return -sum(value * math.log(max(value, 1e-15)) for value in values)

    lower, upper = 1.0 / num_classes, 1.0
    for _ in range(80):
        middle = (lower + upper) / 2.0
        if entropy(middle) >= target:
            lower = middle
        else:
            upper = middle
    return float(lower)


def high_entropy_soft_labels(
    labels: torch.Tensor,
    num_classes: int,
    *,
    true_class_probability: float,
) -> torch.Tensor:
    """Return label-smoothed targets with a declared true-class probability."""

    if num_classes < 2:
        raise ValueError("HAMP requires at least two classes")
    if not (1.0 / num_classes <= true_class_probability <= 1.0):
        raise ValueError("true_class_probability must be in [1/K, 1]")
    other = (1.0 - float(true_class_probability)) / (num_classes - 1)
    targets = torch.full(
        (len(labels), num_classes), other, dtype=torch.float32, device=labels.device
    )
    targets.scatter_(1, labels.long().view(-1, 1), float(true_class_probability))
    return targets


def hamp_training_loss(
    model_output: torch.Tensor,
    labels: torch.Tensor,
    *,
    true_class_probability: float,
    entropy_weight: float,
) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """High-entropy soft-label loss plus an explicit entropy reward.

    ``model_output`` may be logits or normalized log-probabilities; applying
    log-softmax in either case is safe.
    """

    log_probabilities = F.log_softmax(model_output, dim=1)
    probabilities = log_probabilities.exp()
    targets = high_entropy_soft_labels(
        labels,
        probabilities.shape[1],
        true_class_probability=true_class_probability,
    ).to(probabilities.dtype)
    soft_cross_entropy = -(targets * log_probabilities).sum(dim=1).mean()
    entropy = -(probabilities * log_probabilities).sum(dim=1).mean()
    loss = soft_cross_entropy - float(entropy_weight) * entropy
    return loss, {
        "soft_cross_entropy": soft_cross_entropy.detach(),
        "prediction_entropy": entropy.detach(),
    }


class CalibrationSupportGenerator:
    """Generate random inputs using defense-calibration data only.

    The default empirical-marginal generator keeps every tabular coordinate
    inside its observed preprocessed support.  Explicit bounds are checked at
    construction and again after sampling.
    """

    def __init__(
        self,
        calibration_inputs: torch.Tensor,
        *,
        lower: Optional[torch.Tensor] = None,
        upper: Optional[torch.Tensor] = None,
        mode: str = "empirical_marginal",
        seed: int = 2026,
    ) -> None:
        if len(calibration_inputs) == 0:
            raise ValueError("calibration_inputs cannot be empty")
        if mode not in {"empirical_marginal", "empirical_rows", "uniform_box"}:
            raise ValueError(f"unsupported calibration generator mode={mode!r}")
        self.values = calibration_inputs.detach().cpu().float().clone()
        self.lower = (
            self.values.amin(dim=0) if lower is None else torch.as_tensor(lower).cpu().float()
        )
        self.upper = (
            self.values.amax(dim=0) if upper is None else torch.as_tensor(upper).cpu().float()
        )
        if self.lower.shape != self.values.shape[1:] or self.upper.shape != self.values.shape[1:]:
            raise ValueError("bounds must match one calibration input")
        if bool((self.lower > self.upper).any()):
            raise ValueError("lower bound exceeds upper bound")
        tolerance = 1e-6
        if bool((self.values < self.lower - tolerance).any()) or bool(
            (self.values > self.upper + tolerance).any()
        ):
            raise ValueError("calibration inputs lie outside declared support")
        self.mode = mode
        self.seed = int(seed)
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)
        digest = hashlib.sha256(self.values.contiguous().numpy().tobytes()).hexdigest()
        self.calibration_sha256 = digest

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "mode": self.mode,
            "seed": self.seed,
            "calibration_records": len(self.values),
            "calibration_sha256": self.calibration_sha256,
            "support_min": float(self.lower.min().item()),
            "support_max": float(self.upper.max().item()),
        }

    def sample(self, count: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        count = int(count)
        if count <= 0:
            raise ValueError("sample count must be positive")
        if self.mode == "empirical_rows":
            indices = torch.randint(len(self.values), (count,), generator=self.generator)
            result = self.values[indices]
        elif self.mode == "uniform_box":
            uniform = torch.rand(
                (count,) + tuple(self.values.shape[1:]), generator=self.generator
            )
            result = self.lower + uniform * (self.upper - self.lower)
        else:
            flat = self.values.reshape(len(self.values), -1)
            columns = []
            for column in range(flat.shape[1]):
                indices = torch.randint(len(flat), (count,), generator=self.generator)
                columns.append(flat[indices, column])
            result = torch.stack(columns, dim=1).reshape((count,) + self.values.shape[1:])
        if bool((result < self.lower - 1e-6).any()) or bool((result > self.upper + 1e-6).any()):
            raise RuntimeError("generated HAMP input escaped declared support")
        return result.to(device=device, dtype=dtype)


class HAMPOutputOracle(PredictionOracle):
    """Replace confidence values while preserving the raw class ordering."""

    def __init__(
        self,
        raw_oracle: PredictionOracle,
        generator: CalibrationSupportGenerator,
    ) -> None:
        self.raw_oracle = raw_oracle
        self.generator = generator

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "defense": "hamp_output",
            "generator": dict(self.generator.config),
        }

    def predict(
        self,
        inputs: torch.Tensor,
        *,
        query_ids: Optional[Sequence[str]] = None,
    ) -> PredictionBatch:
        raw = self.raw_oracle.predict(inputs, query_ids=query_ids)
        random_inputs = self.generator.sample(
            len(inputs), device=inputs.device, dtype=inputs.dtype
        )
        random_output = self.raw_oracle.predict(random_inputs)
        raw_order = raw.probabilities.argsort(dim=1, descending=True)
        random_sorted = random_output.probabilities.sort(dim=1, descending=True).values
        probabilities = torch.empty_like(raw.probabilities)
        probabilities.scatter_(1, raw_order, random_sorted)
        effective_logits = probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log()
        defended = defended_batch(
            raw,
            logits=effective_logits,
            probabilities=probabilities,
            diagnostics={
                "label_preserved": (probabilities.argmax(dim=1) == raw.labels).float()
            },
            metadata=dict(self.config),
        )
        if not torch.equal(defended.labels, raw.labels):
            raise RuntimeError("HAMP output reassignment failed to preserve predicted labels")
        return defended
