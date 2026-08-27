"""Matched logit and measurement-domain controls, including MemGQ."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .base import PredictionBatch, PredictionOracle, defended_batch
from .discriminator import MembershipDiscriminator


def project_expectation_lattice(values: torch.Tensor, shots: int) -> torch.Tensor:
    """Project Pauli expectation values onto {-1 + 2k/shots}."""

    shots = int(shots)
    if shots <= 0:
        raise ValueError("shots must be positive")
    clipped = values.clamp(-1.0, 1.0)
    counts = torch.round((clipped + 1.0) * shots / 2.0).clamp(0, shots)
    return -1.0 + 2.0 * counts / shots


def quantize_logits(logits: torch.Tensor, step: float) -> torch.Tensor:
    if step <= 0:
        raise ValueError("logit quantization step must be positive")
    # Centering removes the softmax-invariant common offset before quantizing.
    centered = logits - logits.mean(dim=1, keepdim=True)
    return torch.round(centered / float(step)) * float(step)


def _membership_objective(
    probabilities: torch.Tensor,
    initial_probabilities: torch.Tensor,
    initial_labels: torch.Tensor,
    discriminator: MembershipDiscriminator,
    *,
    distortion_weight: float,
    label_weight: float,
    label_margin: float,
) -> torch.Tensor:
    privacy = discriminator(probabilities).abs()
    selected = probabilities.gather(1, initial_labels[:, None]).squeeze(1)
    mask = F.one_hot(initial_labels, probabilities.shape[1]).bool()
    alternatives = probabilities.masked_fill(mask, -torch.inf).max(dim=1).values
    hinge = torch.relu(alternatives - selected + float(label_margin))
    distortion = (probabilities - initial_probabilities).abs().sum(dim=1)
    return (privacy + float(distortion_weight) * distortion + float(label_weight) * hinge).mean()


class LogitGuardOracle(PredictionOracle):
    """Continuous logit sanitizer, optionally followed by matched quantization."""

    def __init__(
        self,
        raw_oracle: PredictionOracle,
        discriminator: MembershipDiscriminator,
        *,
        iterations: int = 100,
        learning_rate: float = 0.05,
        distortion_weight: float = 0.1,
        label_weight: float = 100.0,
        label_margin: float = 1e-5,
        quantization_step: Optional[float] = None,
    ) -> None:
        self.raw_oracle = raw_oracle
        self.discriminator = discriminator.eval()
        self.iterations = int(iterations)
        self.learning_rate = float(learning_rate)
        self.distortion_weight = float(distortion_weight)
        self.label_weight = float(label_weight)
        self.label_margin = float(label_margin)
        self.quantization_step = None if quantization_step is None else float(quantization_step)
        for parameter in self.discriminator.parameters():
            parameter.requires_grad_(False)

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "defense": "logitguard_quantized" if self.quantization_step else "logitguard_continuous",
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "distortion_weight": self.distortion_weight,
            "label_weight": self.label_weight,
            "label_margin": self.label_margin,
            "quantization_step": self.quantization_step,
        }

    def predict(self, inputs: torch.Tensor, *, query_ids=None) -> PredictionBatch:
        raw = self.raw_oracle.predict(inputs, query_ids=query_ids)
        candidate = raw.logits.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([candidate], lr=self.learning_rate)
        for _ in range(self.iterations):
            probabilities = torch.softmax(candidate, dim=1)
            loss = _membership_objective(
                probabilities,
                raw.probabilities,
                raw.labels,
                self.discriminator,
                distortion_weight=self.distortion_weight,
                label_weight=self.label_weight,
                label_margin=self.label_margin,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        logits = candidate.detach()
        if self.quantization_step is not None:
            logits = quantize_logits(logits, self.quantization_step)
        probabilities = torch.softmax(logits, dim=1)
        preserves = probabilities.argmax(1) == raw.labels
        logits = torch.where(preserves[:, None], logits, raw.logits)
        probabilities = torch.softmax(logits, dim=1)
        return defended_batch(
            raw,
            logits=logits,
            probabilities=probabilities,
            diagnostics={
                "label_preserved": preserves.float(),
                "l1_probability_distortion": (probabilities - raw.probabilities).abs().sum(1),
                "initial_membership_abs_logit": self.discriminator(raw.probabilities).abs(),
                "defended_membership_abs_logit": self.discriminator(probabilities).abs(),
            },
            metadata=dict(self.config),
        )


class MeasurementGuardOracle(PredictionOracle):
    """Optimize post-PQC expectation values through the frozen classical head."""

    def __init__(
        self,
        raw_oracle: PredictionOracle,
        head: torch.nn.Module,
        discriminator: MembershipDiscriminator,
        *,
        iterations: int = 100,
        learning_rate: float = 0.05,
        distortion_weight: float = 0.1,
        label_weight: float = 100.0,
        label_margin: float = 1e-5,
        shots: Optional[int] = None,
    ) -> None:
        self.raw_oracle = raw_oracle
        self.head = head.eval()
        self.discriminator = discriminator.eval()
        self.iterations = int(iterations)
        self.learning_rate = float(learning_rate)
        self.distortion_weight = float(distortion_weight)
        self.label_weight = float(label_weight)
        self.label_margin = float(label_margin)
        self.shots = None if shots is None else int(shots)
        for module in (self.head, self.discriminator):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    @property
    def config(self) -> Mapping[str, Any]:
        if self.shots is None:
            name = "measurementguard_continuous"
        else:
            name = "memgq_lattice"
        return {
            "defense": name,
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "distortion_weight": self.distortion_weight,
            "label_weight": self.label_weight,
            "label_margin": self.label_margin,
            "shots": self.shots,
            "measurement_bounds": [-1.0, 1.0],
        }

    def _project(self, values: torch.Tensor) -> torch.Tensor:
        values = values.clamp(-1.0, 1.0)
        return values if self.shots is None else project_expectation_lattice(values, self.shots)

    def predict(self, inputs: torch.Tensor, *, query_ids=None) -> PredictionBatch:
        raw = self.raw_oracle.predict(inputs, query_ids=query_ids)
        if raw.measurement is None:
            raise RuntimeError("measurement defense requires RawOracle measurement capture")
        initial = self._project(raw.measurement.detach())
        candidate = initial.clone()
        best = initial.clone()
        best_score = torch.full(
            (len(initial),), torch.inf, device=initial.device, dtype=initial.dtype
        )
        ever_valid = torch.zeros(len(initial), dtype=torch.bool, device=initial.device)

        for _ in range(self.iterations):
            continuous = candidate.detach().requires_grad_(True)
            logits = self.head(continuous)
            probabilities = torch.softmax(logits, dim=1)
            loss = _membership_objective(
                probabilities,
                raw.probabilities,
                raw.labels,
                self.discriminator,
                distortion_weight=self.distortion_weight,
                label_weight=self.label_weight,
                label_margin=self.label_margin,
            )
            gradient = torch.autograd.grad(loss, continuous)[0]
            candidate = self._project(continuous - self.learning_rate * gradient.sign()).detach()
            with torch.no_grad():
                candidate_probabilities = torch.softmax(self.head(candidate), dim=1)
                score = self.discriminator(candidate_probabilities).abs()
                valid = candidate_probabilities.argmax(1) == raw.labels
                improve = valid & (score < best_score)
                best[improve] = candidate[improve]
                best_score[improve] = score[improve]
                ever_valid |= valid

        # Starting values are a transparent lattice fallback.  For the
        # continuous control, raw measurements always form a valid fallback.
        with torch.no_grad():
            initial_probabilities = torch.softmax(self.head(initial), dim=1)
            initial_valid = initial_probabilities.argmax(1) == raw.labels
            use_initial = ~ever_valid & initial_valid
            best[use_initial] = initial[use_initial]
            ever_valid |= initial_valid
            logits = self.head(best)
            probabilities = torch.softmax(logits, dim=1)

        return defended_batch(
            raw,
            logits=logits,
            probabilities=probabilities,
            measurement=best,
            diagnostics={
                "label_preserved": (probabilities.argmax(1) == raw.labels).float(),
                "optimization_valid": ever_valid.float(),
                "search_censored": (~ever_valid).float(),
                "l1_probability_distortion": (probabilities - raw.probabilities).abs().sum(1),
                "l2_measurement_distortion": (best - raw.measurement).square().sum(1).sqrt(),
                "initial_membership_abs_logit": self.discriminator(raw.probabilities).abs(),
                "defended_membership_abs_logit": self.discriminator(probabilities).abs(),
            },
            metadata=dict(self.config),
        )


class LatticeRoundOracle(PredictionOracle):
    """Finite-shot lattice control without membership-aware optimization."""

    def __init__(self, raw_oracle: PredictionOracle, head: torch.nn.Module, *, shots: int) -> None:
        self.raw_oracle = raw_oracle
        self.head = head.eval()
        self.shots = int(shots)

    @property
    def config(self) -> Mapping[str, Any]:
        return {"defense": "lattice_round", "shots": self.shots}

    def predict(self, inputs: torch.Tensor, *, query_ids=None) -> PredictionBatch:
        raw = self.raw_oracle.predict(inputs, query_ids=query_ids)
        if raw.measurement is None:
            raise RuntimeError("lattice control requires measurement capture")
        measurement = project_expectation_lattice(raw.measurement, self.shots)
        with torch.no_grad():
            logits = self.head(measurement)
            probabilities = torch.softmax(logits, dim=1)
        return defended_batch(
            raw,
            logits=logits,
            probabilities=probabilities,
            measurement=measurement,
            diagnostics={
                "label_preserved": (probabilities.argmax(1) == raw.labels).float(),
                "l2_measurement_distortion": (
                    measurement - raw.measurement
                ).square().sum(1).sqrt(),
            },
            metadata=dict(self.config),
        )


class StickyInputOracle(PredictionOracle):
    """Nearby-query hardening ablation using secret-shifted input buckets."""

    def __init__(
        self,
        oracle: PredictionOracle,
        *,
        resolution: float,
        secret: str,
    ) -> None:
        if resolution <= 0 or not secret:
            raise ValueError("sticky resolution and secret must be non-empty/positive")
        self.oracle = oracle
        self.resolution = float(resolution)
        self._secret_digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()
        # A secret offset prevents an attacker from knowing bucket boundaries.
        integer = int(self._secret_digest[:16], 16)
        self._offset_fraction = (integer % 1_000_003) / 1_000_003.0

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "defense": "sticky_input_ablation",
            "wrapped": dict(self.oracle.config),
            "resolution": self.resolution,
            "secret_sha256": self._secret_digest,
            "secret_stored": False,
        }

    def canonicalize(self, inputs: torch.Tensor) -> torch.Tensor:
        offset = self._offset_fraction * self.resolution
        return torch.round((inputs + offset) / self.resolution) * self.resolution - offset

    def predict(self, inputs: torch.Tensor, *, query_ids=None) -> PredictionBatch:
        canonical = self.canonicalize(inputs)
        result = self.oracle.predict(canonical, query_ids=query_ids)
        result.metadata.update(self.config)
        result.diagnostics["input_linf_canonicalization"] = (
            canonical - inputs
        ).abs().reshape(len(inputs), -1).max(1).values
        return result
