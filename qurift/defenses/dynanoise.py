"""DynaNoise output defense adapted from the official MIT artifact.

Reference artifact revision: 27c6ba5664eb3ba28973d44e2ea4830d15fd3ee5.
The implementation here is native PyTorch and uses the published algorithm:
confidence-dependent Gaussian logit noise followed by temperature scaling.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch

from .base import PredictionBatch, PredictionOracle, defended_batch


class DynaNoiseOracle(PredictionOracle):
    def __init__(
        self,
        raw_oracle: PredictionOracle,
        *,
        base_variance: float = 0.3,
        confidence_lambda: float = 2.0,
        temperature: float = 10.0,
        ensemble_size: int = 1,
        seed: int = 2026,
    ) -> None:
        if base_variance < 0 or confidence_lambda < 0:
            raise ValueError("noise variance parameters must be non-negative")
        if temperature <= 0 or ensemble_size <= 0:
            raise ValueError("temperature and ensemble_size must be positive")
        self.raw_oracle = raw_oracle
        self.base_variance = float(base_variance)
        self.confidence_lambda = float(confidence_lambda)
        self.temperature = float(temperature)
        self.ensemble_size = int(ensemble_size)
        self.seed = int(seed)
        self._generator = torch.Generator(device="cpu").manual_seed(self.seed)

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "defense": "dynanoise",
            "base_variance": self.base_variance,
            "confidence_lambda": self.confidence_lambda,
            "temperature": self.temperature,
            "ensemble_size": self.ensemble_size,
            "seed": self.seed,
            "source_revision": "27c6ba5664eb3ba28973d44e2ea4830d15fd3ee5",
            "source_license": "MIT",
        }

    def predict(
        self,
        inputs: torch.Tensor,
        *,
        query_ids: Optional[Sequence[str]] = None,
    ) -> PredictionBatch:
        raw = self.raw_oracle.predict(inputs, query_ids=query_ids)
        classes = int(raw.probabilities.shape[1])
        entropy = -(raw.probabilities * raw.log_probabilities).sum(dim=1)
        normalized_confidence = 1.0 - entropy / torch.log(
            torch.tensor(float(classes), device=entropy.device, dtype=entropy.dtype)
        )
        variance = self.base_variance * (
            1.0 + self.confidence_lambda * normalized_confidence.clamp(0.0, 1.0)
        )
        members = []
        for _ in range(self.ensemble_size):
            noise = torch.randn(
                raw.logits.shape,
                generator=self._generator,
                dtype=raw.logits.dtype,
                device="cpu",
            ).to(raw.logits.device)
            noisy_logits = raw.logits + noise * variance.sqrt().unsqueeze(1)
            members.append(torch.softmax(noisy_logits / self.temperature, dim=1))
        probabilities = torch.stack(members, dim=0).mean(dim=0)
        effective_logits = probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log()
        return defended_batch(
            raw,
            logits=effective_logits,
            probabilities=probabilities,
            diagnostics={
                "normalized_confidence": normalized_confidence.detach(),
                "noise_variance": variance.detach(),
            },
            metadata=dict(self.config),
        )
