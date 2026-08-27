"""Clean-room PyTorch implementation of the MemGuard paper objective.

No source from the legacy Python-2/Keras repository is copied.  The reference
repository has no declared license, so this module implements only the method
described in the paper: optimize logits toward the defense classifier's
decision boundary subject to label preservation and small L1 distortion.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from .base import PredictionBatch, PredictionOracle, defended_batch
from .discriminator import MembershipDiscriminator


def _label_hinge(probabilities: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    selected = probabilities.gather(1, labels[:, None]).squeeze(1)
    mask = F.one_hot(labels, probabilities.shape[1]).bool()
    alternatives = probabilities.masked_fill(mask, -torch.inf).max(dim=1).values
    return torch.relu(alternatives - selected + float(margin))


class MemGuardOracle(PredictionOracle):
    """Optimize each prediction independently with an official-style c search."""

    def __init__(
        self,
        raw_oracle: PredictionOracle,
        discriminator: MembershipDiscriminator,
        *,
        max_iterations: int = 300,
        step_size: float = 0.1,
        distortion_weights: Sequence[float] = (0.1, 1.0, 10.0, 100.0, 1000.0),
        label_weight: float = 1000.0,
        label_margin: float = 1e-6,
        score_tolerance: float = 0.05,
    ) -> None:
        self.raw_oracle = raw_oracle
        self.discriminator = discriminator.eval()
        self.max_iterations = int(max_iterations)
        self.step_size = float(step_size)
        self.distortion_weights = tuple(float(value) for value in distortion_weights)
        self.label_weight = float(label_weight)
        self.label_margin = float(label_margin)
        self.score_tolerance = float(score_tolerance)
        if self.max_iterations <= 0 or self.step_size <= 0 or not self.distortion_weights:
            raise ValueError("invalid MemGuard optimization settings")
        for parameter in self.discriminator.parameters():
            parameter.requires_grad_(False)

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "defense": "memguard_clean_room",
            "implementation": "paper_objective_native_pytorch",
            "reference_revision": "34e1859",
            "reference_code_copied": False,
            "reference_repository_license": "not_declared",
            "max_iterations": self.max_iterations,
            "step_size": self.step_size,
            "distortion_weights": list(self.distortion_weights),
            "label_weight": self.label_weight,
            "label_margin": self.label_margin,
            "score_tolerance": self.score_tolerance,
        }

    def _sanitize_one(self, initial_logits: torch.Tensor, label: torch.Tensor):
        initial_probabilities = torch.softmax(initial_logits.detach(), dim=0)
        initial_score = self.discriminator(initial_probabilities[None])[0].abs().detach()
        best_logits = initial_logits.detach().clone()
        best_distortion = torch.tensor(float("inf"), device=initial_logits.device)
        best_score = initial_score
        best_iterations = 0
        converged = False

        for distortion_weight in self.distortion_weights:
            candidate = initial_logits.detach().clone().requires_grad_(True)
            for iteration in range(1, self.max_iterations + 1):
                probabilities = torch.softmax(candidate, dim=0)
                attack_score = self.discriminator(probabilities[None])[0].abs()
                distortion = (probabilities - initial_probabilities).abs().sum()
                hinge = _label_hinge(probabilities[None], label[None], self.label_margin)[0]
                objective = (
                    attack_score
                    + self.label_weight * hinge
                    + distortion_weight * distortion
                )
                gradient = torch.autograd.grad(objective, candidate)[0]
                norm = gradient.norm().clamp_min(1e-12)
                candidate = (candidate - self.step_size * gradient / norm).detach().requires_grad_(True)
                with torch.no_grad():
                    current_probabilities = torch.softmax(candidate, dim=0)
                    preserves = int(current_probabilities.argmax()) == int(label)
                    current_score = self.discriminator(current_probabilities[None])[0].abs()
                    current_distortion = (
                        current_probabilities - initial_probabilities
                    ).abs().sum()
                    acceptable = preserves and current_score <= self.score_tolerance
                    if acceptable and current_distortion < best_distortion:
                        best_logits = candidate.detach().clone()
                        best_distortion = current_distortion.detach()
                        best_score = current_score.detach()
                        best_iterations = iteration
                        converged = True
            # Higher distortion penalties cannot improve a found minimum enough
            # to justify another full sweep in the pilot protocol.
            if converged:
                break

        if not converged:
            best_distortion = torch.tensor(0.0, device=initial_logits.device)
        return best_logits, best_score, best_distortion, best_iterations, converged, initial_score

    def predict(
        self,
        inputs: torch.Tensor,
        *,
        query_ids: Optional[Sequence[str]] = None,
    ) -> PredictionBatch:
        raw = self.raw_oracle.predict(inputs, query_ids=query_ids)
        sanitized = []
        scores = []
        initial_scores = []
        distortions = []
        iterations = []
        converged = []
        for logits, label in zip(raw.logits, raw.labels):
            result = self._sanitize_one(logits, label)
            sanitized.append(result[0])
            scores.append(result[1])
            distortions.append(result[2])
            iterations.append(result[3])
            converged.append(result[4])
            initial_scores.append(result[5])
        defended_logits = torch.stack(sanitized)
        probabilities = torch.softmax(defended_logits, dim=1)
        result = defended_batch(
            raw,
            logits=defended_logits,
            probabilities=probabilities,
            diagnostics={
                "initial_membership_abs_logit": torch.stack(initial_scores),
                "defended_membership_abs_logit": torch.stack(scores),
                "l1_probability_distortion": torch.stack(distortions),
                "iterations": torch.tensor(iterations, device=raw.logits.device),
                "converged": torch.tensor(converged, device=raw.logits.device).float(),
                "label_preserved": (probabilities.argmax(1) == raw.labels).float(),
            },
            metadata=dict(self.config),
        )
        if not torch.equal(result.labels, raw.labels):
            raise RuntimeError("MemGuard returned a label-changing prediction")
        return result
