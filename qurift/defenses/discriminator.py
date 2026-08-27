"""Defense-side membership discriminator used by output sanitizers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F


class MembershipDiscriminator(torch.nn.Module):
    """Predict membership from sorted class probabilities.

    Sorting follows MemGuard's label-invariant defense classifier.  Membership
    labels in this PETS package always use 1=member and 0=non-member.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_sizes: Sequence[int] = (256, 128, 64),
    ) -> None:
        super().__init__()
        sizes = [int(num_classes), *(int(value) for value in hidden_sizes), 1]
        if min(sizes) <= 0:
            raise ValueError("all discriminator layer sizes must be positive")
        layers = []
        for index, (input_size, output_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(torch.nn.Linear(input_size, output_size))
            if index < len(sizes) - 2:
                layers.append(torch.nn.ReLU())
        self.network = torch.nn.Sequential(*layers)
        self.num_classes = int(num_classes)
        self.hidden_sizes = tuple(int(value) for value in hidden_sizes)

    @staticmethod
    def features(probabilities: torch.Tensor) -> torch.Tensor:
        return probabilities.sort(dim=1).values

    def forward(self, probabilities: torch.Tensor) -> torch.Tensor:
        return self.network(self.features(probabilities)).squeeze(1)


@dataclass(frozen=True)
class DiscriminatorFitConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    seed: int = 2026


def fit_membership_discriminator(
    probabilities: torch.Tensor,
    membership: torch.Tensor,
    *,
    hidden_sizes: Sequence[int] = (256, 128, 64),
    config: DiscriminatorFitConfig = DiscriminatorFitConfig(),
) -> Tuple[MembershipDiscriminator, Dict[str, object]]:
    """Fit only on a caller-supplied defense-calibration partition."""

    if probabilities.ndim != 2 or len(probabilities) != len(membership):
        raise ValueError("probabilities and membership have incompatible shapes")
    unique = set(int(value) for value in membership.detach().cpu().unique().tolist())
    if unique != {0, 1}:
        raise ValueError("membership calibration labels must contain both 0 and 1")
    if bool((probabilities < 0).any()) or not torch.allclose(
        probabilities.sum(1),
        torch.ones(len(probabilities), device=probabilities.device),
        atol=1e-5,
    ):
        raise ValueError("discriminator inputs must be normalized probabilities")

    torch.manual_seed(int(config.seed))
    device = probabilities.device
    model = MembershipDiscriminator(probabilities.shape[1], hidden_sizes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(int(config.seed))
    labels = membership.to(device=device, dtype=probabilities.dtype)
    detached = probabilities.detach()
    final_loss = float("nan")
    for _ in range(int(config.epochs)):
        order = torch.randperm(len(detached), generator=generator).to(device)
        for start in range(0, len(order), int(config.batch_size)):
            indices = order[start : start + int(config.batch_size)]
            logits = model(detached[indices])
            loss = F.binary_cross_entropy_with_logits(logits, labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        predictions = (model(detached) >= 0).long()
        accuracy = float((predictions == membership.long().to(device)).float().mean().item())
    metadata: Dict[str, object] = {
        "training_partition": "defense_calibration_only",
        "membership_encoding": "1=member,0=nonmember",
        "hidden_sizes": list(hidden_sizes),
        "fit_config": asdict(config),
        "final_minibatch_loss": final_loss,
        "calibration_accuracy": accuracy,
        "records": len(probabilities),
    }
    return model, metadata
