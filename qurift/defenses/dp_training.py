"""Auditable per-example DP-SGD for small QNN target models.

The quantum layers are custom modules, so this implementation computes exact
per-example gradients through ordinary autograd instead of assuming an Opacus
gradient hook supports them.  Privacy accounting is delegated to Opacus's RDP
accountant and is updated with the exact Poisson sampling rate used here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import secrets
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch


@dataclass(frozen=True)
class DPConfig:
    max_grad_norm: float
    noise_multiplier: float
    sample_rate: float
    delta: float
    expected_batch_size: int
    sampler: str = "poisson"
    accountant: str = "rdp"
    randomness_mode: str = "independent_os_entropy_seeded_pytorch_prng"

    def validate(self) -> "DPConfig":
        if self.max_grad_norm <= 0 or self.noise_multiplier <= 0:
            raise ValueError("max_grad_norm and noise_multiplier must be positive")
        if not (0 < self.sample_rate <= 1):
            raise ValueError("sample_rate must be in (0, 1]")
        if not (0 < self.delta < 1):
            raise ValueError("delta must be in (0, 1)")
        if self.expected_batch_size <= 0:
            raise ValueError("expected_batch_size must be positive")
        if self.sampler != "poisson" or self.accountant != "rdp":
            raise ValueError("formal DP-QML protocol requires poisson sampling and RDP accounting")
        return self


class PoissonIndexSampler:
    """Independent Bernoulli inclusion for every record at every step."""

    def __init__(
        self,
        population_size: int,
        sample_rate: float,
        *,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if population_size <= 0 or not (0 < sample_rate <= 1):
            raise ValueError("invalid Poisson sampler parameters")
        if seed is not None and generator is not None:
            raise ValueError("provide either a sampler seed or generator, not both")
        self.population_size = int(population_size)
        self.sample_rate = float(sample_rate)
        self.generator = generator or torch.Generator(device="cpu").manual_seed(
            int(secrets.randbits(63) if seed is None else seed)
        )

    def sample(self) -> torch.Tensor:
        mask = torch.rand(self.population_size, generator=self.generator) < self.sample_rate
        return mask.nonzero(as_tuple=False).flatten()


class OpacusRDPAccountant:
    """Small adapter that fails clearly when the audited dependency is absent."""

    def __init__(self) -> None:
        try:
            from opacus.accountants import RDPAccountant
            import opacus
        except ImportError as error:
            raise RuntimeError(
                "Formal epsilon accounting requires the PETS dependency `opacus==1.5.4` "
                "for this repository's PyTorch 2.5 environment. No epsilon claim was produced."
            ) from error
        self._accountant = RDPAccountant()
        self.version = str(opacus.__version__)
        self.steps = 0

    def step(self, *, noise_multiplier: float, sample_rate: float) -> None:
        self._accountant.step(
            noise_multiplier=float(noise_multiplier), sample_rate=float(sample_rate)
        )
        self.steps += 1

    def epsilon(self, delta: float) -> float:
        if self.steps <= 0:
            raise RuntimeError("privacy accountant has not observed a training step")
        return float(self._accountant.get_epsilon(delta=float(delta)))


def calibrate_noise_multiplier(
    *,
    target_epsilon: float,
    delta: float,
    sample_rate: float,
    steps: int,
    initial_noise_multiplier: Optional[float] = None,
    tolerance: float = 1e-4,
) -> tuple[float, float]:
    """Find a multiplier whose RDP ledger is at or below ``target_epsilon``.

    Calibration uses the exact integer number of accountant steps, avoiding the
    fractional-step rounding mismatch in epoch-based convenience helpers.
    """

    if target_epsilon <= 0 or steps <= 0 or tolerance <= 0:
        raise ValueError("invalid target-epsilon calibration parameters")
    try:
        from opacus.accountants import RDPAccountant
    except ImportError as error:
        raise RuntimeError("noise calibration requires opacus==1.5.4") from error

    def epsilon(noise: float) -> float:
        accountant = RDPAccountant()
        for _ in range(int(steps)):
            accountant.step(noise_multiplier=float(noise), sample_rate=float(sample_rate))
        return float(accountant.get_epsilon(delta=float(delta)))

    lower = 1e-4
    upper = max(float(initial_noise_multiplier or 1.0), lower * 2.0)
    while epsilon(upper) > target_epsilon:
        lower = upper
        upper *= 2.0
        if upper > 1e6:
            raise RuntimeError("could not bracket a DP noise multiplier")
    for _ in range(60):
        middle = (lower + upper) / 2.0
        if epsilon(middle) <= target_epsilon:
            upper = middle
        else:
            lower = middle
        if upper - lower <= tolerance * max(1.0, upper):
            break
    achieved = epsilon(upper)
    if achieved > target_epsilon + tolerance:
        raise RuntimeError("calibrated RDP epsilon exceeds requested target")
    return float(upper), float(achieved)


def _trainable_parameters(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return parameters


def per_example_gradients(
    model: torch.nn.Module,
    losses: torch.Tensor,
) -> List[List[torch.Tensor]]:
    """Compute one unaggregated gradient tuple per example."""

    if losses.ndim != 1:
        raise ValueError("DP loss function must return one scalar per example")
    parameters = _trainable_parameters(model)
    gradients: List[List[torch.Tensor]] = []
    for index in range(len(losses)):
        values = torch.autograd.grad(
            losses[index],
            parameters,
            retain_graph=index < len(losses) - 1,
            create_graph=False,
            allow_unused=True,
        )
        gradients.append(
            [torch.zeros_like(parameter) if value is None else value.detach() for parameter, value in zip(parameters, values)]
        )
    return gradients


def clip_and_aggregate(
    per_sample: Sequence[Sequence[torch.Tensor]],
    *,
    max_grad_norm: float,
) -> tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Clip each complete parameter-gradient vector before summation."""

    if not per_sample:
        raise ValueError("cannot aggregate an empty sample")
    squared_norms = []
    for gradients in per_sample:
        squared_norms.append(sum(gradient.double().square().sum() for gradient in gradients))
    norms = torch.stack(squared_norms).sqrt()
    factors = (float(max_grad_norm) / norms.clamp_min(1e-12)).clamp(max=1.0)
    aggregates = []
    for parameter_index in range(len(per_sample[0])):
        total = sum(
            gradients[parameter_index] * factors[index].to(gradients[parameter_index].dtype)
            for index, gradients in enumerate(per_sample)
        )
        aggregates.append(total)
    return aggregates, norms, factors


class DPTrainingSession:
    """Stateful DP optimizer/accountant pair for a QNN training run."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: DPConfig,
        *,
        accountant: Optional[Any] = None,
        noise_generator: Optional[torch.Generator] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config.validate()
        self.accountant = OpacusRDPAccountant() if accountant is None else accountant
        self.generator = noise_generator or torch.Generator(device="cpu").manual_seed(
            int(secrets.randbits(63))
        )
        self.steps = 0
        self.empty_steps = 0

    def _apply_noisy_update(self, aggregates: Sequence[torch.Tensor]) -> None:
        parameters = _trainable_parameters(self.model)
        if len(aggregates) != len(parameters):
            raise ValueError("gradient aggregate does not align with trainable parameters")
        self.optimizer.zero_grad(set_to_none=True)
        for parameter, aggregate in zip(parameters, aggregates):
            noise = torch.randn(
                aggregate.shape,
                generator=self.generator,
                dtype=aggregate.dtype,
                device="cpu",
            ).to(aggregate.device)
            noisy_sum = aggregate + noise * (
                self.config.noise_multiplier * self.config.max_grad_norm
            )
            parameter.grad = noisy_sum / float(self.config.expected_batch_size)
        self.optimizer.step()

    def step(self, losses: torch.Tensor) -> Mapping[str, float]:
        """Apply one clipped/noised update from per-example losses."""

        if losses.numel() == 0:
            self._apply_noisy_update(
                [torch.zeros_like(parameter) for parameter in _trainable_parameters(self.model)]
            )
            self.accountant.step(
                noise_multiplier=self.config.noise_multiplier,
                sample_rate=self.config.sample_rate,
            )
            self.steps += 1
            self.empty_steps += 1
            return {
                "batch_size": 0.0,
                "mean_grad_norm": 0.0,
                "mean_clip_factor": 0.0,
                "clipped_fraction": 0.0,
                "noise_applied": 1.0,
            }

        gradients = per_example_gradients(self.model, losses)
        aggregates, norms, factors = clip_and_aggregate(
            gradients, max_grad_norm=self.config.max_grad_norm
        )
        self._apply_noisy_update(aggregates)
        self.accountant.step(
            noise_multiplier=self.config.noise_multiplier,
            sample_rate=self.config.sample_rate,
        )
        self.steps += 1
        return {
            "batch_size": float(len(losses)),
            "mean_grad_norm": float(norms.mean().item()),
            "max_grad_norm_before_clipping": float(norms.max().item()),
            "mean_clip_factor": float(factors.mean().item()),
            "clipped_fraction": float((factors < 1).float().mean().item()),
            "noise_applied": 1.0,
        }

    def privacy_report(self) -> Dict[str, Any]:
        epsilon = float(self.accountant.epsilon(self.config.delta))
        if not torch.isfinite(torch.tensor(epsilon)):
            raise RuntimeError("accountant returned a non-finite epsilon; no DP claim is valid")
        return {
            "formal_dp_claim": True,
            "epsilon": epsilon,
            "delta": self.config.delta,
            "steps": self.steps,
            "empty_steps_accounted": self.empty_steps,
            "sampler": self.config.sampler,
            "accountant": self.config.accountant,
            "accountant_version": getattr(self.accountant, "version", "test-double"),
            "neighbor_relation": "add_or_remove_one_training_record",
            "gradient_clipping": "per_example_global_l2",
            "noise_location": "summed_clipped_gradients_before_expected_batch_normalization",
            "empty_step_behavior": "gaussian_update_to_zero_clipped_sum",
            "randomness_streams": "independent_sampler_and_noise_generators",
            "cryptographic_rng": False,
            "config": asdict(self.config),
        }
