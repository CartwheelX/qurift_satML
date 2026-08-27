from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from qurift.defenses.dp_training import (
    DPConfig,
    DPTrainingSession,
    PoissonIndexSampler,
    calibrate_noise_multiplier,
    clip_and_aggregate,
)


class RecordingAccountant:
    version = "unit-test"

    def __init__(self) -> None:
        self.calls = []

    def step(self, *, noise_multiplier: float, sample_rate: float) -> None:
        self.calls.append((noise_multiplier, sample_rate))

    def epsilon(self, delta: float) -> float:
        return len(self.calls) / 10.0 + delta


class PETSDPTests(unittest.TestCase):
    def test_global_per_example_clipping_precedes_aggregation(self) -> None:
        per_sample = [
            [torch.tensor([3.0, 4.0]), torch.tensor([0.0])],
            [torch.tensor([0.0, 0.0]), torch.tensor([0.5])],
        ]
        aggregate, norms, factors = clip_and_aggregate(per_sample, max_grad_norm=1.0)
        self.assertTrue(torch.allclose(norms, torch.tensor([5.0, 0.5], dtype=torch.float64)))
        self.assertTrue(torch.allclose(factors, torch.tensor([0.2, 1.0], dtype=torch.float64)))
        self.assertTrue(torch.allclose(aggregate[0], torch.tensor([0.6, 0.8])))
        self.assertTrue(torch.allclose(aggregate[1], torch.tensor([0.5])))

    def test_session_uses_matching_sampling_rate_and_reports_epsilon(self) -> None:
        torch.manual_seed(1)
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        accountant = RecordingAccountant()
        config = DPConfig(
            max_grad_norm=1.0,
            noise_multiplier=1.2,
            sample_rate=0.25,
            delta=1e-5,
            expected_batch_size=2,
        )
        inputs = torch.randn(3, 2)
        labels = torch.tensor([0, 1, 0])
        losses = F.cross_entropy(model(inputs), labels, reduction="none")
        session = DPTrainingSession(model, optimizer, config, accountant=accountant)
        diagnostics = session.step(losses)
        self.assertEqual(accountant.calls, [(1.2, 0.25)])
        self.assertEqual(diagnostics["batch_size"], 3.0)
        report = session.privacy_report()
        self.assertTrue(report["formal_dp_claim"])
        self.assertEqual(report["gradient_clipping"], "per_example_global_l2")

    def test_poisson_sampler_is_seed_reproducible(self) -> None:
        first = PoissonIndexSampler(100, 0.2, seed=4)
        second = PoissonIndexSampler(100, 0.2, seed=4)
        for _ in range(4):
            self.assertTrue(torch.equal(first.sample(), second.sample()))

    def test_empty_poisson_step_still_applies_noise_and_is_accounted(self) -> None:
        model = torch.nn.Linear(2, 2)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        accountant = RecordingAccountant()
        generator = torch.Generator(device="cpu").manual_seed(91)
        session = DPTrainingSession(
            model,
            optimizer,
            DPConfig(
                max_grad_norm=1.0,
                noise_multiplier=1.0,
                sample_rate=0.25,
                delta=1e-5,
                expected_batch_size=2,
            ),
            accountant=accountant,
            noise_generator=generator,
        )
        diagnostics = session.step(torch.empty(0))
        self.assertEqual(accountant.calls, [(1.0, 0.25)])
        self.assertEqual(diagnostics["batch_size"], 0.0)
        self.assertEqual(diagnostics["noise_applied"], 1.0)
        self.assertTrue(
            any(
                not torch.equal(old, new)
                for old, new in zip(before, model.parameters())
            )
        )
        self.assertEqual(session.privacy_report()["empty_steps_accounted"], 1)

    def test_non_poisson_claim_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DPConfig(
                max_grad_norm=1,
                noise_multiplier=1,
                sample_rate=0.1,
                delta=1e-5,
                expected_batch_size=4,
                sampler="shuffle",
            ).validate()

    def test_exact_step_noise_calibration_respects_target_epsilon(self) -> None:
        noise, epsilon = calibrate_noise_multiplier(
            target_epsilon=4.0,
            delta=1e-5,
            sample_rate=0.08,
            steps=13,
            initial_noise_multiplier=0.88,
        )
        self.assertGreater(noise, 0.88)
        self.assertLessEqual(epsilon, 4.0)


if __name__ == "__main__":
    unittest.main()
