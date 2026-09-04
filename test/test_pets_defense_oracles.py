from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from qurift.defenses.dynanoise import DynaNoiseOracle
from qurift.defenses.discriminator import MembershipDiscriminator
from qurift.defenses.guards import (
    LatticeRoundOracle,
    LogitGuardOracle,
    MeasurementGuardOracle,
    StickyInputOracle,
    project_expectation_lattice,
)
from qurift.defenses.hamp import (
    CalibrationSupportGenerator,
    HAMPOutputOracle,
    hamp_true_probability_from_gamma,
    hamp_training_loss,
    high_entropy_soft_labels,
)
from qurift.defenses.oracle import RawOracle


class ToyMeasuredModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        measured = torch.tanh(inputs)
        return F.log_softmax(self.linear(measured), dim=1)


class PETSOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(9)
        self.model = ToyMeasuredModel().eval()
        self.inputs = torch.randn(8, 3)

    def test_raw_oracle_is_exact_and_captures_measurement(self) -> None:
        direct = self.model(self.inputs)
        output = RawOracle(self.model).predict(self.inputs)
        self.assertTrue(torch.equal(output.model_output, direct))
        self.assertTrue(torch.equal(output.measurement, torch.tanh(self.inputs)))
        self.assertTrue(torch.allclose(output.probabilities, direct.exp()))

    def test_raw_oracle_supports_a_frozen_binary_decision_threshold(self) -> None:
        output = RawOracle(self.model, decision_threshold=0.2).predict(self.inputs)
        self.assertTrue(torch.equal(output.labels, (output.probabilities[:, 1] >= 0.2).long()))
        self.assertEqual(output.metadata["label_rule"], "binary_probability_threshold")

    def test_dynanoise_matches_declared_variance_formula(self) -> None:
        raw = RawOracle(self.model)
        output = DynaNoiseOracle(raw, seed=7).predict(self.inputs)
        expected = 0.3 * (1.0 + 2.0 * output.diagnostics["normalized_confidence"])
        self.assertTrue(torch.allclose(output.diagnostics["noise_variance"], expected))
        self.assertTrue(torch.allclose(output.probabilities.sum(1), torch.ones(8)))

    def test_hamp_output_preserves_labels_and_support(self) -> None:
        calibration = torch.linspace(-1, 1, 36).reshape(12, 3)
        generator = CalibrationSupportGenerator(
            calibration,
            lower=torch.full((3,), -1.0),
            upper=torch.full((3,), 1.0),
            seed=4,
        )
        sampled = generator.sample(20, device=torch.device("cpu"), dtype=torch.float32)
        self.assertTrue(bool((sampled >= -1).all() and (sampled <= 1).all()))
        raw = RawOracle(self.model).predict(self.inputs)
        defended = HAMPOutputOracle(RawOracle(self.model), generator).predict(self.inputs)
        self.assertTrue(torch.equal(raw.labels, defended.labels))

    def test_label_preserving_defenses_honor_the_deployment_threshold(self) -> None:
        threshold = 0.2
        raw_oracle = RawOracle(self.model, decision_threshold=threshold)
        raw = raw_oracle.predict(self.inputs)
        discriminator = MembershipDiscriminator(2, hidden_sizes=(8, 4)).eval()
        defended = LogitGuardOracle(
            raw_oracle, discriminator, iterations=3, learning_rate=0.01
        ).predict(self.inputs)
        self.assertTrue(torch.equal(defended.labels, raw.labels))
        self.assertEqual(defended.metadata["decision_threshold"], threshold)
        self.assertTrue(bool(defended.diagnostics["label_preserved"].all()))

        generator = CalibrationSupportGenerator(
            torch.linspace(-1, 1, 36).reshape(12, 3),
            lower=torch.full((3,), -1.0),
            upper=torch.full((3,), 1.0),
            seed=8,
        )
        hamp = HAMPOutputOracle(raw_oracle, generator).predict(self.inputs)
        self.assertTrue(torch.equal(hamp.labels, raw.labels))
        self.assertTrue(bool(hamp.diagnostics["label_preserved"].all()))

    def test_hamp_soft_targets_and_loss(self) -> None:
        labels = torch.tensor([0, 1, 1])
        targets = high_entropy_soft_labels(labels, 2, true_class_probability=0.7)
        self.assertTrue(torch.allclose(targets.sum(1), torch.ones(3)))
        loss, diagnostics = hamp_training_loss(
            torch.randn(3, 2),
            labels,
            true_class_probability=0.7,
            entropy_weight=0.1,
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIn("prediction_entropy", diagnostics)
        probability = hamp_true_probability_from_gamma(0.95, 2)
        entropy = -(probability * torch.log(torch.tensor(probability)) + (1 - probability) * torch.log(torch.tensor(1 - probability)))
        self.assertAlmostEqual(float(entropy / torch.log(torch.tensor(2.0))), 0.95, places=5)

    def test_expectation_lattice_is_exact(self) -> None:
        values = torch.tensor([[-1.0, -0.74, 0.01, 0.76, 1.0]])
        projected = project_expectation_lattice(values, shots=4)
        counts = (projected + 1.0) * 2.0
        self.assertTrue(torch.allclose(counts, counts.round()))
        self.assertTrue(bool((projected >= -1).all() and (projected <= 1).all()))

    def test_logit_and_measurement_guards_preserve_or_report_labels(self) -> None:
        discriminator = MembershipDiscriminator(2, hidden_sizes=(8, 4)).eval()
        raw_oracle = RawOracle(self.model)
        logit = LogitGuardOracle(
            raw_oracle, discriminator, iterations=3, learning_rate=0.01
        ).predict(self.inputs)
        self.assertTrue(bool(logit.diagnostics["label_preserved"].all()))

        measurement = MeasurementGuardOracle(
            raw_oracle,
            self.model.linear,
            discriminator,
            iterations=3,
            learning_rate=0.01,
            shots=16,
        ).predict(self.inputs)
        lattice_counts = (measurement.measurement + 1.0) * 8.0
        self.assertTrue(torch.allclose(lattice_counts, lattice_counts.round(), atol=1e-5))
        self.assertIn("search_censored", measurement.diagnostics)

        rounded = LatticeRoundOracle(
            raw_oracle, self.model.linear, shots=16
        ).predict(self.inputs)
        self.assertEqual(rounded.measurement.shape, self.inputs.shape)

    def test_sticky_canonicalization_coalesces_nearby_queries(self) -> None:
        sticky = StickyInputOracle(RawOracle(self.model), resolution=0.1, secret="unit-test")
        point = torch.tensor([[0.1234, -0.3123, 0.701]])
        canonical = sticky.canonicalize(point)
        nearby = canonical + 0.001
        self.assertTrue(torch.equal(sticky.canonicalize(canonical), sticky.canonicalize(nearby)))
        first = sticky.predict(canonical).probabilities
        second = sticky.predict(nearby).probabilities
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
