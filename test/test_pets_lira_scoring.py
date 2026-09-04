from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from scipy.special import log_ndtr

ROOT = Path(__file__).resolve().parents[1]
REVIEWER_TOOLS = ROOT / "reviewer_tools"
if str(REVIEWER_TOOLS) not in sys.path:
    sys.path.insert(0, str(REVIEWER_TOOLS))

from qurift.defenses.attacks import adaptive_threshold_metrics
from qurift_lira_attack import LIRA_ATTACK_NAMES, attack_scores, normal_logpdf


class OfflineLiRAScoreTests(unittest.TestCase):
    """Offline LiRA exposes both the paper and released-code definitions."""

    def setUp(self) -> None:
        candidates = 5
        self.distribution = {
            "mean_in": np.full(candidates, 1.0),
            "mean_out": np.zeros(candidates),
            "std_in": np.ones(candidates),
            "std_out": np.ones(candidates),
            "fixed_std_in": 1.0,
            "fixed_std_out": 1.0,
        }

    def test_offline_score_is_the_paper_one_sided_log_cdf(self) -> None:
        observed = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        scores = attack_scores(observed, self.distribution)
        expected_z = (
            observed - self.distribution["mean_out"]
        ) / self.distribution["std_out"]
        expected = log_ndtr(expected_z)
        np.testing.assert_allclose(scores["lira_offline"], expected)
        self.assertTrue(np.all(np.diff(scores["lira_offline"]) > 0))
        self.assertEqual(set(scores), LIRA_ATTACK_NAMES)

    def test_offline_score_is_one_sided_about_out_mean(self) -> None:
        low = attack_scores(np.array([-2.0]), self.distribution)["lira_offline"][0]
        high = attack_scores(np.array([2.0]), self.distribution)["lira_offline"][0]
        self.assertGreater(high, low)

    def test_fixed_variance_offline_uses_the_pooled_out_variance(self) -> None:
        observed = np.array([0.5, -0.5, 2.5, -2.5, 0.0])
        scores = attack_scores(observed, self.distribution)
        expected = log_ndtr(
            (observed - self.distribution["mean_out"])
            / self.distribution["fixed_std_out"]
        )
        np.testing.assert_allclose(scores["lira_offline_fixed_variance"], expected)

    def test_one_sided_variant_is_retained_under_an_explicit_name(self) -> None:
        observed = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        scores = attack_scores(observed, self.distribution)
        expected = (
            observed - self.distribution["mean_out"]
        ) / self.distribution["std_out"]
        np.testing.assert_allclose(scores["lira_offline_one_sided_z"], expected)
        self.assertTrue(np.all(np.diff(scores["lira_offline_one_sided_z"]) > 0))
        np.testing.assert_array_equal(
            np.argsort(scores["lira_offline"]),
            np.argsort(scores["lira_offline_one_sided_z"]),
        )

    def test_released_code_density_score_is_retained_under_an_explicit_name(self) -> None:
        observed = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        scores = attack_scores(observed, self.distribution)
        expected = -normal_logpdf(
            observed,
            self.distribution["mean_out"],
            self.distribution["std_out"],
        )
        np.testing.assert_allclose(
            scores["lira_offline_density_surprise"], expected
        )
        self.assertAlmostEqual(
            scores["lira_offline_density_surprise"][0],
            scores["lira_offline_density_surprise"][-1],
        )

    def test_online_score_remains_the_likelihood_ratio(self) -> None:
        observed = np.array([0.0, 1.0, 2.0])
        distribution = {key: value[:3] if isinstance(value, np.ndarray) else value
                        for key, value in self.distribution.items()}
        scores = attack_scores(observed, distribution)
        expected = normal_logpdf(
            observed, distribution["mean_in"], distribution["std_in"]
        ) - normal_logpdf(observed, distribution["mean_out"], distribution["std_out"])
        np.testing.assert_allclose(scores["lira_online"], expected)


class ScoreOrientationTests(unittest.TestCase):
    """A known-direction score must not have its sign relearned per block."""

    @staticmethod
    def metrics(calibration, calibration_labels, evaluation, labels, **kwargs):
        return adaptive_threshold_metrics(
            torch.tensor(calibration, dtype=torch.double),
            torch.tensor(calibration_labels, dtype=torch.long),
            torch.tensor(evaluation, dtype=torch.double),
            torch.tensor(labels, dtype=torch.long),
            **kwargs,
        )

    def setUp(self) -> None:
        # A calibration split whose sign is wrong by chance, over an evaluation
        # split where members genuinely score higher.
        self.calibration = [1.0, 0.9, 0.2, 0.1]
        self.calibration_labels = [0, 0, 1, 1]
        self.evaluation = [0.1, 0.2, 0.9, 1.0]
        self.labels = [0, 0, 1, 1]

    def test_calibrated_orientation_can_invert_a_correct_score(self) -> None:
        result = self.metrics(
            self.calibration, self.calibration_labels, self.evaluation, self.labels
        )
        self.assertEqual(result["score_direction"], "inverted_from_calibration")
        # The evaluation split is perfectly separable in the defined direction,
        # yet the reported AUC is its complement.
        self.assertAlmostEqual(result["auc"], 0.0)

    def test_fixed_orientation_keeps_the_defined_direction(self) -> None:
        result = self.metrics(
            self.calibration,
            self.calibration_labels,
            self.evaluation,
            self.labels,
            orientation="fixed",
        )
        self.assertEqual(result["score_direction"], "as_defined_fixed")
        self.assertAlmostEqual(result["auc"], 1.0)

    def test_fixed_and_calibrated_agree_when_calibration_is_right(self) -> None:
        calibration = [0.1, 0.2, 0.9, 1.0]
        fixed = self.metrics(
            calibration, self.calibration_labels, self.evaluation, self.labels,
            orientation="fixed",
        )
        calibrated = self.metrics(
            calibration, self.calibration_labels, self.evaluation, self.labels
        )
        self.assertAlmostEqual(fixed["auc"], calibrated["auc"])
        self.assertEqual(calibrated["score_direction"], "as_defined")

    def test_unknown_orientation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported score orientation"):
            self.metrics(
                self.calibration, self.calibration_labels, self.evaluation, self.labels,
                orientation="whatever",
            )


if __name__ == "__main__":
    unittest.main()
