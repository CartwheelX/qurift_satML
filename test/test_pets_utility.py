from __future__ import annotations

import unittest

import numpy as np

from qurift.defenses.utility import (
    binary_threshold_predictions,
    calibrated_binary_utility,
    select_binary_decision_threshold,
)


class PETSUtilityTests(unittest.TestCase):
    def test_validation_threshold_recovers_ranked_but_sub_half_scores(self) -> None:
        probabilities = np.array(
            [
                [0.98, 0.02],
                [0.90, 0.10],
                [0.75, 0.25],
                [0.60, 0.40],
            ]
        )
        labels = np.array([0, 0, 1, 1])
        rule = select_binary_decision_threshold(probabilities, labels)
        self.assertLess(rule["threshold"], 0.5)
        predicted = binary_threshold_predictions(probabilities, rule["threshold"])
        self.assertTrue(np.array_equal(predicted, labels))
        metrics = calibrated_binary_utility(probabilities, labels, rule["threshold"])
        self.assertEqual(metrics["calibrated_balanced_accuracy"], 1.0)
        self.assertEqual(metrics["calibrated_prediction_collapse"], 0.0)

    def test_threshold_selection_requires_both_classes(self) -> None:
        with self.assertRaises(ValueError):
            select_binary_decision_threshold(
                np.array([[0.8, 0.2], [0.7, 0.3]]), np.array([0, 0])
            )


if __name__ == "__main__":
    unittest.main()
