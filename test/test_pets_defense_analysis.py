from __future__ import annotations

import unittest

import pandas as pd
import numpy as np

from pets_tools.analyze_defenses import (
    difference_in_differences,
    effective_defense,
    paired_structural_effects,
)
from qurift.defenses.utility import classification_utility_from_arrays


class PETSDefenseAnalysisTests(unittest.TestCase):
    def test_imbalanced_utility_reports_ranking_and_threshold_metrics(self) -> None:
        truth = np.array([0, 0, 0, 1])
        probability = np.array(
            [[0.9, 0.1], [0.8, 0.2], [0.6, 0.4], [0.55, 0.45]]
        )
        utility = classification_utility_from_arrays(probability, truth)
        self.assertEqual(utility["prediction_collapse"], 1.0)
        self.assertEqual(utility["minority_class_recall"], 0.0)
        self.assertGreater(utility["task_roc_auc"], 0.5)
        self.assertGreater(utility["task_average_precision"], 0.25)

    def test_effective_defense_names_training_and_output_composition(self) -> None:
        self.assertEqual(effective_defense("none", "dynanoise"), "dynanoise")
        self.assertEqual(effective_defense("l2", "none"), "l2")
        self.assertEqual(effective_defense("hamp_train", "hamp_output"), "hamp_full")

    def test_paired_high_low_and_difference_in_differences(self) -> None:
        rows = []
        for block in ("b1", "b2"):
            for defense, low, high in (("none", 0.55, 0.65), ("guard", 0.53, 0.57)):
                rows.extend(
                    [
                        {"block_id": block, "structural_role": "low", "effective_defense": defense, "attack": "loss", "auc": low},
                        {"block_id": block, "structural_role": "high", "effective_defense": defense, "attack": "loss", "auc": high},
                    ]
                )
        frame = pd.DataFrame(rows)
        paired = paired_structural_effects(
            frame,
            group_columns=["effective_defense", "attack"],
            outcome="auc",
            draws=100,
            seed=1,
        )
        guard = paired[paired.effective_defense.eq("guard")].iloc[0]
        self.assertAlmostEqual(guard.mean_difference, 0.04)
        did = difference_in_differences(
            frame,
            group_columns=["effective_defense", "attack"],
            outcome="auc",
            baseline="none",
            draws=100,
            seed=1,
        )
        self.assertAlmostEqual(did.iloc[0].mean_difference_in_differences, -0.06)
        utility = frame.drop(columns="attack").rename(columns={"auc": "accuracy"})
        utility_did = difference_in_differences(
            utility,
            group_columns=["effective_defense"],
            outcome="accuracy",
            baseline="none",
            draws=100,
            seed=1,
        )
        self.assertAlmostEqual(
            utility_did.iloc[0].mean_difference_in_differences, -0.06
        )


if __name__ == "__main__":
    unittest.main()
