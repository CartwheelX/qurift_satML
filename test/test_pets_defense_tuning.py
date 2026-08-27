from __future__ import annotations

import unittest

import pandas as pd

from pets_tools.build_defense_tuning_targets import build_tuning_targets
from pets_tools.select_defense_tuning import (
    choose_settings,
    eligible_settings,
    freeze_confirmatory_manifest,
)


class PETSDefenseTuningTests(unittest.TestCase):
    def test_grid_is_isolated_to_development_block(self) -> None:
        base = pd.DataFrame(
            [
                {
                    "target_id": f"target_{block}_{role}",
                    "block_id": block,
                    "structural_cell_id": role,
                    "defense_structural_role": role,
                    "weight_decay": 0.0,
                }
                for block in ("pets_b01", "pets_b02")
                for role in ("low", "high")
            ]
        )
        targets = build_tuning_targets(
            base,
            block_id="pets_b01",
            l2_weight_decays=[1e-3, 1e-4],
            dp_epsilons=[8.0, 16.0],
        )
        self.assertEqual(len(targets), 8)
        self.assertEqual(set(targets.block_id), {"pets_b01"})
        self.assertEqual(set(targets.tuning_family), {"l2", "dp_qml"})
        self.assertFalse(targets.target_id.duplicated().any())

    def test_selection_uses_predeclared_utility_constraints(self) -> None:
        rows = []
        for family, values in (("l2", (1e-3, 1e-4)), ("dp_qml", (8.0, 16.0))):
            for value in values:
                for role in ("low", "high"):
                    eligible = value in {1e-4, 16.0}
                    rows.append(
                        {
                            "tuning_family": family,
                            "tuning_value": value,
                            "structural_role": role,
                            "task_roc_auc": 0.75 if eligible else 0.60,
                            "task_average_precision": 0.40,
                            "minority_class_recall": 0.10,
                            "prediction_collapse": 0.0,
                            "calibrated_balanced_accuracy": 0.70,
                            "calibrated_minority_class_recall": 0.20,
                            "calibrated_prediction_collapse": 0.0,
                        }
                    )
        summary = eligible_settings(
            pd.DataFrame(rows),
            minimum_roc_auc=0.65,
            minimum_average_precision=0.30,
            minimum_minority_recall=0.02,
            minimum_balanced_accuracy=0.55,
        )
        self.assertEqual(
            choose_settings(summary), {"l2": 1e-4, "dp_qml": 16.0}
        )

    def test_confirmatory_manifest_excludes_development_block(self) -> None:
        rows = []
        for block in ("pets_b01", "pets_b02", "pets_b03"):
            for cell in ("low", "high"):
                for defense in ("none", "l2", "hamp_train", "dp_qml"):
                    rows.append(
                        {
                            "target_id": f"{block}_{cell}_{defense}",
                            "block_id": block,
                            "structural_cell_id": cell,
                            "training_defense": defense,
                            "weight_decay": 0.01,
                            "dp_target_epsilon": 4.0,
                        }
                    )
        frozen = freeze_confirmatory_manifest(
            pd.DataFrame(rows),
            chosen={"l2": 1e-4, "dp_qml": 16.0},
            development_block="pets_b01",
        )
        self.assertEqual(len(frozen), 16)
        self.assertEqual(set(frozen.block_id), {"pets_b02", "pets_b03"})
        self.assertEqual(
            set(frozen.loc[frozen.training_defense.eq("l2"), "weight_decay"]),
            {1e-4},
        )
        self.assertEqual(
            set(
                frozen.loc[
                    frozen.training_defense.eq("dp_qml"), "dp_target_epsilon"
                ]
            ),
            {16.0},
        )


if __name__ == "__main__":
    unittest.main()
