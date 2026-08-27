from __future__ import annotations

import unittest

import pandas as pd

from pets_tools.build_defense_targets import build_targets


class PETSTargetBuilderTests(unittest.TestCase):
    def test_fresh_high_low_targets_are_paired(self) -> None:
        source = pd.DataFrame(
            [
                {"structural_cell_id": "low", "data_seed": 1, "model_seed": 2, "target_id": "a"},
                {"structural_cell_id": "high", "data_seed": 1, "model_seed": 2, "target_id": "b"},
            ]
        )
        result = build_targets(
            source,
            low_cell="low",
            high_cell="high",
            blocks=3,
            data_seed_start=100,
            model_seed_start=200,
        )
        self.assertEqual(len(result), 6)
        for _, group in result.groupby("block_id"):
            self.assertEqual(set(group.defense_structural_role), {"low", "high"})
            self.assertEqual(group.data_seed.nunique(), 1)
            self.assertEqual(group.model_seed.nunique(), 1)

    def test_seed_reuse_is_rejected(self) -> None:
        source = pd.DataFrame(
            [
                {"structural_cell_id": "low", "data_seed": 100, "model_seed": 200},
                {"structural_cell_id": "high", "data_seed": 100, "model_seed": 200},
            ]
        )
        with self.assertRaises(ValueError):
            build_targets(
                source,
                low_cell="low",
                high_cell="high",
                blocks=1,
                data_seed_start=100,
                model_seed_start=300,
            )


if __name__ == "__main__":
    unittest.main()
