from __future__ import annotations

import unittest

import pandas as pd

from pets_tools.build_defense_training_variants import expand_variants


class PETSTrainingVariantTests(unittest.TestCase):
    def test_expansion_has_four_distinct_training_conditions(self) -> None:
        source = pd.DataFrame(
            [{"target_id": "a", "weight_decay": 0.0}, {"target_id": "b", "weight_decay": 0.0}]
        )
        result = expand_variants(
            source,
            l2_weight_decay=0.01,
            hamp_gamma=0.95,
            hamp_alpha=0.001,
            dp_target_epsilon=4.0,
            dp_max_grad_norm=1.0,
            dp_delta=1e-5,
        )
        self.assertEqual(len(result), 8)
        self.assertEqual(set(result.training_defense), {"none", "l2", "hamp_train", "dp_qml"})
        l2 = result[result.training_defense.eq("l2")]
        self.assertTrue(l2.weight_decay.eq(0.01).all())
        self.assertTrue(result[~result.training_defense.eq("l2")].weight_decay.eq(0.0).all())
        self.assertTrue(result.dp_target_epsilon.eq(4.0).all())
        dp = result[result.training_defense.eq("dp_qml")]
        self.assertTrue(dp.dp_optimizer.eq("rmsprop").all())
        self.assertTrue(dp.dp_batch_size.eq(32).all())
        self.assertTrue(dp.dp_epochs.eq(30).all())
        self.assertTrue(dp.dp_scheduler.eq("none").all())


if __name__ == "__main__":
    unittest.main()
