from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pets_tools.analyze_confirmatory_v2 import (
    LIRA_PROTOCOL,
    LITERATURE_DEFENSES,
    PRIMARY_ATTACK,
    exact_sign_flip_p,
    holm_adjust,
    low_fpr_resolution_table,
    validate_lira_alias_equivalence,
    validate_primary_matrix,
)
from pets_tools.build_confirmatory_targets import (
    DEFAULT_ROLES,
    build_structural_targets,
    discovery_stress_evidence,
)
from pets_tools.validate_confirmatory_manifest import validate
from reviewer_tools.qurift_lira_attack import (
    cell_id,
    reference_pairing_id,
    reference_training_spec,
)


def source_table() -> pd.DataFrame:
    rows = []
    for index, cell in enumerate(DEFAULT_ROLES.values()):
        family, reps, depth = (
            ("eff_su2", 1, 6)
            if cell == "eff_su2_r1_d6"
            else ("eff_su2", 5, 6)
            if cell == "eff_su2_r5_d6"
            else ("zz", 5, 6)
        )
        rows.append(
            {
                "target_id": f"source_{index}",
                "structural_cell_id": cell,
                "fm_kind": family,
                "reps": reps,
                "depth": depth,
                "data_seed": 40 + index,
                "model_seed": 50 + index,
                "batch_size": 16,
                "epochs": 100,
                "learning_rate": 0.05,
            }
        )
    return pd.DataFrame(rows)


class ConfirmatoryV2Tests(unittest.TestCase):
    def test_three_roles_are_paired_with_fresh_seeds(self) -> None:
        source = source_table()
        structural = build_structural_targets(
            source,
            roles=DEFAULT_ROLES,
            blocks=8,
            data_seed_start=1000,
            model_seed_start=2000,
            forbidden_data_seeds=set(source.data_seed),
            forbidden_model_seeds=set(source.model_seed),
        )
        self.assertEqual(len(structural), 24)
        self.assertEqual(structural.block_id.nunique(), 8)
        for _, block in structural.groupby("block_id"):
            self.assertEqual(set(block.defense_structural_role), set(DEFAULT_ROLES))
            self.assertEqual(block.data_seed.nunique(), 1)
            self.assertEqual(block.model_seed.nunique(), 1)

    def test_manifest_validator_keeps_watkins_dp_primary(self) -> None:
        source = source_table()
        structural = build_structural_targets(
            source,
            roles=DEFAULT_ROLES,
            blocks=8,
            data_seed_start=1000,
            model_seed_start=2000,
            forbidden_data_seeds=set(source.data_seed),
            forbidden_model_seeds=set(source.model_seed),
        )
        frames = []
        for defense in ("none", "l2", "hamp_train", "dp_qml"):
            frame = structural.copy()
            frame["training_defense"] = defense
            frame["target_id"] = frame.target_id + f"__{defense}"
            frame["experiment"] = "petsv2_credit_confirmatory_training"
            frame["dp_batch_size"] = 32
            frame["dp_epochs"] = 30
            frame["dp_learning_rate"] = 0.05
            frame["dp_protocol"] = "watkins_faithful_core_v2"
            frame["dp_delta"] = 1e-5
            frame["dp_max_grad_norm"] = 1.0
            frame["dp_optimizer"] = "rmsprop"
            frame["dp_scheduler"] = "none"
            frames.append(frame)
        payload = validate(pd.concat(frames, ignore_index=True), [source])
        self.assertTrue(payload["valid"])

    def test_stress_role_is_backed_by_credit_discovery_blocks(self) -> None:
        rows = []
        means = {
            "eff_su2_r1_d6": 0.52,
            "eff_su2_r5_d6": 0.56,
            "zz_r5_d6": 0.61,
        }
        for seed in range(8):
            for cell, value in means.items():
                rows.append(
                    {
                        "attack": "lira_online_fixed_variance",
                        "auc": value + seed * 0.001,
                        "structural_cell_id": cell,
                        "data_seed": seed,
                    }
                )
        evidence = discovery_stress_evidence(
            pd.DataFrame(rows), roles=DEFAULT_ROLES
        )
        self.assertEqual(evidence["independent_blocks"], 8)
        self.assertEqual(evidence["stress_minus_repetition"]["positive_blocks"], 8)
        self.assertAlmostEqual(
            evidence["stress_minus_repetition"]["mean_auc_difference"], 0.05
        )

    def test_reference_banks_separate_training_mechanisms(self) -> None:
        base = {
            "structural_cell_id": "eff_su2_r1_d6",
            "block_id": "petsv2_b01",
            "weight_decay": 0.0,
        }
        ordinary = cell_id({**base, "training_defense": "none"})
        hamp = cell_id({**base, "training_defense": "hamp_train"})
        dp = cell_id(
            {
                **base,
                "training_defense": "dp_qml",
                "dp_target_epsilon": 64,
                "dp_batch_size": 32,
                "dp_epochs": 30,
                "dp_learning_rate": 0.05,
            }
        )
        self.assertEqual(len({ordinary, hamp, dp}), 3)
        self.assertEqual(ordinary, "eff_su2_r1_d6_wd0_blockpetsv2_b01")
        self.assertEqual(
            cell_id({**base, "training_defense": np.nan}), ordinary
        )
        self.assertEqual(
            reference_pairing_id({**base, "training_defense": "none"}),
            reference_pairing_id({**base, "training_defense": "dp_qml"}),
        )

    def test_dp_reference_uses_the_dp_schedule_columns(self) -> None:
        spec = reference_training_spec(
            {
                "training_defense": "dp_qml",
                "batch_size": 16,
                "epochs": 100,
                "learning_rate": 0.01,
                "dp_batch_size": 32,
                "dp_epochs": 30,
                "dp_learning_rate": 0.05,
            }
        )
        self.assertEqual(spec["optimizer"], "rmsprop")
        self.assertEqual(spec["scheduler"], "none")
        self.assertEqual(spec["batch_size"], 32)
        self.assertEqual(spec["epochs"], 30)
        self.assertEqual(spec["learning_rate"], 0.05)

    def test_exact_sign_flip_and_holm_are_deterministic(self) -> None:
        values = np.arange(1, 9, dtype=float)
        self.assertAlmostEqual(exact_sign_flip_p(values), 2 / 256)
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

    def test_primary_analysis_refuses_an_incomplete_matrix(self) -> None:
        rows = [
            {
                "block_id": f"b{block}",
                "effective_defense": defense,
                "structural_role": "stress",
                "attack": PRIMARY_ATTACK,
            }
            for block in range(8)
            for defense in ("none", *LITERATURE_DEFENSES)
        ]
        frame = pd.DataFrame(rows)
        validate_primary_matrix(frame)
        with self.assertRaisesRegex(ValueError, "incomplete primary endpoint"):
            validate_primary_matrix(frame.iloc[:-1])

    def test_low_fpr_resolution_is_reported_not_implied(self) -> None:
        frame = pd.DataFrame(
            {
                "source_protocol": ["lira", "lira"],
                "attack": [PRIMARY_ATTACK, PRIMARY_ATTACK],
                "effective_defense": ["none", "none"],
                "n_evaluation_nonmember": [100, 100],
                "empirical_fpr_resolution": [0.01, 0.01],
                "target_0_1_fpr_resolvable": [False, False],
                "target_1_fpr_resolvable": [True, True],
            }
        )
        result = low_fpr_resolution_table(frame).iloc[0]
        self.assertEqual(result.minimum_nonmembers, 100)
        self.assertEqual(result.fraction_target_0_1_fpr_resolvable, 0.0)
        self.assertEqual(result.fraction_target_1_fpr_resolvable, 1.0)

    def test_one_sided_z_alias_is_not_counted_as_an_independent_test(self) -> None:
        rows = []
        pairs = (
            ("lira_offline", "lira_offline_one_sided_z"),
            (
                "lira_offline_fixed_variance",
                "lira_offline_one_sided_z_fixed_variance",
            ),
        )
        for canonical, alias in pairs:
            for target, auc in (("target_a", 0.61), ("target_b", 0.58)):
                fixed = {
                    "target_id": target,
                    "effective_defense": "none",
                    "structural_role": "stress",
                    "source_protocol": LIRA_PROTOCOL,
                    "auc": auc,
                    "balanced_accuracy": auc - 0.02,
                }
                rows.append({**fixed, "attack": canonical})
                rows.append({**fixed, "attack": alias})
        result = validate_lira_alias_equivalence(pd.DataFrame(rows))
        self.assertEqual(len(result), 2)
        self.assertFalse(result.included_in_inferential_multiplicity.any())


if __name__ == "__main__":
    unittest.main()
