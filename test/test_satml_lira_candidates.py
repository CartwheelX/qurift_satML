from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from pets_tools.score_defended_lira import (
    candidate_partitions,
    lira_selection_fingerprint,
)
from qurift.defenses.protocol import (
    CONFIRMATORY_CREDIT_LABEL_QUOTAS,
    CONFIRMATORY_CREDIT_QUOTA_PLAN,
    build_defense_partitions,
)
from qurift.defenses.protocol_pooled import build_pooled_defense_partitions
from reviewer_tools.qurift_lira_attack import CandidateDataset, cell_id


class LiRACandidateTests(unittest.TestCase):
    def test_reference_bank_identity_includes_credit_block(self) -> None:
        base = {"structural_cell_id": "z_r1_d2", "weight_decay": 0.0}
        first = cell_id({**base, "block_id": "credit_b01"})
        second = cell_id({**base, "block_id": "credit_b02"})
        self.assertNotEqual(first, second)
        self.assertIn("credit_b01", first)

    def test_candidate_indices_address_selected_tensors(self) -> None:
        inputs = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        labels = torch.tensor([0, 1, 0, 1, 0, 1])
        candidates = CandidateDataset(inputs, labels)
        subset = torch.utils.data.Subset(candidates, [1, 4])
        self.assertTrue(torch.equal(subset[0]["image"], inputs[1]))
        self.assertEqual(int(subset[1]["digit"]), int(labels[4]))

    @staticmethod
    def common_quota_partitions(*, multiplier: int = 1):
        kwargs = dict(
            train_labels=[0] * 156 + [1] * 44,
            valid_labels=[0] * 156 + [1] * 44,
            test_labels=[0] * 1558 + [1] * 442,
            defense_per_class=50,
            attack_per_class=50,
            evaluation_per_class=100,
            seed=2026,
            label_quotas=CONFIRMATORY_CREDIT_LABEL_QUOTAS,
            quota_plan_name=CONFIRMATORY_CREDIT_QUOTA_PLAN,
        )
        if multiplier == 1:
            return build_defense_partitions(**kwargs)
        return build_pooled_defense_partitions(
            **kwargs, nonmember_multiplier=multiplier
        )

    def test_common_quota_plan_is_fixed_and_uses_every_member(self) -> None:
        partitions = self.common_quota_partitions()
        observed = {}
        all_members = []
        for name in (
            "defense_calibration",
            "attack_calibration",
            "final_evaluation",
        ):
            refs = getattr(partitions, name)
            observed[name] = dict(
                Counter(ref.task_label for ref in refs if ref.membership == 1)
            )
            all_members.extend(
                ref.record_id for ref in refs if ref.membership == 1
            )
        self.assertEqual(observed, CONFIRMATORY_CREDIT_LABEL_QUOTAS)
        self.assertEqual(len(all_members), 200)
        self.assertEqual(len(set(all_members)), 200)

    def test_common_quota_is_compatible_with_smallest_reference_pool(self) -> None:
        partitions = self.common_quota_partitions()
        labels = torch.tensor(
            [0] * 156 + [1] * 44 + [0] * 168 + [1] * 32,
            dtype=torch.long,
        )
        samples = SimpleNamespace(
            membership=torch.tensor([1] * 200 + [0] * 200),
            labels=labels,
            split_names=["train"] * 200 + ["test"] * 200,
            source_indices=list(range(200)) + list(range(200)),
            sample_ids=[f"candidate-{index}" for index in range(400)],
        )
        attack, evaluation = candidate_partitions(
            samples,
            partitions,
            attack_per_class=50,
            evaluation_per_class=100,
            seed=2026,
        )
        self.assertEqual(len(attack), 100)
        self.assertEqual(len(evaluation), 200)
        for indices, per_membership in ((attack, 50), (evaluation, 100)):
            selected = labels[torch.as_tensor(indices)].numpy()
            np.testing.assert_array_equal(
                selected[:per_membership], selected[per_membership:]
            )
        first = lira_selection_fingerprint(
            samples, partitions, attack, evaluation
        )
        second = lira_selection_fingerprint(
            samples, partitions, attack, evaluation
        )
        changed = lira_selection_fingerprint(
            samples, partitions, attack[::-1], evaluation
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_pooled_low_fpr_partition_reuses_common_members(self) -> None:
        balanced = self.common_quota_partitions()
        pooled = self.common_quota_partitions(multiplier=10)
        self.assertEqual(
            pooled.defense_calibration, balanced.defense_calibration
        )
        self.assertEqual(pooled.attack_calibration, balanced.attack_calibration)
        pooled_members = tuple(
            ref for ref in pooled.final_evaluation if ref.membership == 1
        )
        balanced_members = tuple(
            ref for ref in balanced.final_evaluation if ref.membership == 1
        )
        self.assertEqual(pooled_members, balanced_members)
        nonmember_counts = Counter(
            ref.task_label
            for ref in pooled.final_evaluation
            if ref.membership == 0
        )
        self.assertEqual(nonmember_counts, Counter({0: 790, 1: 210}))



if __name__ == "__main__":
    unittest.main()
