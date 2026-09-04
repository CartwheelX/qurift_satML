from __future__ import annotations

import unittest
from collections import Counter

import numpy as np

from qurift.defenses.protocol import (
    PARTITION_PROTOCOL,
    build_defense_partitions,
    partition_fingerprint,
)
from qurift.defenses.protocol_pooled import (
    POOLED_PARTITION_PROTOCOL,
    build_pooled_defense_partitions,
    pooled_partition_fingerprint,
)


def imbalanced_labels(count: int, *, prevalence: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(count) < prevalence).astype(int)


def credit_shaped() -> dict:
    """Split sizes and prevalence matching the frozen Credit-default protocol."""

    return {
        "train_labels": imbalanced_labels(200, prevalence=0.221, seed=1),
        "valid_labels": imbalanced_labels(200, prevalence=0.221, seed=2),
        "test_labels": imbalanced_labels(2000, prevalence=0.221, seed=3),
        "defense_per_class": 50,
        "attack_per_class": 50,
        "evaluation_per_class": 100,
        "seed": 2026,
    }


class PETSPooledPartitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kwargs = credit_shaped()
        self.frozen = build_defense_partitions(**self.kwargs)

    def test_multiplier_one_reproduces_the_frozen_contract(self) -> None:
        pooled = build_pooled_defense_partitions(**self.kwargs, nonmember_multiplier=1)
        self.assertEqual(pooled.defense_calibration, self.frozen.defense_calibration)
        self.assertEqual(pooled.attack_calibration, self.frozen.attack_calibration)
        self.assertEqual(pooled.final_evaluation, self.frozen.final_evaluation)
        self.assertEqual(
            pooled_partition_fingerprint(pooled), partition_fingerprint(self.frozen)
        )
        # The manifest must be byte-identical, extra keys included, so a default
        # run cannot be told apart from one predating this module.
        self.assertEqual(pooled.to_json(), self.frozen.to_json())
        self.assertEqual(pooled.to_json()["protocol"], PARTITION_PROTOCOL)

    def test_widening_preserves_members_and_calibration(self) -> None:
        frozen_members = [
            record for record in self.frozen.final_evaluation if record.membership == 1
        ]
        frozen_nonmember_ids = {
            record.record_id
            for record in self.frozen.final_evaluation
            if record.membership == 0
        }
        for multiplier in (2, 5, 10):
            with self.subTest(multiplier=multiplier):
                pooled = build_pooled_defense_partitions(
                    **self.kwargs, nonmember_multiplier=multiplier
                )
                self.assertEqual(
                    pooled.defense_calibration, self.frozen.defense_calibration
                )
                self.assertEqual(
                    pooled.attack_calibration, self.frozen.attack_calibration
                )
                members = [
                    record
                    for record in pooled.final_evaluation
                    if record.membership == 1
                ]
                nonmembers = [
                    record
                    for record in pooled.final_evaluation
                    if record.membership == 0
                ]
                # Same members in the same order keeps the enlarged run
                # comparable, and the HSJ subset takes a prefix of this list.
                self.assertEqual(members, frozen_members)
                self.assertEqual(len(nonmembers), multiplier * len(frozen_members))
                self.assertTrue(
                    frozen_nonmember_ids
                    <= {record.record_id for record in nonmembers}
                )

    def test_label_distribution_scales_exactly(self) -> None:
        for multiplier in (2, 5, 10):
            with self.subTest(multiplier=multiplier):
                pooled = build_pooled_defense_partitions(
                    **self.kwargs, nonmember_multiplier=multiplier
                )
                members = Counter(
                    record.task_label
                    for record in pooled.final_evaluation
                    if record.membership == 1
                )
                nonmembers = Counter(
                    record.task_label
                    for record in pooled.final_evaluation
                    if record.membership == 0
                )
                self.assertEqual(set(members), set(nonmembers))
                for label, count in members.items():
                    self.assertEqual(nonmembers[label], count * multiplier)

    def test_widened_manifest_declares_the_pooled_protocol(self) -> None:
        payload = build_pooled_defense_partitions(
            **self.kwargs, nonmember_multiplier=5
        ).to_json()
        self.assertEqual(payload["protocol"], POOLED_PARTITION_PROTOCOL)
        self.assertEqual(payload["evaluation_nonmember_multiplier"], 5)
        self.assertNotEqual(payload["protocol"], PARTITION_PROTOCOL)

    def test_partitions_stay_disjoint_when_widened(self) -> None:
        pooled = build_pooled_defense_partitions(**self.kwargs, nonmember_multiplier=10)
        groups = [
            {record.record_id for record in pooled.defense_calibration},
            {record.record_id for record in pooled.attack_calibration},
            {record.record_id for record in pooled.final_evaluation},
        ]
        for index, left in enumerate(groups):
            for right in groups[index + 1:]:
                self.assertFalse(left & right)

    def test_infeasible_multiplier_reports_the_largest_usable_value(self) -> None:
        # The majority class exhausts its non-member supply before the split
        # total does, so the error must name the per-label limit.
        with self.assertRaisesRegex(ValueError, "largest feasible value"):
            build_pooled_defense_partitions(**self.kwargs, nonmember_multiplier=19)

    def test_multiplier_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            build_pooled_defense_partitions(**self.kwargs, nonmember_multiplier=0)


if __name__ == "__main__":
    unittest.main()
