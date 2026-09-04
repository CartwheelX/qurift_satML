from __future__ import annotations

import unittest
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from qurift.defenses.attacks import adaptive_threshold_metrics, attack_signals
from qurift.defenses.oracle import RawOracle
from qurift.defenses.protocol import build_defense_partitions, partition_fingerprint
from pets_tools.run_defense_hsj import hsj_record_seed
from pets_tools.run_query_stress import PROTOCOL, existing_result_matches, nearby_query_features
from pets_tools.score_defended_lira import candidate_partitions, defense_monte_carlo_draws


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.linear(inputs), dim=1)


class PETSProtocolTests(unittest.TestCase):
    def test_lira_only_averages_stochastic_defenses(self) -> None:
        self.assertEqual(defense_monte_carlo_draws("dynanoise", 10), 10)
        self.assertEqual(defense_monte_carlo_draws("hamp_output", 10), 10)
        self.assertEqual(defense_monte_carlo_draws("memgq_lattice", 10), 1)
        self.assertEqual(defense_monte_carlo_draws("none", 10), 1)

    def test_nearby_queries_are_reproducible_from_common_seed(self) -> None:
        torch.manual_seed(4)
        inputs = torch.randn(6, 2).clamp(-0.8, 0.8)
        labels = torch.tensor([0, 1, 0, 1, 0, 1])
        oracle = RawOracle(TinyModel())
        kwargs = dict(
            oracle=oracle,
            inputs=inputs,
            labels=labels,
            ids=[f"sample:{index}" for index in range(len(inputs))],
            queries=5,
            radius=0.01,
            batch_size=3,
        )
        first, _ = nearby_query_features(**kwargs, seed=2026)
        second, _ = nearby_query_features(**kwargs, seed=2026)
        different, _ = nearby_query_features(**kwargs, seed=2027)
        self.assertTrue((first == second).all())
        self.assertFalse((first == different).all())

    def test_query_resume_only_accepts_matching_current_protocol(self) -> None:
        payload = {
            "protocol": PROTOCOL,
            "defenses": ["none", "memgq_lattice"],
            "queries": 32,
            "linf_radius": 0.005,
            "common_random_perturbations_across_defenses": True,
        }
        self.assertTrue(
            existing_result_matches(
                payload,
                defenses=["memgq_lattice", "none"],
                queries=32,
                radius=0.005,
            )
        )
        self.assertFalse(
            existing_result_matches(
                payload,
                defenses=["none", "lattice_round", "memgq_lattice"],
                queries=32,
                radius=0.005,
            )
        )
        payload["protocol_arguments"] = {"seed": 2026}
        self.assertTrue(
            existing_result_matches(
                payload,
                defenses=["none", "memgq_lattice"],
                queries=32,
                radius=0.005,
                protocol_arguments={"seed": 2026},
            )
        )
        self.assertFalse(
            existing_result_matches(
                payload,
                defenses=["none", "memgq_lattice"],
                queries=32,
                radius=0.005,
                protocol_arguments={"seed": 2027},
            )
        )

    def test_hsj_record_seed_varies_by_record_without_a_defense_input(self) -> None:
        self.assertEqual(
            hsj_record_seed(2026, "target", "final_evaluation", "test:7"),
            hsj_record_seed(2026, "target", "final_evaluation", "test:7"),
        )
        self.assertNotEqual(
            hsj_record_seed(2026, "target", "final_evaluation", "test:7"),
            hsj_record_seed(2026, "target", "final_evaluation", "test:8"),
        )

    def test_partitions_are_balanced_disjoint_and_reproducible(self) -> None:
        kwargs = dict(
            train_labels=[0] * 156 + [1] * 44,
            valid_labels=[0] * 156 + [1] * 44,
            test_labels=[0] * 1558 + [1] * 442,
            defense_per_class=50,
            attack_per_class=50,
            evaluation_per_class=100,
            seed=2026,
        )
        first = build_defense_partitions(**kwargs)
        second = build_defense_partitions(**kwargs)
        self.assertEqual(partition_fingerprint(first), partition_fingerprint(second))
        for name in ("defense_calibration", "attack_calibration", "final_evaluation"):
            values = getattr(first, name)
            self.assertEqual(sum(item.membership for item in values), len(values) // 2)
            member_labels = Counter(
                item.task_label for item in values if item.membership == 1
            )
            nonmember_labels = Counter(
                item.task_label for item in values if item.membership == 0
            )
            self.assertEqual(member_labels, nonmember_labels)
            members = [item.task_label for item in values if item.membership == 1]
            nonmembers = [item.task_label for item in values if item.membership == 0]
            self.assertEqual(members, nonmembers)
        self.assertEqual(
            Counter(item.task_label for item in first.defense_calibration if item.membership == 1),
            Counter({0: 39, 1: 11}),
        )
        self.assertEqual(
            Counter(item.task_label for item in first.final_evaluation if item.membership == 1),
            Counter({0: 78, 1: 22}),
        )

    def test_lira_candidate_subsets_are_task_label_matched(self) -> None:
        train_labels = torch.tensor([0] * 156 + [1] * 44)
        test_labels = torch.tensor([0] * 156 + [1] * 44)
        partitions = build_defense_partitions(
            train_labels=train_labels,
            valid_labels=train_labels,
            test_labels=[0] * 1558 + [1] * 442,
            defense_per_class=50,
            attack_per_class=50,
            evaluation_per_class=100,
            seed=2026,
        )
        samples = SimpleNamespace(
            membership=torch.tensor([1] * 200 + [0] * 200),
            labels=torch.cat([train_labels, test_labels]),
            split_names=["train"] * 200 + ["test"] * 200,
            source_indices=list(range(200)) + list(range(200)),
        )
        attack, evaluation = candidate_partitions(
            samples,
            partitions,
            attack_per_class=50,
            evaluation_per_class=100,
            seed=2026,
        )
        for indices, per_membership in ((attack, 50), (evaluation, 100)):
            labels = samples.labels[torch.as_tensor(indices)].numpy()
            np.testing.assert_array_equal(
                labels[:per_membership], labels[per_membership:]
            )

    def test_partition_overcommit_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_defense_partitions(
                train_labels=[0, 1] * 5,
                valid_labels=[0, 1] * 5,
                test_labels=[0, 1] * 5,
                defense_per_class=5,
                attack_per_class=5,
                evaluation_per_class=5,
                seed=1,
            )

    def test_attack_scores_use_one_as_member(self) -> None:
        torch.manual_seed(2)
        model = TinyModel()
        output = RawOracle(model).predict(torch.randn(8, 2))
        true = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        signals = attack_signals(output, true)
        membership = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
        metrics = adaptive_threshold_metrics(
            signals["loss"], membership, signals["loss"], membership
        )
        self.assertIn("auc", metrics)
        self.assertIn("threshold", metrics)
        self.assertIn(metrics["score_direction"], {"as_defined", "inverted_from_calibration"})


if __name__ == "__main__":
    unittest.main()
