from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
import torch

from reviewer_tools.qurift_label_only_hsj import (
    hsj_boundary_distance,
    input_bounds_for_dataset,
)


class LabelOnlyHSJTests(unittest.TestCase):
    @staticmethod
    def threshold_labels(points: torch.Tensor) -> torch.Tensor:
        return (points.reshape(len(points), -1)[:, 0] >= 0.0).long()

    @staticmethod
    def constant_labels(points: torch.Tensor) -> torch.Tensor:
        return torch.zeros(len(points), dtype=torch.long)

    def run_search(self, **overrides):
        settings = {
            "origin": torch.tensor([-0.5], dtype=torch.float32),
            "true_label": 0,
            "original_prediction": 0,
            "query_fn": self.threshold_labels,
            "lower": torch.tensor([-1.0]),
            "upper": torch.tensor([1.0]),
            "max_queries": 128,
            "init_queries": 32,
            "init_batch_size": 8,
            "iterations": 4,
            "gradient_samples": 8,
            "binary_steps": 10,
            "step_search_steps": 5,
            "gradient_delta_ratio": 0.1,
            "min_gradient_delta": 1e-4,
            "seed": 17,
        }
        settings.update(overrides)
        return hsj_boundary_distance(**settings)

    def test_boundary_search_is_finite_and_budgeted(self) -> None:
        result = self.run_search()
        self.assertTrue(result["adversarial_initialization_found"])
        self.assertFalse(result["search_censored"])
        self.assertAlmostEqual(result["boundary_distance"], 0.5, delta=0.01)
        self.assertLessEqual(1 + result["boundary_queries"], 128)

    def test_search_is_deterministic_for_a_fixed_seed(self) -> None:
        first = self.run_search()
        second = self.run_search()
        self.assertEqual(first, second)

    def test_initially_misclassified_record_receives_zero(self) -> None:
        result = self.run_search(true_label=1, original_prediction=0)
        self.assertEqual(result["boundary_distance"], 0.0)
        self.assertEqual(result["boundary_queries"], 0)
        self.assertEqual(result["stopping_reason"], "initially_misclassified")

    def test_constant_classifier_is_capped_not_nan(self) -> None:
        result = self.run_search(query_fn=self.constant_labels, init_queries=24)
        self.assertTrue(result["search_censored"])
        self.assertFalse(result["adversarial_initialization_found"])
        self.assertEqual(result["boundary_distance"], 2.0)
        self.assertEqual(result["initialization_queries"], 24)
        self.assertLessEqual(1 + result["boundary_queries"], 128)

    def test_credit_and_wdbc_use_declared_unit_box(self) -> None:
        for dataset in ("credit_default", "breast_cancer_wdbc"):
            bounds = input_bounds_for_dataset(dataset)
            self.assertEqual((bounds.lower, bounds.upper), (-1.0, 1.0))

    def test_explicit_bounds_require_an_ordered_pair(self) -> None:
        with self.assertRaises(ValueError):
            input_bounds_for_dataset("credit_default", clip_min=0.0)
        with self.assertRaises(ValueError):
            input_bounds_for_dataset(
                "credit_default", clip_min=1.0, clip_max=-1.0
            )

    def test_satml_launchers_do_not_use_the_old_chord_attack(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        for relative in (
            "commands/satml_run_credit_label_only_hsj.sh",
            "commands/satml_run_added_attacks.sh",
            "commands/satml_run_fresh_selector.sh",
        ):
            text = (repo_root / relative).read_text(encoding="utf-8")
            self.assertIn("run_label_only_hsj_multigpu.py", text)
            self.assertNotIn("run_label_only_boundary_multigpu.py", text)
        credit_full = (repo_root / "commands/satml_run_credit_attacks.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("satml_run_credit_label_only_hsj.sh", credit_full)
        self.assertNotIn("run_label_only_boundary_multigpu.py", credit_full)


class LabelOnlyHSJPilotAnalysisTests(unittest.TestCase):
    @staticmethod
    def sample_frame(sample_id: str = "member:0") -> pd.DataFrame:
        return pd.DataFrame(
            {
                "target_id": ["target"],
                "sample_id": [sample_id],
                "membership": [1],
                "true_label": [0],
                "source_split": ["train"],
                "source_index": [0],
                "initially_correct": [True],
                "search_censored": [False],
                "boundary_distance": [0.5],
                "total_label_queries": [16],
            }
        )

    def test_pilot_candidate_validation_accepts_common_candidates(self) -> None:
        from satml_tools.analyze_label_only_hsj_pilot import validate_common_candidates

        first = self.sample_frame()
        first.insert(0, "query_budget", 128)
        second = self.sample_frame()
        second.insert(0, "query_budget", 512)
        validate_common_candidates(pd.concat([first, second], ignore_index=True))

    def test_pilot_candidate_validation_rejects_changed_candidates(self) -> None:
        from satml_tools.analyze_label_only_hsj_pilot import validate_common_candidates

        first = self.sample_frame()
        first.insert(0, "query_budget", 128)
        second = self.sample_frame("member:1")
        second.insert(0, "query_budget", 512)
        with self.assertRaises(ValueError):
            validate_common_candidates(pd.concat([first, second], ignore_index=True))


if __name__ == "__main__":
    unittest.main()
