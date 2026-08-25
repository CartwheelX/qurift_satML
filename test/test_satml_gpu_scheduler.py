from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from reviewer_tools.gpu_scheduler import GPUState, plan_gpu_slots


class GPUSchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = patch.dict(
            os.environ,
            {
                key: value
                for key, value in os.environ.items()
                if not key.startswith("QURIFT_GPU_")
            },
            clear=True,
        )
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()

    def test_hsj_slots_are_memory_capped_and_balanced(self) -> None:
        states = {
            1: GPUState(1, total_mb=81920, used_mb=42691, utilization=0),
            2: GPUState(2, total_mb=81920, used_mb=0, utilization=0),
        }
        plan = plan_gpu_slots(
            [1, 2],
            jobs_per_gpu=3,
            profile_name="label_only_hsj",
            pending_jobs=5,
            states=states,
        )
        self.assertEqual(plan.capacities, {1: 2, 2: 3})
        self.assertEqual(plan.assigned, {1: 2, 2: 3})
        self.assertEqual(plan.concurrency, 5)

    def test_auto_qnn_slots_fill_a_short_manifest_fairly(self) -> None:
        states = {
            gpu: GPUState(gpu, total_mb=81920, used_mb=0, utilization=0)
            for gpu in range(7)
        }
        plan = plan_gpu_slots(
            list(states),
            jobs_per_gpu="auto",
            profile_name="qnn_train",
            pending_jobs=15,
            states=states,
        )
        self.assertEqual(plan.concurrency, 15)
        self.assertLessEqual(max(plan.assigned.values()) - min(plan.assigned.values()), 1)
        self.assertTrue(all(value <= 3 for value in plan.assigned.values()))

    def test_existing_high_utilization_reduces_capacity(self) -> None:
        state = {0: GPUState(0, total_mb=81920, used_mb=0, utilization=91)}
        plan = plan_gpu_slots(
            [0],
            jobs_per_gpu="auto",
            profile_name="qnn_train",
            states=state,
        )
        self.assertEqual(plan.capacities[0], 1)
        self.assertEqual(plan.concurrency, 1)


if __name__ == "__main__":
    unittest.main()
