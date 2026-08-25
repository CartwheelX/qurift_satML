from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from reviewer_tools.qurift_noisy_eval import (
    aggregate_condition_shards,
    condition_belongs_to_shard,
    evaluate_query_count_runs,
    merge_counts,
    parse_query_shot_pairs,
    shard_output_path,
)
from reviewer_tools.reviewer_common import stratified_bootstrap_tpr_at_fpr
from reviewer_tools.qurift_qiskit_bridge import (
    BackendNoiseMetadata,
    load_backend_noise_snapshot,
    write_backend_snapshot,
)
from satml_tools.analyze_noise_budget import analyze_noise, analyze_utility


class NoiseBudgetTests(unittest.TestCase):
    def test_condition_shards_are_disjoint_and_complete(self) -> None:
        assignments = [
            [ordinal for ordinal in range(120) if condition_belongs_to_shard(
                ordinal, shard_index=shard, shard_count=4
            )]
            for shard in range(4)
        ]
        self.assertTrue(all(len(values) == 30 for values in assignments))
        self.assertEqual(sorted(value for values in assignments for value in values), list(range(120)))

    def test_condition_shard_outputs_merge_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for shard in range(2):
                status = pd.DataFrame([{
                    "target_id": "target", "mode": "ideal_shot", "queries": 1,
                    "shots": 128, "simulator_seed": shard,
                    "aggregation": "mean_api_probabilities", "status": "ok",
                }])
                status.to_csv(shard_output_path(
                    root, "condition_status.csv", shard_index=shard, shard_count=2
                ), index=False)
                for filename in (
                    "condition_metrics_raw.csv", "per_sample_predictions.csv", "failures.csv"
                ):
                    pd.DataFrame().to_csv(shard_output_path(
                        root, filename, shard_index=shard, shard_count=2
                    ), index=False)
            report = aggregate_condition_shards(
                root, shard_count=2, bootstrap=10, bootstrap_seed=3
            )
            self.assertEqual(report["conditions"], 2)
            self.assertEqual(report["failures"], 0)
            merged = pd.read_csv(root / "condition_status.csv")
            self.assertEqual(set(merged.simulator_seed), {0, 1})

    def test_query_shot_pairs_preserve_declared_budget(self) -> None:
        pairs = parse_query_shot_pairs("1x2560,20x128")
        self.assertEqual(pairs, [(1, 2560), (20, 128)])
        self.assertEqual({queries * shots for queries, shots in pairs}, {2560})

    def test_pooled_count_diagnostic_is_aggregated_per_circuit(self) -> None:
        merged = merge_counts([[{"0": 3, "1": 1}], [{"0": 2, "1": 2}]])
        self.assertEqual(merged, [{"0": 5, "1": 3}])

    def test_api_queries_pass_through_nonlinear_head_before_averaging(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(1, 2, bias=False)
                with torch.no_grad():
                    self.linear.weight.copy_(torch.tensor([[2.0], [0.0]]))

        expectations, query_pv, mean_pv, pooled_pv = evaluate_query_count_runs(
            Model(),
            [[{"0": 10}], [{"0": 5, "1": 5}]],
            n_wires=1,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(expectations.shape), (2, 1, 1))
        self.assertEqual(tuple(query_pv.shape), (2, 1, 2))
        torch.testing.assert_close(mean_pv, query_pv.mean(dim=0))
        self.assertFalse(torch.allclose(mean_pv, pooled_pv))

    def test_low_fpr_bootstrap_respects_bounds(self) -> None:
        y = [1] * 20 + [0] * 200
        scores = list(range(220, 200, -1)) + list(range(200, 0, -1))
        low, high, valid = stratified_bootstrap_tpr_at_fpr(
            y, scores, requested_fpr=0.01, n_boot=250, seed=3, chunk_size=64
        )
        self.assertEqual(valid, 250)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)

    def test_backend_snapshot_excludes_credentials(self) -> None:
        class Serializable:
            def to_dict(self):
                return {"value": 1}

        class Backend:
            def properties(self):
                return Serializable()

            def configuration(self):
                return Serializable()

        metadata = BackendNoiseMetadata(
            requested_backend_name="fake", requested_noise_backend_name="fake",
            resolved_backend_name="fake", resolved_noise_backend_name="fake",
            authentication_mode="environment_credentials", noise_model_loaded=True,
            noise_load_error=None, gate_error_enabled=True, readout_error_enabled=True,
            thermal_relaxation_enabled=True, calibration_timestamp="2026-01-01T00:00:00Z",
            basis_gates=["x"], noise_basis_gates=["x"], coupling_map=[], backend_num_qubits=1,
            noise_instructions=["x"], noise_qubits=[[0]], backend_mismatch=False,
        )
        context = SimpleNamespace(
            metadata=metadata, backend=Backend(), noise_backend=None, noise_model=Serializable()
        )
        with tempfile.TemporaryDirectory() as directory:
            write_backend_snapshot(context, Path(directory))
            combined = "".join(path.read_text() for path in Path(directory).glob("*.json"))
        self.assertNotIn("token", combined.lower())
        self.assertIn("credentials_recorded", combined)

    def test_frozen_snapshot_round_trip_and_hash_guard(self) -> None:
        try:
            from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
        except ImportError:
            self.skipTest("qiskit-aer is unavailable")

        class Backend:
            def properties(self):
                return None

            def configuration(self):
                return None

        metadata = BackendNoiseMetadata(
            requested_backend_name="fake", requested_noise_backend_name="fake",
            resolved_backend_name="fake", resolved_noise_backend_name="fake",
            authentication_mode="environment_credentials", noise_model_loaded=True,
            noise_load_error=None, gate_error_enabled=True, readout_error_enabled=True,
            thermal_relaxation_enabled=True, calibration_timestamp="2026-01-01T00:00:00Z",
            basis_gates=["x"], noise_basis_gates=[], coupling_map=[], backend_num_qubits=1,
            noise_instructions=[], noise_qubits=[], backend_mismatch=False,
        )
        noise_model = NoiseModel()
        # Thermal relaxation serializes Kraus matrices as complex NumPy arrays.
        # This catches lossy ``default=str`` JSON serialization, which an empty
        # NoiseModel round trip cannot detect.
        noise_model.add_all_qubit_quantum_error(
            thermal_relaxation_error(50_000.0, 70_000.0, 100.0), ["x"]
        )
        context = SimpleNamespace(
            metadata=metadata, backend=Backend(), noise_backend=None,
            noise_model=noise_model,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_backend_snapshot(context, root)
            loaded = load_backend_noise_snapshot(root, require_noise=True)
            self.assertEqual(loaded.metadata.authentication_mode, "frozen_local_snapshot")
            self.assertIsNotNone(loaded.noise_model)
            self.assertIn("x", loaded.noise_model.noise_instructions)
            metadata_path = root / "metadata.json"
            metadata_path.write_text(metadata_path.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_backend_noise_snapshot(root, require_noise=True)

    def test_noise_analysis_does_not_pool_calibration_profiles(self) -> None:
        rows = []
        for calibration, offset in (("cal-a", 0.0), ("cal-b", 0.1)):
            for target_index in range(3):
                target = f"t{target_index}"
                rows.append(
                    {"target_id": target, "calibration_profile": calibration, "mode": "exact",
                     "queries": 0, "shots": 0, "total_shots": 0, "backend_name": "none",
                     "noise_model_loaded": False, "simulator_seed": -1,
                     "metric_scope": "membership", "metric_name": "loss_auc",
                     "value": 0.55 + 0.05 * target_index}
                )
                for simulator_seed in range(2):
                    for queries, shots in ((1, 2560), (20, 128)):
                        rows.append(
                            {"target_id": target, "calibration_profile": calibration,
                             "mode": "noisy_shot", "queries": queries, "shots": shots,
                             "total_shots": 2560, "backend_name": "fake",
                             "noise_model_loaded": True, "simulator_seed": simulator_seed,
                             "metric_scope": "membership", "metric_name": "loss_auc",
                             "value": 0.54 + 0.05 * target_index + offset + queries * 0.0001}
                        )
        summary, query, ordering = analyze_noise(pd.DataFrame(rows))
        self.assertEqual(set(summary.calibration_profile), {"cal-a", "cal-b"})
        self.assertEqual(set(query.calibration_profile), {"cal-a", "cal-b"})
        self.assertEqual(set(ordering.calibration_profile), {"cal-a", "cal-b"})
        self.assertTrue((query.n_simulator_seeds == 2).all())

    def test_noise_utility_keeps_query_budget_keys(self) -> None:
        rows = []
        for seed in range(3):
            rows.append(
                {"target_id": "t", "calibration_profile": "cal", "mode": "noisy_shot",
                 "queries": 20, "shots": 128, "total_shots": 2560, "backend_name": "fake",
                 "noise_model_loaded": True, "simulator_seed": seed, "metric_scope": "test",
                 "metric_name": "target_accuracy", "value": 0.7 + seed * 0.01}
            )
        utility = analyze_utility(pd.DataFrame(rows))
        self.assertEqual(len(utility), 1)
        self.assertEqual(int(utility.iloc[0].n_simulator_replicates), 3)
        self.assertEqual(int(utility.iloc[0].total_shots), 2560)


if __name__ == "__main__":
    unittest.main()
