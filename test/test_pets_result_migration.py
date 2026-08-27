from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from pets_tools.archive_incompatible_pilot_results import archive_incompatible
from pets_tools.run_defense_evaluation import EVALUATION_PROTOCOL
from qurift.defenses.protocol import PARTITION_PROTOCOL


class PETSResultMigrationTests(unittest.TestCase):
    def test_old_results_are_moved_and_current_results_are_retained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            archive = root / "archive"
            analysis = root / "analysis"
            analysis.mkdir()
            (analysis / "old.csv").write_text("x\n1\n")
            old = results / "old_target"
            old.mkdir(parents=True)
            (old / "evaluation_metadata.json").write_text(
                json.dumps({"protocol": "pets_defense_pilot_v1"})
            )
            current = results / "current_target"
            current.mkdir(parents=True)
            (current / "evaluation_metadata.json").write_text(
                json.dumps(
                    {
                        "protocol": EVALUATION_PROTOCOL,
                        "utility_evaluation": {"scope": "full_held_out_test_split"},
                    }
                )
            )
            (current / "partition_manifest.json").write_text(
                json.dumps({"protocol": PARTITION_PROTOCOL})
            )
            (current / "test_utility_predictions.csv").write_text("x\n1\n")
            payload = archive_incompatible(
                pd.DataFrame({"target_id": ["old_target", "current_target", "absent"]}),
                result_root=results,
                archive_root=archive,
                analysis_dir=analysis,
                stamp="20260826T000000Z",
            )
            self.assertFalse(old.exists())
            self.assertTrue(
                (archive / "20260826T000000Z" / "defenses" / "old_target").exists()
            )
            self.assertTrue(current.exists())
            self.assertFalse(analysis.exists())
            self.assertEqual(payload["moved_target_ids"], ["old_target"])
            self.assertEqual(payload["already_current_target_ids"], ["current_target"])
            self.assertEqual(payload["absent_target_ids"], ["absent"])


if __name__ == "__main__":
    unittest.main()
