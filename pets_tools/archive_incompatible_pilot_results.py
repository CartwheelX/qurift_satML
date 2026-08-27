#!/usr/bin/env python3
"""Recoverably archive PETS results produced before the label-matched protocol."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pets_tools.run_defense_evaluation import EVALUATION_PROTOCOL
from qurift.defenses.protocol import PARTITION_PROTOCOL


def is_current(target_dir: Path) -> bool:
    metadata_path = target_dir / "evaluation_metadata.json"
    partition_path = target_dir / "partition_manifest.json"
    utility_path = target_dir / "test_utility_predictions.csv"
    if not metadata_path.exists() or not partition_path.exists() or not utility_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
        partition = json.loads(partition_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return bool(
        metadata.get("protocol") == EVALUATION_PROTOCOL
        and metadata.get("utility_evaluation", {}).get("scope")
        == "full_held_out_test_split"
        and partition.get("protocol") == PARTITION_PROTOCOL
    )


def archive_incompatible(
    targets: pd.DataFrame,
    *,
    result_root: Path,
    archive_root: Path,
    analysis_dir: Path | None,
    stamp: str,
) -> dict:
    destination_root = archive_root / stamp
    moved = []
    current = []
    absent = []
    for target_id in targets.target_id.astype(str).drop_duplicates():
        source = result_root / target_id
        if not source.exists():
            absent.append(target_id)
            continue
        if is_current(source):
            current.append(target_id)
            continue
        destination = destination_root / "defenses" / target_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"archive destination already exists: {destination}")
        source.replace(destination)
        moved.append(target_id)
    analysis_archived = None
    if moved and analysis_dir is not None and analysis_dir.exists():
        destination = destination_root / "pilot_analysis"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"archive destination already exists: {destination}")
        analysis_dir.replace(destination)
        analysis_archived = str(destination.resolve())
    payload = {
        "protocol": "pets_recoverable_pre_label_match_archive_v1",
        "timestamp_utc": stamp,
        "required_evaluation_protocol": EVALUATION_PROTOCOL,
        "required_partition_protocol": PARTITION_PROTOCOL,
        "moved_target_ids": moved,
        "already_current_target_ids": current,
        "absent_target_ids": absent,
        "analysis_archived": analysis_archived,
        "target_checkpoints_moved": False,
        "lira_reference_checkpoints_moved": False,
    }
    if moved:
        destination_root.mkdir(parents=True, exist_ok=True)
        (destination_root / "archive_manifest.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument(
        "--result-root", type=Path, default=Path("pets_results/defenses")
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("pets_results/archive/pre_label_matched"),
    )
    parser.add_argument(
        "--analysis-dir", type=Path, default=Path("pets_results/pilot_analysis")
    )
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = archive_incompatible(
        pd.read_csv(args.targets),
        result_root=args.result_root,
        archive_root=args.archive_root,
        analysis_dir=args.analysis_dir,
        stamp=stamp,
    )
    print(
        f"[OK] archived={len(payload['moved_target_ids'])} "
        f"current={len(payload['already_current_target_ids'])} "
        f"absent={len(payload['absent_target_ids'])}"
    )


if __name__ == "__main__":
    main()
