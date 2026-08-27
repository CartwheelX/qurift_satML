#!/usr/bin/env python3
"""Create auditable filtered target manifests without shell quoting tricks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--training-defenses", default="")
    parser.add_argument("--block-ids", default="")
    parser.add_argument("--structural-roles", default="")
    args = parser.parse_args()
    frame = pd.read_csv(args.targets)
    filters = {}
    for column, raw in (
        ("training_defense", args.training_defenses),
        ("block_id", args.block_ids),
        ("defense_structural_role", args.structural_roles),
    ):
        values = [value.strip() for value in raw.split(",") if value.strip()]
        if values:
            if column not in frame:
                parser.error(f"target table has no {column!r} column")
            frame = frame[frame[column].astype(str).isin(values)]
            filters[column] = values
    if frame.empty:
        parser.error("filters selected zero targets")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "protocol": "pets_filtered_target_manifest_v1",
                "source": str(args.targets.resolve()),
                "filters": filters,
                "targets": len(frame),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[OK] targets={len(frame)} -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
