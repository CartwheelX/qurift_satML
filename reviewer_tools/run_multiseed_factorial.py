#!/usr/bin/env python3
"""Launch the pre-specified feature-map × repetition × depth subset."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("reviewer_targets/multiseed_factorial_targets.csv"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--script", type=Path, default=Path("experiments/qurift_main.py"))
    parser.add_argument("--out", type=Path, default=Path("reviewer_runs"))
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--jobs-per-gpu", default="2")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    args = parser.parse_args()

    runner = Path(__file__).with_name("run_target_table_dgx.py")
    command = [
        sys.executable,
        str(runner),
        "--targets",
        str(args.targets),
        "--repo-root",
        str(args.repo_root),
        "--script",
        str(args.script),
        "--out",
        str(args.out),
        "--gpus",
        args.gpus,
        "--jobs-per-gpu",
        str(args.jobs_per_gpu),
        "--cpu-threads",
        str(args.cpu_threads),
    ]
    if args.resume:
        command.append("--resume")
    if args.dry_run:
        command.append("--dry-run")
    if args.max_jobs is not None:
        command.extend(["--max-jobs", str(args.max_jobs)])
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
