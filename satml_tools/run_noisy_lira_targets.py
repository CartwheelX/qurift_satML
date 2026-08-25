#!/usr/bin/env python3
"""Parallel target-side launcher for frozen-snapshot noisy LiRA scoring."""
from __future__ import annotations

import argparse
import queue
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REVIEWER = ROOT / "reviewer_tools"
for path in (ROOT, REVIEWER):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from reviewer_tools.gpu_scheduler import describe_gpu_plan, plan_gpu_slots
from reviewer_tools.run_lira_reference_multigpu import parse_gpus, run_commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--run-root", type=Path, default=Path("reviewer_runs"))
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--num-references", type=int, default=16)
    parser.add_argument("--modes", default="ideal_shot,noisy_shot")
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--simulator-seeds", default="0,1,2,3,4")
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--jobs-per-gpu", default="1")
    parser.add_argument(
        "--gpu-scheduling", choices=("adaptive", "fixed"), default="adaptive"
    )
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-targets", type=int, default=None)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    targets = (args.targets if args.targets.is_absolute() else repo / args.targets).resolve()
    run_root = (args.run_root if args.run_root.is_absolute() else repo / args.run_root).resolve()
    reference_dir = (
        args.reference_dir if args.reference_dir.is_absolute() else repo / args.reference_dir
    ).resolve()
    out_dir = (args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir).resolve()
    snapshot = (args.snapshot if args.snapshot.is_absolute() else repo / args.snapshot).resolve()
    table = pd.read_csv(targets)
    tasks = []
    worker = repo / "satml_tools" / "noisy_lira.py"
    for _, row in table.iterrows():
        target_id = str(row.target_id)
        command = [
            sys.executable, str(worker),
            "--repo-root", str(repo), "--targets", str(targets),
            "--run-root", str(run_root), "--reference-dir", str(reference_dir),
            "--out-dir", str(out_dir), "--snapshot", str(snapshot),
            "--target-id", target_id, "--num-references", str(args.num_references),
            "--modes", args.modes, "--shots", str(args.shots),
            "--simulator-seeds", args.simulator_seeds,
            "--device", "cuda",
        ]
        if args.resume:
            command.append("--resume")
        tasks.append(
            {
                "name": f"noisy_lira_{target_id}", "kind": "noisy_lira",
                "target_id": target_id,
                "structural_cell_id": str(row.get("structural_cell_id", "")),
                "reference_id": "", "command": command,
                "repo_root": str(repo), "cpu_threads": args.cpu_threads,
            }
        )
    if args.max_targets is not None:
        tasks = tasks[: args.max_targets]
    gpus = parse_gpus(args.gpus, dry_run=args.dry_run)
    plan = plan_gpu_slots(
        gpus,
        jobs_per_gpu=args.jobs_per_gpu,
        profile_name="noisy_lira",
        pending_jobs=len(tasks),
        adaptive=args.gpu_scheduling == "adaptive",
        dry_run=args.dry_run,
    )
    slots: queue.Queue[int] = queue.Queue()
    for gpu in plan.tickets:
        slots.put(gpu)
    print(describe_gpu_plan(plan), flush=True)
    status = run_commands(
        tasks,
        gpu_slots=slots,
        concurrency=plan.concurrency,
        logs_dir=out_dir / "logs",
        status_path=out_dir / "target_status.csv",
        dry_run=args.dry_run,
    )
    if not args.dry_run and (status.empty or not status.status.eq("ok").all()):
        raise SystemExit(f"Noisy LiRA failed; inspect {out_dir / 'target_status.csv'}")
    if not args.dry_run and args.max_targets is None:
        command = [
            sys.executable, str(worker), "--repo-root", str(repo),
            "--targets", str(targets), "--run-root", str(run_root),
            "--reference-dir", str(reference_dir), "--out-dir", str(out_dir),
            "--snapshot", str(snapshot), "--aggregate",
        ]
        raise SystemExit(subprocess.call(command, cwd=repo))


if __name__ == "__main__":
    main()
