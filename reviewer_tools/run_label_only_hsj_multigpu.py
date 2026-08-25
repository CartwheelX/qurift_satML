#!/usr/bin/env python3
"""Evaluate the QuRiFT hard-label HSJ MIA across multiple GPUs."""
from __future__ import annotations

import argparse
import queue
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from gpu_scheduler import describe_gpu_plan, plan_gpu_slots
from run_lira_reference_multigpu import parse_gpus, run_commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("reviewer_targets/multiseed_factorial_targets.csv"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--run-root", type=Path, default=Path("reviewer_runs"))
    parser.add_argument(
        "--target-id",
        action="append",
        default=[],
        help="Restrict scoring to one or more explicit target IDs (repeatable).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reviewer_results/label_only_hsj"),
    )
    parser.add_argument("--n-member", type=int, default=200)
    parser.add_argument("--n-nonmember", type=int, default=200)
    parser.add_argument("--max-queries", type=int, default=512)
    parser.add_argument("--init-queries", type=int, default=128)
    parser.add_argument("--init-batch-size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--gradient-samples", type=int, default=32)
    parser.add_argument("--binary-steps", type=int, default=10)
    parser.add_argument("--step-search-steps", type=int, default=10)
    parser.add_argument("--gradient-delta-ratio", type=float, default=0.1)
    parser.add_argument("--min-gradient-delta", type=float, default=1e-4)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--clip-min", type=float, default=None)
    parser.add_argument("--clip-max", type=float, default=None)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--jobs-per-gpu", default="1")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    args = parser.parse_args()

    if args.n_member < 1 or args.n_nonmember < 1:
        parser.error("--n-member and --n-nonmember must be positive")
    for name in (
        "max_queries",
        "init_queries",
        "init_batch_size",
        "gradient_samples",
        "binary_steps",
        "step_search_steps",
        "query_batch_size",
        "bootstrap",
        "cpu_threads",
    ):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.iterations < 0:
        parser.error("--iterations cannot be negative")
    if (args.clip_min is None) != (args.clip_max is None):
        parser.error("--clip-min and --clip-max must be supplied together")

    repo_root = args.repo_root.resolve()
    targets_path = (
        args.targets if args.targets.is_absolute() else repo_root / args.targets
    ).resolve()
    run_root = (
        args.run_root if args.run_root.is_absolute() else repo_root / args.run_root
    ).resolve()
    out_dir = (
        args.out_dir if args.out_dir.is_absolute() else repo_root / args.out_dir
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(targets_path)
    targets = targets[
        targets["architecture"].astype(str).str.lower().isin(["qnn", "hqnn", "qcnn"])
    ].copy()
    if args.target_id:
        requested = list(dict.fromkeys(str(value) for value in args.target_id))
        available = set(targets["target_id"].astype(str))
        missing = sorted(set(requested) - available)
        if missing:
            raise SystemExit(f"Requested target IDs are absent from the table: {missing}")
        order = {target_id: index for index, target_id in enumerate(requested)}
        targets = targets[targets["target_id"].astype(str).isin(requested)].copy()
        targets["_requested_order"] = targets["target_id"].astype(str).map(order)
        targets = targets.sort_values("_requested_order").drop(columns="_requested_order")
    if targets.empty:
        raise SystemExit("No supported targets found")
    worker = Path(__file__).with_name("qurift_label_only_hsj.py").resolve()
    tasks: list[dict[str, Any]] = []
    for _, row in targets.iterrows():
        target_id = str(row["target_id"])
        command = [
            sys.executable,
            str(worker),
            "score-target",
            "--repo-root",
            str(repo_root),
            "--targets",
            str(targets_path),
            "--run-root",
            str(run_root),
            "--out-dir",
            str(out_dir),
            "--target-id",
            target_id,
            "--n-member",
            str(args.n_member),
            "--n-nonmember",
            str(args.n_nonmember),
            "--max-queries",
            str(args.max_queries),
            "--init-queries",
            str(args.init_queries),
            "--init-batch-size",
            str(args.init_batch_size),
            "--iterations",
            str(args.iterations),
            "--gradient-samples",
            str(args.gradient_samples),
            "--binary-steps",
            str(args.binary_steps),
            "--step-search-steps",
            str(args.step_search_steps),
            "--gradient-delta-ratio",
            str(args.gradient_delta_ratio),
            "--min-gradient-delta",
            str(args.min_gradient_delta),
            "--query-batch-size",
            str(args.query_batch_size),
            "--bootstrap",
            str(args.bootstrap),
            "--seed",
            str(args.seed),
            "--device",
            "cuda",
        ]
        if args.clip_min is not None and args.clip_max is not None:
            command.extend(["--clip-min", str(args.clip_min), "--clip-max", str(args.clip_max)])
        if args.resume:
            command.append("--resume")
        tasks.append(
            {
                "name": f"label_only_hsj_{target_id}",
                "kind": "label_only_hsj",
                "target_id": target_id,
                "structural_cell_id": row.get("structural_cell_id", ""),
                "reference_id": "",
                "command": command,
                "repo_root": str(repo_root),
                "cpu_threads": args.cpu_threads,
            }
        )
    if args.max_jobs is not None:
        tasks = tasks[: args.max_jobs]

    gpus = parse_gpus(args.gpus, dry_run=args.dry_run)
    plan = plan_gpu_slots(
        gpus,
        jobs_per_gpu=args.jobs_per_gpu,
        profile_name="label_only_hsj",
        pending_jobs=len(tasks),
        adaptive=True,
        dry_run=args.dry_run,
    )
    concurrency = plan.concurrency
    slots: queue.Queue[int] = queue.Queue()
    for gpu in plan.tickets:
        slots.put(gpu)
    print(
        describe_gpu_plan(plan) + "\n" +
        f"GPUs={gpus}; jobs_per_gpu_max={args.jobs_per_gpu}; "
        f"concurrency={concurrency}; targets={len(tasks)}",
        flush=True,
    )
    status = run_commands(
        tasks,
        gpu_slots=slots,
        concurrency=concurrency,
        logs_dir=out_dir / "logs",
        status_path=out_dir / "target_scoring_status.csv",
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return
    if status.empty or not status["status"].eq("ok").all():
        raise SystemExit(
            f"Label-only HSJ evaluation failed; inspect {out_dir / 'target_scoring_status.csv'}"
        )
    if args.max_jobs is None:
        aggregate = [
            sys.executable,
            str(worker),
            "aggregate",
            "--repo-root",
            str(repo_root),
            "--targets",
            str(targets_path),
            "--run-root",
            str(run_root),
            "--out-dir",
            str(out_dir),
            "--bootstrap",
            str(args.bootstrap),
            "--seed",
            str(args.seed),
            "--device",
            "cpu",
        ]
        raise SystemExit(subprocess.call(aggregate, cwd=repo_root))


if __name__ == "__main__":
    main()
