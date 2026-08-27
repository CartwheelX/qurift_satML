#!/usr/bin/env python3
"""GPU-aware launcher for defended HSJ, LiRA, and nearby-query attacks."""
from __future__ import annotations

import argparse
import queue
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "reviewer_tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from gpu_scheduler import describe_gpu_plan, plan_gpu_slots  # noqa: E402
from run_lira_reference_multigpu import parse_gpus, run_commands  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", choices=("hsj", "lira", "query_stress"), required=True)
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("pets_targets/credit_defense_training_targets.csv"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--run-root", type=Path, default=Path("pets_runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("pets_results/defenses"))
    parser.add_argument("--reference-dir", type=Path, default=Path("pets_results/lira_references"))
    parser.add_argument("--defenses", default="none,dynanoise,memgq_lattice,memgq_lattice_sticky")
    parser.add_argument(
        "--status-label",
        default="",
        help="Optional status filename label when one attack is launched in multiple groups.",
    )
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--jobs-per-gpu", default="1")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--archive-existing-query-stress",
        action="store_true",
        help=(
            "Archive each existing nearby-query metrics file before recomputing it; "
            "valid only with --attack query_stress."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--num-references", type=int, default=16)
    parser.add_argument("--mc-samples", type=int, default=10)
    parser.add_argument("--hsj-records-per-class", type=int, default=20)
    parser.add_argument("--max-queries", type=int, default=512)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--radius", type=float, default=0.005)
    args = parser.parse_args()
    if args.archive_existing_query_stress and args.attack != "query_stress":
        parser.error("--archive-existing-query-stress requires --attack query_stress")
    repo_root = args.repo_root.resolve()
    targets_path = args.targets if args.targets.is_absolute() else repo_root / args.targets
    run_root = args.run_root if args.run_root.is_absolute() else repo_root / args.run_root
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo_root / args.out_dir
    reference_dir = (
        args.reference_dir if args.reference_dir.is_absolute() else repo_root / args.reference_dir
    )
    targets = pd.read_csv(targets_path)
    defenses = [value.strip() for value in args.defenses.split(",") if value.strip()]
    tasks = []
    if args.attack == "lira":
        targets = targets[targets.training_defense.astype(str).isin(["none", "l2"])]
    defense_jobs = [",".join(defenses)] if args.attack == "query_stress" else defenses
    for _, row in targets.iterrows():
        for defense in defense_jobs:
            target_id = str(row.target_id)
            if args.attack == "hsj":
                worker = repo_root / "pets_tools" / "run_defense_hsj.py"
                extra = [
                    "--defense",
                    defense,
                    "--hsj-records-per-class",
                    str(args.hsj_records_per_class),
                    "--max-queries",
                    str(args.max_queries),
                ]
                profile = "label_only_hsj"
            elif args.attack == "lira":
                worker = repo_root / "pets_tools" / "score_defended_lira.py"
                extra = [
                    "--defense",
                    defense,
                    "--reference-dir",
                    str(reference_dir),
                    "--num-references",
                    str(args.num_references),
                    "--mc-samples",
                    str(args.mc_samples),
                ]
                profile = "lira"
            else:
                worker = repo_root / "pets_tools" / "run_query_stress.py"
                extra = [
                    "--defenses",
                    defense,
                    "--queries",
                    str(args.queries),
                    "--radius",
                    str(args.radius),
                ]
                if args.archive_existing_query_stress:
                    extra.append("--archive-existing")
                profile = "learned_mia"
            command = [
                sys.executable,
                str(worker),
                "--repo-root",
                str(repo_root),
                "--targets",
                str(targets_path),
                "--target-id",
                target_id,
                "--run-root",
                str(run_root),
                "--out-dir",
                str(out_dir),
                "--device",
                "cuda",
                *extra,
            ]
            if args.resume:
                command.append("--resume")
            tasks.append(
                {
                    "name": f"{args.attack}_{target_id}_{defense.replace(',', '-')}",
                    "kind": f"pets_{args.attack}",
                    "target_id": target_id,
                    "structural_cell_id": str(row.get("structural_cell_id", "")),
                    "command": command,
                    "repo_root": str(repo_root),
                    "cpu_threads": args.cpu_threads,
                }
            )
    if args.max_jobs is not None:
        tasks = tasks[: args.max_jobs]
    if not tasks:
        raise SystemExit("No compatible adaptive-attack jobs")
    gpus = parse_gpus(args.gpus, dry_run=args.dry_run)
    plan = plan_gpu_slots(
        gpus,
        jobs_per_gpu=args.jobs_per_gpu,
        profile_name=profile,
        pending_jobs=len(tasks),
        adaptive=True,
        dry_run=args.dry_run,
    )
    slots: queue.Queue[int] = queue.Queue()
    for gpu in plan.tickets:
        slots.put(gpu)
    print(describe_gpu_plan(plan), flush=True)
    status_label = args.status_label.strip() or args.attack
    if re.fullmatch(r"[A-Za-z0-9_-]+", status_label) is None:
        parser.error("--status-label must contain only letters, digits, '_' or '-'")
    status = run_commands(
        tasks,
        gpu_slots=slots,
        concurrency=plan.concurrency,
        logs_dir=repo_root / "pets_logs" / args.attack,
        status_path=out_dir / f"{status_label}_status.csv",
        dry_run=args.dry_run,
    )
    if not args.dry_run and (status.empty or not status.status.eq("ok").all()):
        raise SystemExit(f"PETS {args.attack} jobs failed; inspect status CSV")


if __name__ == "__main__":
    main()
