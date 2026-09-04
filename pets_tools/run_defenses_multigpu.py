#!/usr/bin/env python3
"""GPU-aware launcher for PETS training and prediction-defense evaluations."""
from __future__ import annotations

import argparse
import queue
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "reviewer_tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from gpu_scheduler import describe_gpu_plan, plan_gpu_slots  # noqa: E402
from run_lira_reference_multigpu import parse_gpus, run_commands  # noqa: E402


RAW_OUTPUT_DEFENSES = (
    "none,dynanoise,hamp_output,memguard,logitguard_continuous,"
    "logitguard_quantized,measurementguard_continuous,lattice_round,"
    "memgq_lattice,memgq_lattice_sticky"
)


def slot_plan(args, *, profile: str, jobs: int):
    gpus = parse_gpus(args.gpus, dry_run=args.dry_run)
    plan = plan_gpu_slots(
        gpus,
        jobs_per_gpu=args.jobs_per_gpu,
        profile_name=profile,
        pending_jobs=jobs,
        adaptive=True,
        dry_run=args.dry_run,
    )
    slots: queue.Queue[int] = queue.Queue()
    for gpu in plan.tickets:
        slots.put(gpu)
    print(describe_gpu_plan(plan), flush=True)
    return plan, slots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("pets_targets/credit_defense_training_targets.csv"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--run-root", type=Path, default=Path("pets_runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("pets_results/defenses"))
    parser.add_argument("--logs-dir", type=Path, default=Path("pets_logs"))
    parser.add_argument("--phase", choices=("all", "train", "evaluate"), default="all")
    parser.add_argument("--block-id", action="append", default=[])
    parser.add_argument("--target-id", action="append", default=[])
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--jobs-per-gpu", default="1")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--optimizer-iterations", type=int, default=30)
    parser.add_argument("--discriminator-epochs", type=int, default=100)
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--evaluation-nonmember-multiplier", type=int, default=1)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    targets_path = args.targets if args.targets.is_absolute() else repo_root / args.targets
    run_root = args.run_root if args.run_root.is_absolute() else repo_root / args.run_root
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo_root / args.out_dir
    logs_dir = args.logs_dir if args.logs_dir.is_absolute() else repo_root / args.logs_dir
    targets = pd.read_csv(targets_path)
    if args.block_id:
        targets = targets[targets.block_id.astype(str).isin(args.block_id)]
    if args.target_id:
        targets = targets[targets.target_id.astype(str).isin(args.target_id)]
    if targets.empty:
        raise SystemExit("No PETS targets remain after filtering")
    if args.max_jobs is not None:
        targets = targets.head(args.max_jobs)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase in {"all", "train"}:
        worker = repo_root / "pets_tools" / "train_defended_target.py"
        tasks = []
        for _, row in targets.iterrows():
            target_id = str(row.target_id)
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
                "--device",
                "cuda",
            ]
            if args.resume:
                command.append("--resume")
            tasks.append(
                {
                    "name": f"train_{target_id}",
                    "kind": "pets_train",
                    "target_id": target_id,
                    "structural_cell_id": str(row.get("structural_cell_id", "")),
                    "command": command,
                    "repo_root": str(repo_root),
                    "cpu_threads": args.cpu_threads,
                }
            )
        plan, slots = slot_plan(args, profile="qnn_train", jobs=len(tasks))
        status = run_commands(
            tasks,
            gpu_slots=slots,
            concurrency=plan.concurrency,
            logs_dir=logs_dir / "training",
            status_path=out_dir / "training_status.csv",
            dry_run=args.dry_run,
        )
        if not args.dry_run and (status.empty or not status.status.eq("ok").all()):
            raise SystemExit("PETS target training failed; inspect pets_results training status")

    if args.phase in {"all", "evaluate"}:
        worker = repo_root / "pets_tools" / "run_defense_evaluation.py"
        tasks = []
        for _, row in targets.iterrows():
            target_id = str(row.target_id)
            training_defense = str(row.get("training_defense", "none"))
            if training_defense == "none":
                defenses = RAW_OUTPUT_DEFENSES
            elif training_defense == "hamp_train":
                defenses = "none,hamp_output"
            else:
                defenses = "none"
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
                "--defenses",
                defenses,
                "--optimizer-iterations",
                str(args.optimizer_iterations),
                "--discriminator-epochs",
                str(args.discriminator_epochs),
                "--shots",
                str(args.shots),
                "--evaluation-nonmember-multiplier",
                str(args.evaluation_nonmember_multiplier),
            ]
            if args.resume:
                command.append("--resume")
            tasks.append(
                {
                    "name": f"evaluate_{target_id}",
                    "kind": "pets_evaluate",
                    "target_id": target_id,
                    "structural_cell_id": str(row.get("structural_cell_id", "")),
                    "command": command,
                    "repo_root": str(repo_root),
                    "cpu_threads": args.cpu_threads,
                }
            )
        plan, slots = slot_plan(args, profile="learned_mia", jobs=len(tasks))
        status = run_commands(
            tasks,
            gpu_slots=slots,
            concurrency=plan.concurrency,
            logs_dir=logs_dir / "evaluation",
            status_path=out_dir / "evaluation_status.csv",
            dry_run=args.dry_run,
        )
        if not args.dry_run and (status.empty or not status.status.eq("ok").all()):
            raise SystemExit("PETS defense evaluation failed; inspect evaluation status")


if __name__ == "__main__":
    main()
