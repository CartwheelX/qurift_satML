#!/usr/bin/env python3
"""DGX-aware launcher for QuRiFT reviewer target tables.

Each target is an independent single-GPU process. The launcher supports multiple
GPU slots, deterministic model/data seeds, resume, incremental status files,
unique output directories, and failure logs.
"""
from __future__ import annotations

import argparse
import os
import queue
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from gpu_scheduler import describe_gpu_plan, plan_gpu_slots
from reviewer_common import as_bool, atomic_write_csv, safe_int


def detect_gpus() -> List[int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        text=True,
    )
    gpu_ids = [int(value.strip()) for value in output.splitlines() if value.strip()]
    if not gpu_ids:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return gpu_ids


def parse_gpus(value: str) -> List[int]:
    if value.strip().lower() == "auto":
        return detect_gpus()
    gpu_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must be 'auto' or a comma-separated list")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"Duplicate GPU IDs: {gpu_ids}")
    return gpu_ids


def text_value(row: pd.Series, key: str, default: str) -> str:
    value = row.get(key, default)
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    return default if text.lower() in {"", "nan", "none"} else text


def build_command(
    row: pd.Series,
    script: Path,
    out_root: Path,
) -> Tuple[List[str], Path, Path, Path, Path]:
    target_id = str(row["target_id"])
    experiment = text_value(row, "experiment", "reviewer")
    output_dir = out_root / experiment / target_id
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = (output_dir / "target_model.pt").resolve()
    attack_path = (output_dir / "target_attack_data.pt").resolve()
    metrics_json = (output_dir / "target_export_summary.json").resolve()
    preprocessor_path = (output_dir / "dataset_preprocessor.joblib").resolve()
    provenance_path = (output_dir / "dataset_provenance.json").resolve()
    log_path = output_dir / "train.log"

    model_seed = safe_int(row.get("model_seed", row.get("seed", 43)), 43)
    data_seed = safe_int(row.get("data_seed", 43), 43)
    dataset = text_value(row, "dataset", "mnist").lower()
    architecture = text_value(row, "architecture", "qnn").lower()

    command = [
        sys.executable,
        str(script.resolve()),
        "--model-type",
        architecture,
        "--dataset",
        dataset,
        "--target-id",
        target_id,
        "--experiment-id",
        experiment,
        "--run-id",
        str(safe_int(row.get("source_run_id", -1), -1)),
        "--model-seed",
        str(model_seed),
        "--data-seed",
        str(data_seed),
        "--random-ops",
        "0",
        "--vector-train",
        str(safe_int(row.get("vector_train", 200), 200)),
        "--vector-valid",
        str(safe_int(row.get("vector_valid", 200), 200)),
        "--vector-test",
        str(safe_int(row.get("vector_test", 200), 200)),
        "--batch-size",
        str(safe_int(row.get("batch_size", 16), 16)),
        "--epochs",
        str(safe_int(row.get("epochs", 100), 100)),
        "--n-wires",
        str(safe_int(row["n_wires"])),
        "--depth",
        str(safe_int(row["depth"])),
        "--qlayer-ent-kind",
        text_value(row, "ql_ent", "linear"),
        "--qlayer-twoq-op",
        text_value(row, "ql_op", "crz"),
        "--fm-kind",
        text_value(row, "fm_kind", "z").lower(),
        "--train_target",
        "--export-attack-data",
        "--attack-feature-mode",
        "pv+stats",
        "--target-model-path",
        str(model_path),
        "--attack-data-out",
        str(attack_path),
        "--attack-metrics-out",
        str(metrics_json),
    ]

    if dataset == "credit_default":
        command += [
            "--credit-data-path",
            text_value(
                row,
                "credit_data_path",
                "data/credit_default/credit_default.csv.gz",
            ),
            "--credit-pca-components",
            str(safe_int(row.get("credit_pca_components", row.get("n_wires", 6)), 6)),
            "--preprocessor-out",
            str(preprocessor_path),
            "--dataset-provenance-out",
            str(provenance_path),
        ]
    elif dataset == "breast_cancer_wdbc":
        command += [
            "--wdbc-data-path",
            text_value(row, "wdbc_data_path", "data/wdbc/wdbc.csv.gz"),
            "--wdbc-pca-components",
            str(safe_int(row.get("wdbc_pca_components", row.get("n_wires", 6)), 6)),
            "--preprocessor-out",
            str(preprocessor_path),
            "--dataset-provenance-out",
            str(provenance_path),
        ]
    elif dataset == "fashion_mnist":
        command += ["--dataset-provenance-out", str(provenance_path)]

    learning_rate = row.get("learning_rate", None)
    try:
        learning_rate_available = learning_rate is not None and not pd.isna(learning_rate)
    except Exception:
        learning_rate_available = False
    if learning_rate_available:
        command += ["--learning-rate", str(float(learning_rate))]
    weight_decay = row.get("weight_decay", 0.0)
    try:
        weight_decay = 0.0 if pd.isna(weight_decay) else float(weight_decay)
    except Exception:
        weight_decay = 0.0
    command += ["--weight-decay", str(weight_decay)]

    if as_bool(row.get("ql_rev", False)):
        command.append("--qlayer-ent-wire-reverse")
    if as_bool(row.get("extra_feats", False)):
        command.append("--extra-feats")

    if dataset == "moons":
        command += ["--moons-noise", "0.3"]
    elif dataset == "circles":
        command += ["--circles-noise", "0.3"]
    elif dataset == "blobs":
        command += [
            "--blobs-n-features",
            "4",
            "--blobs-cluster-std",
            "2.1",
            "--blobs-center-distance",
            "3.5",
        ]

    feature_map = text_value(row, "fm_kind", "z").lower()
    repetitions = str(safe_int(row.get("reps", 1), 1))
    padding = text_value(row, "pad_mode", "wrap")
    feature_entanglement = text_value(row, "fm_ent", "linear")
    feature_gate = text_value(row, "fm_op", "cx")
    feature_angle_scale_raw = row.get("feature_angle_scale", 1.0)
    try:
        feature_angle_scale = 1.0 if pd.isna(feature_angle_scale_raw) else float(feature_angle_scale_raw)
    except Exception:
        feature_angle_scale = 1.0

    if feature_map == "z":
        command += [
            "--fm-z-reps", repetitions,
            "--fm-z-pad-mode", padding,
            "--fm-z-alpha", str(feature_angle_scale),
        ]
    elif feature_map == "zz":
        command += [
            "--fm-zz-reps",
            repetitions,
            "--fm-zz-pad-mode",
            padding,
            "--fm-zz-entanglement",
            feature_entanglement,
            "--fm-zz-alpha",
            str(feature_angle_scale),
        ]
    elif feature_map == "eff_su2":
        command += [
            "--fm-eff-reps",
            repetitions,
            "--fm-eff-pad-mod",
            padding,
            "--fm-eff-ent-kind",
            feature_entanglement,
            "--fm-eff-twoq-op",
            feature_gate,
            "--fm-eff-alpha",
            str(feature_angle_scale),
        ]
    elif feature_map == "pauli":
        command += [
            "--fm-pauli-reps",
            repetitions,
            "--fm-pauli-pad",
            padding,
            "--fm-pauli-entanglement",
            feature_entanglement,
            "--fm-pauli-alpha",
            str(feature_angle_scale),
        ]
    else:
        raise ValueError(f"Unsupported feature map: {feature_map}")

    extra = text_value(row, "extra_cli_args", "")
    if extra:
        command.extend(shlex.split(extra))
    return command, log_path, model_path, attack_path, metrics_json


def estimated_cost(row: pd.Series) -> float:
    wires = max(1, safe_int(row.get("n_wires", 1), 1))
    depth = max(1, safe_int(row.get("depth", 1), 1))
    reps = max(1, safe_int(row.get("reps", 1), 1))
    epochs = max(1, safe_int(row.get("epochs", 1), 1))
    batch = max(1, safe_int(row.get("batch_size", 1), 1))
    return float((2**wires) * depth * reps * epochs * batch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--script", type=Path, default=Path("experiments/qurift_main.py")
    )
    parser.add_argument("--out", type=Path, default=Path("reviewer_runs"))
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--jobs-per-gpu", default="2")
    parser.add_argument(
        "--gpu-scheduling", choices=("adaptive", "fixed"), default="adaptive"
    )
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument(
        "--largest-first",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if not args.targets.exists():
        parser.error(f"Targets file not found: {args.targets}")

    repo_root = args.repo_root.resolve()
    script = args.script if args.script.is_absolute() else repo_root / args.script
    if not script.exists():
        parser.error(f"QuRiFT driver not found: {script}")

    targets = pd.read_csv(args.targets)
    required = {
        "target_id",
        "dataset",
        "architecture",
        "fm_kind",
        "n_wires",
        "reps",
        "depth",
    }
    missing = required - set(targets.columns)
    if missing:
        parser.error(f"Target table missing columns: {sorted(missing)}")
    if targets["target_id"].astype(str).duplicated().any():
        parser.error("target_id values must be unique")

    if args.largest_first:
        targets = targets.copy()
        targets["_estimated_cost"] = targets.apply(estimated_cost, axis=1)
        targets = targets.sort_values("_estimated_cost", ascending=False)
    if args.max_jobs is not None:
        targets = targets.head(max(0, args.max_jobs))

    args.out.mkdir(parents=True, exist_ok=True)
    status_path = args.out / f"{args.targets.stem}_run_status.csv"
    failure_path = args.out / f"{args.targets.stem}_failures.csv"

    if args.dry_run:
        for _, row in targets.iterrows():
            command, *_ = build_command(row, script, args.out)
            print(shlex.join(command))
        print(f"[DRY RUN] {len(targets)} commands")
        return

    gpu_ids = parse_gpus(args.gpus)
    plan = plan_gpu_slots(
        gpu_ids,
        jobs_per_gpu=args.jobs_per_gpu,
        profile_name="qnn_train",
        pending_jobs=len(targets),
        adaptive=args.gpu_scheduling == "adaptive",
        dry_run=args.dry_run,
    )
    slots: queue.Queue[int] = queue.Queue()
    for gpu_id in plan.tickets:
        slots.put(gpu_id)
    workers = plan.concurrency
    print(describe_gpu_plan(plan), flush=True)

    results: List[Dict[str, object]] = []

    def run(row: pd.Series) -> Dict[str, object]:
        target_id = str(row["target_id"])
        command, log_path, model_path, attack_path, metrics_json = build_command(
            row, script, args.out
        )
        base = {
            "target_id": target_id,
            "experiment": text_value(row, "experiment", "reviewer"),
            "model_seed": safe_int(row.get("model_seed", row.get("seed", 43)), 43),
            "data_seed": safe_int(row.get("data_seed", 43), 43),
            "command": shlex.join(command),
            "log_path": str(log_path),
            "model_path": str(model_path),
            "attack_path": str(attack_path),
            "metrics_json": str(metrics_json),
        }
        preprocessor_path = model_path.parent / "dataset_preprocessor.joblib"
        provenance_path = model_path.parent / "dataset_provenance.json"
        complete = (
            model_path.exists()
            and model_path.stat().st_size > 0
            and attack_path.exists()
            and attack_path.stat().st_size > 0
        )
        dataset_name = text_value(row, "dataset", "mnist").lower()
        if dataset_name in {"credit_default", "breast_cancer_wdbc"}:
            complete = (
                complete
                and preprocessor_path.exists()
                and preprocessor_path.stat().st_size > 0
                and provenance_path.exists()
                and provenance_path.stat().st_size > 0
            )
        elif dataset_name == "fashion_mnist":
            complete = (
                complete
                and provenance_path.exists()
                and provenance_path.stat().st_size > 0
            )
        if args.resume and complete:
            return {
                **base,
                "status": "skipped",
                "gpu": "",
                "return_code": 0,
                "seconds": 0.0,
                "error": "",
            }

        gpu_id = slots.get()
        started = time.time()
        try:
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": str(gpu_id),
                    "OMP_NUM_THREADS": str(args.cpu_threads),
                    "MKL_NUM_THREADS": str(args.cpu_threads),
                    "OPENBLAS_NUM_THREADS": str(args.cpu_threads),
                    "NUMEXPR_NUM_THREADS": str(args.cpu_threads),
                    "PYTHONUNBUFFERED": "1",
                    "QURIFT_JOB_ID": target_id,
                    "QURIFT_DISABLE_DEBUG_EXPORTS": "1",
                    "QURIFT_DISABLE_CIRCUIT_EXPORTS": "1",
                }
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.run(
                    command,
                    cwd=repo_root,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    check=False,
                )
            model_ok = model_path.exists() and model_path.stat().st_size > 0
            attack_ok = attack_path.exists() and attack_path.stat().st_size > 0
            success = process.returncode == 0 and model_ok and attack_ok
            error = ""
            if process.returncode != 0:
                error = f"nonzero_exit={process.returncode}"
            elif not model_ok:
                error = "missing_or_empty_model"
            elif not attack_ok:
                error = "missing_or_empty_attack_payload"
            return {
                **base,
                "status": "ok" if success else "error",
                "gpu": gpu_id,
                "return_code": process.returncode,
                "seconds": round(time.time() - started, 3),
                "error": error,
            }
        except Exception as exc:
            return {
                **base,
                "status": "error",
                "gpu": gpu_id,
                "return_code": -1,
                "seconds": round(time.time() - started, 3),
                "error": repr(exc),
            }
        finally:
            slots.put(gpu_id)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run, row) for _, row in targets.iterrows()]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            atomic_write_csv(pd.DataFrame(results), status_path)
            atomic_write_csv(
                pd.DataFrame([row for row in results if row["status"] == "error"]),
                failure_path,
            )
            print(
                f"[{index}/{len(futures)}] {result['target_id']} -> "
                f"{result['status']} (gpu={result['gpu']}, sec={result['seconds']})"
            )
            if args.fail_fast and result["status"] == "error":
                for pending in futures:
                    pending.cancel()
                break

    errors = sum(result["status"] == "error" for result in results)
    print(f"[DONE] status={status_path.resolve()} errors={errors}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
