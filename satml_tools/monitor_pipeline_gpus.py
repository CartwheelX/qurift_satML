#!/usr/bin/env python3
"""Record GPU telemetry until the unattended SaTML pipeline exits."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import signal
import subprocess
import time


STOP = False


def _stop(*_: object) -> None:
    global STOP
    STOP = True


def process_alive(pid: int) -> bool:
    try:
        Path(f"/proc/{pid}").stat()
    except (FileNotFoundError, PermissionError):
        return False
    return True


def read_pipeline_pid(pid_file: Path) -> int:
    text = pid_file.read_text(encoding="utf-8").strip()
    if not text.isdigit():
        raise ValueError(f"Invalid pipeline PID in {pid_file}: {text!r}")
    return int(text)


def gpu_rows() -> list[dict[str, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu,"
            "utilization.memory,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != 7:
            continue
        rows.append(
            {
                "gpu": int(values[0]),
                "memory_used_mb": int(values[1]),
                "memory_total_mb": int(values[2]),
                "gpu_util_percent": int(values[3]),
                "memory_util_percent": int(values[4]),
                "power_watts": round(float(values[5]), 2),
                "temperature_c": int(values[6]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pid-file",
        type=Path,
        default=Path("satml_results/unattended_pipeline/pipeline.pid"),
    )
    parser.add_argument(
        "--stage-file",
        type=Path,
        default=Path("satml_results/unattended_pipeline/current_stage.txt"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument(
        "--samples", type=int, default=0, help="Zero records until the pipeline exits."
    )
    args = parser.parse_args()
    if args.interval <= 0 or args.samples < 0:
        parser.error("--interval must be positive and --samples non-negative")
    pipeline_pid = read_pipeline_pid(args.pid_file)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp_utc",
        "pipeline_pid",
        "stage",
        "gpu",
        "memory_used_mb",
        "memory_total_mb",
        "gpu_util_percent",
        "memory_util_percent",
        "power_watts",
        "temperature_c",
    ]
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    write_header = not args.out.exists() or args.out.stat().st_size == 0
    sample = 0
    with args.out.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        while not STOP and process_alive(pipeline_pid):
            stage = (
                args.stage_file.read_text(encoding="utf-8").strip()
                if args.stage_file.exists()
                else "unknown"
            )
            timestamp = datetime.now(timezone.utc).isoformat()
            try:
                rows = gpu_rows()
            except (OSError, subprocess.SubprocessError, ValueError):
                rows = []
            for row in rows:
                writer.writerow(
                    {
                        "timestamp_utc": timestamp,
                        "pipeline_pid": pipeline_pid,
                        "stage": stage,
                        **row,
                    }
                )
            handle.flush()
            sample += 1
            if args.samples and sample >= args.samples:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
