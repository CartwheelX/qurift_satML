#!/usr/bin/env python3
"""Memory-aware GPU slot planning for independent QuRiFT workers."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import subprocess
from typing import Iterable, Mapping, Optional, Sequence, Union


@dataclass(frozen=True)
class GPUState:
    index: int
    total_mb: int
    used_mb: int
    utilization: int

    @property
    def free_mb(self) -> int:
        return max(0, self.total_mb - self.used_mb)


@dataclass(frozen=True)
class WorkloadProfile:
    estimated_job_mb: int
    reserve_mb: int
    max_jobs_per_gpu: int


@dataclass(frozen=True)
class GPUPlan:
    profile: str
    mode: str
    tickets: tuple[int, ...]
    capacities: Mapping[int, int]
    assigned: Mapping[int, int]
    states: Mapping[int, GPUState]

    @property
    def concurrency(self) -> int:
        return len(self.tickets)


PROFILES: Mapping[str, WorkloadProfile] = {
    # A measured six-wire target-training worker uses roughly 0.6 GiB and
    # 9--12% SM on this host. The 2 GiB allowance remains conservative.
    "qnn_train": WorkloadProfile(estimated_job_mb=2048, reserve_mb=4096, max_jobs_per_gpu=6),
    "lira": WorkloadProfile(estimated_job_mb=2048, reserve_mb=4096, max_jobs_per_gpu=6),
    "learned_mia": WorkloadProfile(estimated_job_mb=4096, reserve_mb=4096, max_jobs_per_gpu=3),
    # HSJ has shown substantially larger per-process allocations. Do not infer
    # its capacity from the lightweight target-training profile.
    "label_only_hsj": WorkloadProfile(
        estimated_job_mb=15360, reserve_mb=8192, max_jobs_per_gpu=3
    ),
    # Aer/noisy scoring can have workload-dependent allocations and is kept at
    # one process per GPU unless a separate benchmark justifies changing it.
    "noise": WorkloadProfile(estimated_job_mb=24576, reserve_mb=8192, max_jobs_per_gpu=1),
    # The CPU-only Aer wheel still uses CUDA for the small Torch model/head.
    # Multiple Aer processes are therefore safe and make use of host cores.
    "aer_cpu": WorkloadProfile(
        estimated_job_mb=2048, reserve_mb=4096, max_jobs_per_gpu=4
    ),
    "noisy_lira": WorkloadProfile(
        estimated_job_mb=24576, reserve_mb=8192, max_jobs_per_gpu=1
    ),
}


def query_gpu_states(gpu_ids: Sequence[int]) -> dict[int, GPUState]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    requested = set(int(value) for value in gpu_ids)
    states: dict[int, GPUState] = {}
    for line in output.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != 4:
            continue
        index, total, used, utilization = map(int, values)
        if index in requested:
            states[index] = GPUState(index, total, used, utilization)
    missing = sorted(requested - set(states))
    if missing:
        raise RuntimeError(f"nvidia-smi did not report requested GPUs: {missing}")
    return states


def _positive_environment_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _profile_with_environment(profile_name: str) -> WorkloadProfile:
    try:
        base = PROFILES[profile_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown GPU workload profile {profile_name!r}; expected {sorted(PROFILES)}"
        ) from exc
    prefix = "QURIFT_GPU_" + profile_name.upper()
    return WorkloadProfile(
        estimated_job_mb=_positive_environment_int(
            f"{prefix}_JOB_MEMORY_MB", base.estimated_job_mb
        ),
        reserve_mb=_positive_environment_int("QURIFT_GPU_RESERVE_MB", base.reserve_mb),
        max_jobs_per_gpu=_positive_environment_int(
            f"{prefix}_MAX_JOBS", base.max_jobs_per_gpu
        ),
    )


def _requested_limit(
    jobs_per_gpu: Union[str, int], profile: WorkloadProfile
) -> tuple[str, int]:
    text = str(jobs_per_gpu).strip().lower()
    if text == "auto":
        return "adaptive-auto", profile.max_jobs_per_gpu
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError("jobs_per_gpu must be a positive integer or 'auto'") from exc
    if value < 1:
        raise ValueError("jobs_per_gpu must be positive")
    return "adaptive-capped", value


def plan_gpu_slots(
    gpu_ids: Sequence[int],
    *,
    jobs_per_gpu: Union[str, int],
    profile_name: str,
    pending_jobs: Optional[int] = None,
    adaptive: bool = True,
    dry_run: bool = False,
    states: Optional[Mapping[int, GPUState]] = None,
) -> GPUPlan:
    """Return balanced per-GPU tickets, capped by live memory and utilization.

    Numeric ``jobs_per_gpu`` is treated as an upper bound. ``auto`` uses the
    workload profile's tested upper bound. Tickets are allocated round-robin so
    a short manifest does not unnecessarily stack work on its first GPU.
    """
    gpu_ids = list(dict.fromkeys(int(value) for value in gpu_ids))
    if not gpu_ids:
        raise ValueError("At least one GPU is required")
    profile = _profile_with_environment(profile_name)
    requested_mode, requested_limit = _requested_limit(jobs_per_gpu, profile)
    if pending_jobs is not None and pending_jobs < 0:
        raise ValueError("pending_jobs cannot be negative")
    if states is None:
        if dry_run:
            states = {
                gpu: GPUState(gpu, total_mb=81920, used_mb=0, utilization=0)
                for gpu in gpu_ids
            }
        else:
            states = query_gpu_states(gpu_ids)
    else:
        states = dict(states)

    capacities: dict[int, int] = {}
    for gpu in gpu_ids:
        state = states[gpu]
        if adaptive:
            memory_budget = max(0, state.free_mb - profile.reserve_mb)
            capacity = min(requested_limit, memory_budget // profile.estimated_job_mb)
            # Respect substantial activity from work that was already present
            # when this launcher started. At least one slot remains when memory
            # permits so an explicitly selected GPU is not silently discarded.
            if state.utilization >= 80:
                capacity = min(capacity, 1)
            elif state.utilization >= 50:
                capacity = min(capacity, max(1, math.ceil(requested_limit / 2)))
        else:
            capacity = requested_limit
        capacities[gpu] = max(0, int(capacity))

    available = sum(capacities.values())
    if available == 0 and (pending_jobs is None or pending_jobs > 0):
        details = ", ".join(
            f"gpu={gpu}:free={states[gpu].free_mb}MiB" for gpu in gpu_ids
        )
        raise RuntimeError(
            f"No safe GPU slots for profile={profile_name}; {details}. "
            "Free memory or adjust the documented QURIFT_GPU_* overrides."
        )
    ticket_limit = available if pending_jobs is None else min(available, pending_jobs)
    tickets: list[int] = []
    level = 0
    while len(tickets) < ticket_limit:
        added = False
        for gpu in gpu_ids:
            if capacities[gpu] > level and len(tickets) < ticket_limit:
                tickets.append(gpu)
                added = True
        if not added:
            break
        level += 1
    assigned = {gpu: tickets.count(gpu) for gpu in gpu_ids}
    mode = requested_mode if adaptive else "fixed"
    return GPUPlan(profile_name, mode, tuple(tickets), capacities, assigned, states)


def describe_gpu_plan(plan: GPUPlan) -> str:
    header = (
        f"[GPU PLAN] profile={plan.profile} mode={plan.mode} "
        f"concurrency={plan.concurrency}"
    )
    rows = [header]
    for gpu, state in plan.states.items():
        rows.append(
            f"[GPU PLAN] gpu={gpu} used={state.used_mb}MiB free={state.free_mb}MiB "
            f"util={state.utilization}% safe_capacity={plan.capacities[gpu]} "
            f"assigned_slots={plan.assigned[gpu]}"
        )
    return "\n".join(rows)
