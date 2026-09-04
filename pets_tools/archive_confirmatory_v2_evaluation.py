#!/usr/bin/env python3
"""Recoverably archive PETS-v2 evaluation artifacts before a corrected rerun.

The target checkpoints and LiRA reference banks are protected inputs.  This
tool moves only target result directories, non-training launcher status files,
and (when requested) Stage-4/6 worker logs.  Each rename is atomic on the local
filesystem and a write-ahead manifest makes an interrupted transaction
auditable and recoverable.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable


PROTOCOL = "pets_confirmatory_v2_evaluation_refresh_archive_v1"
EXPECTED_STAGE4_STATUSES = {
    "evaluation_status.csv",
    "hsj_output_boundary_conditions_status.csv",
    "hsj_training_defenses_status.csv",
    "query_stress_matched_controls_status.csv",
}
WORKER_LOG_DIRS = ("evaluation", "hsj", "query_stress", "lira")
STAGE_LAUNCHER_LOGS = ("stage4_evaluation.log", "stage6_lira_scoring.log")
WORKER_NAMES = (
    "run_defenses_multigpu.py",
    "run_adaptive_attacks_multigpu.py",
    "run_lira_reference_multigpu.py",
    "train_defended_target.py",
    "run_defense_evaluation.py",
    "run_defense_hsj.py",
    "run_query_stress.py",
    "score_defended_lira.py",
    "qurift_lira_attack.py",
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_fingerprint(path: Path) -> dict[str, Any]:
    """Return a content-addressed, deterministic fingerprint for a file/tree."""

    if path.is_symlink():
        raise ValueError(f"symbolic links are not allowed in archive inputs: {path}")
    if path.is_file():
        return {
            "sha256": file_sha256(path),
            "files": 1,
            "directories": 0,
            "bytes": path.stat().st_size,
        }
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = 0
    directories = 0
    total_bytes = 0
    for item in sorted(path.rglob("*"), key=lambda value: value.relative_to(path).as_posix()):
        relative = item.relative_to(path).as_posix()
        if item.is_symlink():
            raise ValueError(f"symbolic links are not allowed in archive inputs: {item}")
        if item.is_dir():
            directories += 1
            digest.update(b"D\0" + relative.encode("utf-8") + b"\0")
            continue
        if not item.is_file():
            raise ValueError(f"unsupported filesystem entry: {item}")
        item_digest = file_sha256(item)
        size = item.stat().st_size
        files += 1
        total_bytes += size
        digest.update(
            b"F\0"
            + relative.encode("utf-8")
            + b"\0"
            + str(size).encode("ascii")
            + b"\0"
            + item_digest.encode("ascii")
            + b"\0"
        )
    return {
        "sha256": digest.hexdigest(),
        "files": files,
        "directories": directories,
        "bytes": total_bytes,
    }


def target_ids(path: Path, expected: int) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "target_id" not in rows[0]:
        raise ValueError(f"target manifest has no target_id column: {path}")
    values = [str(row["target_id"]).strip() for row in rows]
    if any(not value for value in values):
        raise ValueError("target manifest contains an empty target_id")
    if len(values) != len(set(values)):
        raise ValueError("target manifest contains duplicate target IDs")
    if len(values) != expected:
        raise ValueError(f"expected {expected} targets, found {len(values)}")
    return values


def active_v2_processes() -> list[dict[str, Any]]:
    active = []
    own_pid = os.getpid()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == own_pid:
            continue
        try:
            raw = (proc / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        if not command:
            continue
        is_launcher = (
            "pets_run_credit_confirmatory_v2.sh" in command
            or "commands/pets_finalize.sh" in command
        )
        is_worker = "pets_v2" in command and any(name in command for name in WORKER_NAMES)
        if is_launcher or is_worker:
            active.append({"pid": int(proc.name), "command": command})
    return sorted(active, key=lambda item: item["pid"])


def acquire_stage_locks(log_root: Path) -> list[Any]:
    """Hold every launcher stage lock until this process exits."""

    log_root.mkdir(parents=True, exist_ok=True)
    handles = []
    try:
        for stage in range(1, 8):
            lock_path = log_root / f".confirmatory_v2_stage_{stage}.lock"
            handle = lock_path.open("a", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                owner_path = Path(str(lock_path) + ".owner")
                owner = owner_path.read_text().strip().replace("\n", " ") if owner_path.exists() else "unknown"
                handle.close()
                raise RuntimeError(
                    f"PETS v2 stage {stage} is active (owner: {owner})"
                ) from error
            handles.append(handle)
    except Exception:
        for handle in handles:
            handle.close()
        raise
    return handles


def nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            raise FileNotFoundError(f"no existing parent for {path}")
        current = current.parent
    return current


def make_entry(source: Path, destination: Path, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "fingerprint": path_fingerprint(source),
        "status": "planned",
    }


def protected_inventory(
    run_root: Path,
    reference_root: Path,
    training_status: Path,
    *,
    expected_targets: int,
    expected_references: int,
) -> dict[str, Any]:
    model_paths = sorted(run_root.rglob("target_model.pt"))
    reference_scores = sorted(reference_root.rglob("reference_*.npz"))
    reference_checkpoints = sorted(reference_root.rglob("reference_*.pt"))
    if len(model_paths) != expected_targets:
        raise ValueError(
            f"protected run tree has {len(model_paths)} target checkpoints; expected {expected_targets}"
        )
    if len(reference_scores) != expected_references:
        raise ValueError(
            f"protected reference tree has {len(reference_scores)} score files; expected {expected_references}"
        )
    if len(reference_checkpoints) != expected_references:
        raise ValueError(
            "protected reference tree has "
            f"{len(reference_checkpoints)} checkpoints; expected {expected_references}"
        )
    if not training_status.is_file():
        raise FileNotFoundError(f"protected training status is missing: {training_status}")
    return {
        "run_root": str(run_root.resolve()),
        "run_tree": path_fingerprint(run_root),
        "target_checkpoint_count": len(model_paths),
        "reference_root": str(reference_root.resolve()),
        "reference_tree": path_fingerprint(reference_root),
        "reference_score_count": len(reference_scores),
        "reference_checkpoint_count": len(reference_checkpoints),
        "training_status": str(training_status.resolve()),
        "training_status_fingerprint": path_fingerprint(training_status),
    }


def build_plan(args: argparse.Namespace, stamp: str) -> tuple[dict[str, Any], Path]:
    targets_path = args.targets.resolve()
    result_root = args.result_root.resolve()
    run_root = args.run_root.resolve()
    reference_root = args.reference_root.resolve()
    log_root = args.log_root.resolve()
    archive_root = args.archive_root.resolve()
    ids = target_ids(targets_path, args.expected_targets)
    if not result_root.is_dir():
        raise FileNotFoundError(result_root)
    target_sources = [result_root / target_id for target_id in ids]
    missing = [str(path) for path in target_sources if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} target result directories are missing; first: {missing[0]}"
        )
    extra = sorted(
        path.name
        for path in result_root.iterdir()
        if path.is_dir() and path.name.startswith("PETSV2_") and path.name not in set(ids)
    )
    if extra:
        raise ValueError(f"unexpected PETSV2 result directories would remain: {extra}")

    training_status = result_root / "training_status.csv"
    status_sources = sorted(
        path
        for path in result_root.glob("*_status.csv")
        if path.name != training_status.name
    )
    names = {path.name for path in status_sources}
    missing_statuses = sorted(EXPECTED_STAGE4_STATUSES - names)
    if missing_statuses:
        raise FileNotFoundError(
            f"required Stage-4 status files are missing: {missing_statuses}"
        )

    destination_root = archive_root / stamp
    entries = [
        make_entry(source, destination_root / "defenses" / source.name, "target_result")
        for source in target_sources
    ]
    entries.extend(
        make_entry(
            source,
            destination_root / "defenses" / "status_files" / source.name,
            "non_training_status",
        )
        for source in status_sources
    )
    if args.archive_worker_logs:
        for name in WORKER_LOG_DIRS:
            source = log_root / name
            if source.exists():
                entries.append(
                    make_entry(source, destination_root / "worker_logs" / name, "worker_log_tree")
                )
        for name in STAGE_LAUNCHER_LOGS:
            source = log_root / name
            if source.exists():
                entries.append(
                    make_entry(
                        source,
                        destination_root / "worker_logs" / source.name,
                        "stage_launcher_log",
                    )
                )

    destinations = [entry["destination"] for entry in entries]
    if len(destinations) != len(set(destinations)):
        raise ValueError("archive plan contains duplicate destinations")
    for entry in entries:
        if Path(entry["destination"]).exists():
            raise FileExistsError(entry["destination"])

    protected = protected_inventory(
        run_root,
        reference_root,
        training_status,
        expected_targets=args.expected_targets,
        expected_references=args.expected_references,
    )
    payload = {
        "protocol": PROTOCOL,
        "state": "planned",
        "mode": "execute" if args.execute else "dry_run",
        "timestamp_utc": stamp,
        "reason": args.reason,
        "targets_manifest": str(targets_path),
        "archive_root": str(destination_root),
        "archive_worker_logs": bool(args.archive_worker_logs),
        "planned_target_directories": len(target_sources),
        "planned_non_training_status_files": len(status_sources),
        "planned_entries": len(entries),
        "protected_before": protected,
        "preservation_contract": {
            "pets_v2_runs_moved": False,
            "lira_reference_files_moved": False,
            "training_status_moved": False,
        },
        "entries": entries,
    }
    return payload, destination_root


def same_filesystem(sources: Iterable[Path], destination_root: Path) -> None:
    destination_parent = nearest_existing_parent(destination_root.parent)
    destination_device = destination_parent.stat().st_dev
    for source in sources:
        if source.stat().st_dev != destination_device:
            raise OSError(
                f"atomic rename is impossible across filesystems: {source} -> {destination_root}"
            )


def execute_plan(payload: dict[str, Any], destination_root: Path) -> None:
    sources = [Path(entry["source"]) for entry in payload["entries"]]
    same_filesystem(sources, destination_root)
    destination_root.mkdir(parents=True, exist_ok=False)
    manifest_path = destination_root / "archive_manifest.json"
    payload["state"] = "moving"
    atomic_json(manifest_path, payload)
    moved: list[dict[str, Any]] = []
    try:
        for entry in payload["entries"]:
            source = Path(entry["source"])
            destination = Path(entry["destination"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            entry["status"] = "moving"
            atomic_json(manifest_path, payload)
            moved.append(entry)
            os.replace(source, destination)
            observed = path_fingerprint(destination)
            if observed != entry["fingerprint"]:
                raise RuntimeError(f"post-move fingerprint mismatch: {destination}")
            entry["status"] = "moved"
            atomic_json(manifest_path, payload)

        protected = payload["protected_before"]
        protected_after = protected_inventory(
            Path(protected["run_root"]),
            Path(protected["reference_root"]),
            Path(protected["training_status"]),
            expected_targets=int(protected["target_checkpoint_count"]),
            expected_references=int(protected["reference_score_count"]),
        )
        if protected_after != protected:
            raise RuntimeError("a protected checkpoint/reference/status artifact changed")
        payload["protected_after"] = protected_after
        payload["state"] = "complete"
        payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(manifest_path, payload)
        manifest_hash = file_sha256(manifest_path)
        hash_path = destination_root / "archive_manifest.sha256"
        temporary = hash_path.with_suffix(hash_path.suffix + ".tmp")
        temporary.write_text(f"{manifest_hash}  archive_manifest.json\n")
        os.replace(temporary, hash_path)
    except BaseException as error:
        rollback_failures = []
        for entry in reversed(moved):
            source = Path(entry["source"])
            destination = Path(entry["destination"])
            try:
                if destination.exists() and not source.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(destination, source)
                    entry["status"] = "rolled_back"
            except Exception as rollback_error:  # pragma: no cover - catastrophic I/O
                rollback_failures.append(
                    {"path": str(destination), "error": repr(rollback_error)}
                )
        payload["state"] = "rollback_incomplete" if rollback_failures else "rolled_back"
        payload["failure"] = repr(error)
        payload["rollback_failures"] = rollback_failures
        atomic_json(manifest_path, payload)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("pets_v2_targets/credit_confirmatory_training_targets.csv"),
    )
    parser.add_argument("--result-root", type=Path, default=Path("pets_v2_results/defenses"))
    parser.add_argument("--run-root", type=Path, default=Path("pets_v2_runs"))
    parser.add_argument(
        "--reference-root", type=Path, default=Path("pets_v2_results/lira_references")
    )
    parser.add_argument("--log-root", type=Path, default=Path("pets_v2_logs"))
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("pets_v2_results/archive/evaluation_refresh"),
    )
    parser.add_argument("--reason", required=True)
    parser.add_argument("--archive-worker-logs", action="store_true")
    parser.add_argument("--expected-targets", type=int, default=96)
    parser.add_argument("--expected-references", type=int, default=1536)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        args.dry_run = True
    if not args.reason.strip():
        parser.error("--reason must not be empty")

    locks = acquire_stage_locks(args.log_root.resolve())
    try:
        active = active_v2_processes()
        if active:
            raise RuntimeError(
                "active PETS-v2 stage processes detected:\n" + json.dumps(active, indent=2)
            )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        payload, destination_root = build_plan(args, stamp)
        if args.dry_run:
            print(json.dumps(payload, indent=2, sort_keys=True))
            print(
                f"[DRY-RUN] would archive {payload['planned_target_directories']} target dirs, "
                f"{payload['planned_non_training_status_files']} status files, "
                f"{payload['planned_entries']} total entries"
            )
            return
        execute_plan(payload, destination_root)
        print(f"[OK] recoverable archive: {destination_root}")
        print(f"[OK] manifest: {destination_root / 'archive_manifest.json'}")
    finally:
        for handle in locks:
            handle.close()


if __name__ == "__main__":
    main()
