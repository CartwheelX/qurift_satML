#!/usr/bin/env python3
"""Evaluate a saved QuRiFT target under exact and finite-shot Aer modes.

Modes
-----
* exact: saved target evaluated with noiseless TorchQuantum.
* ideal_shot: the same converted/transpiled circuits evaluated with ideal Aer.
* noisy_shot: the same converted/transpiled circuits evaluated with an IBM
  backend-derived Aer noise model.

The noisy mode never silently falls back to ideal Aer.  If noise cannot be
loaded, `--require-noise` aborts; otherwise the noisy condition is explicitly
recorded as skipped/failed while ideal-shot evaluation may continue.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import hashlib
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from qurift_qiskit_bridge import (  # noqa: E402
    BackendNoiseContext,
    circuit_resource_counts,
    counts_to_z_expectations,
    load_backend_noise_context,
    load_backend_noise_snapshot,
    run_aer_counts,
    transpile_for_backend,
    write_backend_snapshot,
)
from qurift_target_loader import (  # noqa: E402
    apply_classical_head,
    build_config,
    build_dataset,
    build_qiskit_circuits,
    exact_probabilities,
    import_qurift_main,
    instantiate_model,
    load_saved_model,
    read_target_row,
    resolve_target_paths,
    select_member_nonmember_samples,
    target_row_to_jsonable,
    verify_attack_payload,
)


CI_SIMULATOR = "95% paired/percentile bootstrap over finite-shot simulator seeds"
CI_NONE = "not applicable: deterministic exact evaluation or one simulator seed"
PAYLOAD_SCHEMA_VERSION = 2
API_AGGREGATION = "mean_api_probabilities"
POOLED_COUNT_AGGREGATION = "pooled_counts_before_head"


def parse_int_list(text: str) -> List[int]:
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    return values


def parse_modes(text: str) -> List[str]:
    allowed = {"exact", "ideal_shot", "noisy_shot"}
    values = []
    for token in str(text).split(","):
        token = token.strip().lower()
        if token:
            if token not in allowed:
                raise ValueError(f"Unknown mode {token!r}; choices are {sorted(allowed)}")
            if token not in values:
                values.append(token)
    return values


def parse_query_shot_pairs(text: str) -> List[Tuple[int, int]]:
    """Parse QUERYxSHOTS pairs, where shots are per query."""
    output: List[Tuple[int, int]] = []
    for token in str(text).split(","):
        token = token.strip().lower()
        if not token:
            continue
        pieces = token.split("x")
        if len(pieces) != 2:
            raise ValueError(f"Invalid query-shot pair {token!r}; expected QUERYxSHOTS")
        queries, shots = map(int, pieces)
        if queries <= 0 or shots <= 0:
            raise ValueError("Queries and shots per query must be positive")
        pair = (queries, shots)
        if pair not in output:
            output.append(pair)
    return output


def merge_counts(replicates: Sequence[Sequence[Mapping[Any, int]]]) -> List[Dict[Any, int]]:
    if not replicates:
        raise ValueError("No count replicates to merge")
    width = len(replicates[0])
    if any(len(run) != width for run in replicates):
        raise ValueError("Repeated-query count runs have inconsistent circuit counts")
    merged: List[Dict[Any, int]] = []
    for circuit_index in range(width):
        total: Dict[Any, int] = {}
        for run in replicates:
            for key, value in run[circuit_index].items():
                total[key] = total.get(key, 0) + int(value)
        merged.append(total)
    return merged


def evaluate_query_count_runs(
    model: torch.nn.Module,
    count_runs: Sequence[Sequence[Mapping[Any, int]]],
    *,
    n_wires: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the post-measurement API separately to every count run.

    Returns ``(query_expectations, query_probabilities, mean_probabilities,
    pooled_count_probabilities)``.  The first two tensors retain a leading
    query axis.  The pooled-count output is a diagnostic for a count-returning
    interface; it is never substituted for the attacker-facing mean API
    response.
    """
    if not count_runs:
        raise ValueError("At least one query count run is required")
    query_expectations: List[torch.Tensor] = []
    query_probabilities: List[torch.Tensor] = []
    with torch.no_grad():
        for run in count_runs:
            expectations = np.stack(
                [counts_to_z_expectations(item, int(n_wires)) for item in run],
                axis=0,
            )
            measured = torch.tensor(expectations, dtype=torch.float32, device=device)
            probabilities = apply_classical_head(model, measured).detach().cpu()
            query_expectations.append(measured.detach().cpu())
            query_probabilities.append(probabilities)

        pooled_counts = merge_counts(count_runs)
        pooled_expectations = np.stack(
            [counts_to_z_expectations(item, int(n_wires)) for item in pooled_counts],
            axis=0,
        )
        pooled_measured = torch.tensor(
            pooled_expectations, dtype=torch.float32, device=device
        )
        pooled_probabilities = apply_classical_head(
            model, pooled_measured
        ).detach().cpu()

    expectation_tensor = torch.stack(query_expectations, dim=0)
    probability_tensor = torch.stack(query_probabilities, dim=0)
    mean_probabilities = probability_tensor.mean(dim=0)
    return (
        expectation_tensor,
        probability_tensor,
        mean_probabilities,
        pooled_probabilities,
    )


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def probability_statistics(probabilities: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    probabilities = probabilities.float().cpu()
    labels = labels.long().cpu()
    eps = 1e-12
    prediction = probabilities.argmax(dim=1)
    correct = (prediction == labels).float()
    true_probability = probabilities[torch.arange(len(labels)), labels].clamp_min(eps)
    loss = -torch.log(true_probability)
    entropy = -(probabilities.clamp_min(eps) * torch.log(probabilities.clamp_min(eps))).sum(dim=1)
    top = probabilities.topk(k=min(2, probabilities.shape[1]), dim=1).values
    confidence = top[:, 0]
    margin = top[:, 0] - top[:, 1] if probabilities.shape[1] >= 2 else top[:, 0]
    return {
        "prediction": prediction,
        "correctness": correct,
        "loss": loss,
        "entropy": entropy,
        "confidence": confidence,
        "margin": margin,
        "max_probability": confidence,
    }


def safe_auc(membership: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(membership)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(membership, score))
    except Exception:
        return float("nan")


def condition_metric_rows(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    membership: torch.Tensor,
    split_codes: torch.Tensor,
    base: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    stats = probability_statistics(probabilities, labels)
    rows: List[Dict[str, Any]] = []
    for split_code, split_name in ((0, "train"), (1, "test")):
        mask = split_codes == split_code
        if not bool(mask.any()):
            continue
        split_probs = probabilities[mask]
        split_labels = labels[mask]
        split_stats = probability_statistics(split_probs, split_labels)
        rows.extend(
            [
                {**base, "metric_scope": split_name, "metric_name": "target_accuracy", "value": float(split_stats["correctness"].mean().item())},
                {**base, "metric_scope": split_name, "metric_name": "target_loss", "value": float(split_stats["loss"].mean().item())},
                {**base, "metric_scope": split_name, "metric_name": "n_records", "value": float(mask.sum().item())},
            ]
        )

    membership_np = membership.cpu().numpy().astype(int)
    attacks = {
        "loss_auc": -stats["loss"].numpy(),
        "entropy_auc": -stats["entropy"].numpy(),
        "confidence_auc": stats["confidence"].numpy(),
        "margin_auc": stats["margin"].numpy(),
        "correctness_auc": stats["correctness"].numpy(),
        "max_probability_auc": stats["max_probability"].numpy(),
    }
    for name, score in attacks.items():
        rows.append(
            {
                **base,
                "metric_scope": "membership",
                "metric_name": name,
                "value": safe_auc(membership_np, score),
            }
        )
    train_acc = next((row["value"] for row in rows if row["metric_scope"] == "train" and row["metric_name"] == "target_accuracy"), np.nan)
    test_acc = next((row["value"] for row in rows if row["metric_scope"] == "test" and row["metric_name"] == "target_accuracy"), np.nan)
    rows.append(
        {
            **base,
            "metric_scope": "target",
            "metric_name": "generalization_gap_accuracy",
            "value": float(train_acc - test_acc),
        }
    )
    return rows


def sample_prediction_rows(
    probabilities: torch.Tensor,
    samples: Any,
    base: Mapping[str, Any],
    circuit_metrics: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    stats = probability_statistics(probabilities, samples.labels)
    output: List[Dict[str, Any]] = []
    for index in range(len(samples.labels)):
        row: Dict[str, Any] = {
            **base,
            "sample_id": samples.sample_ids[index],
            "source_split": samples.split_names[index],
            "source_index": int(samples.source_indices[index]),
            "label": int(samples.labels[index].item()),
            "membership": int(samples.membership[index].item()),
            "membership_convention": "1=member,0=nonmember",
            "prediction": int(stats["prediction"][index].item()),
            "loss": float(stats["loss"][index].item()),
            "entropy": float(stats["entropy"][index].item()),
            "confidence": float(stats["confidence"][index].item()),
            "margin": float(stats["margin"][index].item()),
            "correctness": int(stats["correctness"][index].item()),
        }
        for class_index, value in enumerate(probabilities[index].tolist()):
            row[f"p_{class_index}"] = float(value)
        if circuit_metrics is not None:
            metric = circuit_metrics[index]
            for key in (
                "transpiled_depth",
                "transpiled_total_gates",
                "transpiled_one_qubit_gates",
                "transpiled_two_qubit_gates",
                "transpiled_multi_qubit_gates",
            ):
                row[key] = metric.get(key)
        output.append(row)
    return output


def save_condition_payload(
    path: Path,
    *,
    probabilities: torch.Tensor,
    samples: Any,
    base: Mapping[str, Any],
    metadata: Mapping[str, Any],
    query_expectations: Optional[torch.Tensor] = None,
    query_probabilities: Optional[torch.Tensor] = None,
    pooled_count_probabilities: Optional[torch.Tensor] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = probability_statistics(probabilities, samples.labels)
    features = torch.cat(
        [
            probabilities.cpu().float(),
            stats["loss"].reshape(-1, 1),
            stats["entropy"].reshape(-1, 1),
            stats["confidence"].reshape(-1, 1),
            stats["margin"].reshape(-1, 1),
            stats["correctness"].reshape(-1, 1),
        ],
        dim=1,
    )
    payload = {
        "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
        "X": features.cpu().float(),
        "pv": probabilities.cpu().float(),
        "pv_dim": int(probabilities.shape[1]),
        "y_true": samples.labels.cpu().long(),
        "y_pred": stats["prediction"].cpu().long(),
        "correct": stats["correctness"].cpu().long(),
        "membership": samples.membership.cpu().long(),
        "split": samples.split_codes.cpu().long(),
        "sample_ids": list(samples.sample_ids),
        "stats": {
            "loss": stats["loss"].cpu().float(),
            "entropy": stats["entropy"].cpu().float(),
            "conf": stats["confidence"].cpu().float(),
            "margin": stats["margin"].cpu().float(),
        },
        "meta": {
            **dict(base),
            **dict(metadata),
            "membership_convention": "1=member",
            "api_aggregation": str(base.get("aggregation", API_AGGREGATION)),
            "query_axis_retained": query_probabilities is not None,
        },
    }
    if query_expectations is not None:
        payload["query_expectations"] = query_expectations.cpu().float()
    if query_probabilities is not None:
        payload["query_pv"] = query_probabilities.cpu().float()
    if pooled_count_probabilities is not None:
        payload["pooled_count_pv"] = pooled_count_probabilities.cpu().float()
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def simulator_seed_bootstrap(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(int(seed))
    output = np.empty(int(n_boot), dtype=float)
    for index in range(int(n_boot)):
        output[index] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
    return float(np.quantile(output, 0.025)), float(np.quantile(output, 0.975)), int(n_boot)


def summarize_metrics(raw: pd.DataFrame, bootstrap: int, seed: int) -> pd.DataFrame:
    group_columns = [
        "target_id",
        "structural_cell_id",
        "fm_kind",
        "reps",
        "depth",
        "model_seed",
        "data_seed",
        "mode",
        "shots",
        "queries",
        "total_shots",
        "aggregation",
        "calibration_timestamp",
        "snapshot_manifest_sha256",
        "metric_scope",
        "metric_name",
        "backend_name",
        "noise_model_loaded",
        "quantum_execution_scope",
    ]
    rows = []
    for keys, group in raw.groupby(group_columns, dropna=False):
        record = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        values = pd.to_numeric(group["value"], errors="coerce").dropna().to_numpy(dtype=float)
        record["n_replicates"] = int(len(values))
        record["mean"] = float(np.mean(values)) if len(values) else np.nan
        record["sd"] = float(np.std(values, ddof=1)) if len(values) >= 2 else np.nan
        stable_text = "|".join(str(value) for value in (keys if isinstance(keys, tuple) else (keys,)))
        stable_offset = int(hashlib.sha256(stable_text.encode("utf-8")).hexdigest()[:8], 16)
        low, high, valid = simulator_seed_bootstrap(
            values,
            bootstrap,
            int(seed) + stable_offset,
        )
        record["ci95_low"] = low
        record["ci95_high"] = high
        record["valid_bootstrap_replicates"] = valid
        record["ci_method"] = CI_SIMULATOR if len(values) >= 2 else CI_NONE
        record["bootstrap_unit"] = "simulator_seeds" if len(values) >= 2 else "none"
        record["bootstrap_replicates"] = int(bootstrap if len(values) >= 2 else 0)
        rows.append(record)
    return pd.DataFrame(rows)


def condition_key(mode: str, shots: int, queries: int, simulator_seed: int) -> str:
    return f"{mode}_q{int(queries)}_shots{int(shots)}_total{int(shots) * int(queries)}_sim{int(simulator_seed)}"


def load_existing_payload(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    version = int(payload.get("payload_schema_version", 0))
    if version != PAYLOAD_SCHEMA_VERSION:
        raise RuntimeError(
            f"Cannot resume incompatible noisy payload schema {version} at {path}; "
            "use a new output directory or remove only this incomplete condition"
        )
    probabilities = payload.get("pv", payload.get("probabilities"))
    if probabilities is None:
        raise KeyError(f"Existing payload has no probability vectors: {path}")
    payload["pv"] = torch.as_tensor(probabilities).float().cpu()
    return payload


def structural_base(row: Mapping[str, Any]) -> Dict[str, Any]:
    def integer(name: str, fallback: int = 0) -> int:
        value = row.get(name, fallback)
        try:
            return int(float(value))
        except Exception:
            return int(fallback)

    return {
        "structural_cell_id": str(row.get("structural_cell_id", "")),
        "fm_kind": str(row.get("fm_kind", "")),
        "reps": integer("reps"),
        "depth": integer("depth"),
        "model_seed": integer("model_seed", integer("seed")),
        "data_seed": integer("data_seed"),
    }


def shard_output_path(
    target_out: Path, filename: str, *, shard_index: int, shard_count: int
) -> Path:
    if int(shard_count) == 1:
        return target_out / filename
    path = Path(filename)
    suffix = f".shard{int(shard_index):03d}-of-{int(shard_count):03d}"
    return target_out / f"{path.stem}{suffix}{path.suffix}"


def condition_belongs_to_shard(
    ordinal: int, *, shard_index: int, shard_count: int
) -> bool:
    return int(ordinal) % int(shard_count) == int(shard_index)


def _read_nonempty_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        frame = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None
    return None if frame.empty else frame


def aggregate_condition_shards(
    target_out: Path,
    *,
    shard_count: int,
    bootstrap: int,
    bootstrap_seed: int,
) -> Dict[str, int]:
    """Merge independently written condition shards into canonical outputs."""
    if int(shard_count) < 2:
        raise ValueError("aggregate_condition_shards requires shard_count >= 2")
    target_out = target_out.resolve()
    specifications = {
        "condition_status.csv": [
            "target_id", "mode", "queries", "shots", "simulator_seed", "aggregation"
        ],
        "condition_metrics_raw.csv": [
            "target_id", "mode", "queries", "shots", "simulator_seed",
            "aggregation", "metric_scope", "metric_name",
        ],
        "per_sample_predictions.csv": [
            "target_id", "mode", "queries", "shots", "simulator_seed",
            "aggregation", "sample_id",
        ],
        "failures.csv": [
            "target_id", "mode", "queries", "shots", "simulator_seed", "stage"
        ],
    }
    merged: Dict[str, pd.DataFrame] = {}
    for filename, deduplicate_by in specifications.items():
        frames = []
        for shard_index in range(int(shard_count)):
            path = shard_output_path(
                target_out,
                filename,
                shard_index=shard_index,
                shard_count=shard_count,
            )
            if not path.exists():
                raise FileNotFoundError(f"Missing condition shard output: {path}")
            frame = _read_nonempty_csv(path)
            if frame is not None:
                frames.append(frame)
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        keys = [column for column in deduplicate_by if column in frame]
        if keys:
            frame = frame.drop_duplicates(keys, keep="last")
            frame = frame.sort_values(keys).reset_index(drop=True)
        atomic_csv(frame, target_out / filename)
        merged[filename] = frame

    raw_metrics = merged["condition_metrics_raw.csv"]
    if not raw_metrics.empty:
        summary = summarize_metrics(raw_metrics, int(bootstrap), int(bootstrap_seed))
        atomic_csv(summary, target_out / "condition_metrics_summary.csv")

    shard_metadata = shard_output_path(
        target_out,
        "run_metadata.json",
        shard_index=0,
        shard_count=shard_count,
    )
    if shard_metadata.exists():
        metadata = json.loads(shard_metadata.read_text(encoding="utf-8"))
        metadata["condition_shards"] = int(shard_count)
        metadata["condition_shards_aggregated"] = True
        atomic_json(metadata, target_out / "run_metadata.json")
    return {
        "conditions": len(merged["condition_status.csv"]),
        "metric_rows": len(raw_metrics),
        "prediction_rows": len(merged["per_sample_predictions.csv"]),
        "failures": len(merged["failures.csv"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--run-root", type=Path, default=Path("reviewer_runs"))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--attack-data-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("reviewer_results/noisy_eval"))
    parser.add_argument("--modes", default="exact,ideal_shot,noisy_shot")
    parser.add_argument("--shots", default="128,512,1024")
    parser.add_argument("--queries", default="1", help="Legacy Cartesian query-count grid.")
    parser.add_argument(
        "--query-shot-pairs",
        default="",
        help="Comma-separated QUERYxSHOTS_PER_QUERY conditions, e.g. 1x2560,20x128. Overrides --shots/--queries.",
    )
    parser.add_argument("--simulator-seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--transpiler-seed", type=int, default=0)
    parser.add_argument("--optimization-level", type=int, choices=[0, 1, 2, 3], default=1)
    parser.add_argument("--backend-name", default="ibm_kingston")
    parser.add_argument("--noise-backend-name", default=None)
    parser.add_argument("--ibm-account-name", default=None)
    parser.add_argument(
        "--backend-snapshot",
        type=Path,
        default=None,
        help=(
            "Credential-free frozen snapshot directory produced by "
            "probe_ibm_backend_noise.py. When set, no IBM service is contacted."
        ),
    )
    parser.add_argument("--require-noise", action="store_true")
    parser.add_argument("--allow-backend-mismatch", action="store_true")
    parser.add_argument("--n-member", type=int, default=0, help="0 means all training/member samples")
    parser.add_argument("--n-nonmember", type=int, default=0, help="0 means all test/non-member samples")
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--qiskit-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-attack-payload", action="store_true")
    parser.add_argument("--condition-shard-index", type=int, default=0)
    parser.add_argument("--condition-shard-count", type=int, default=1)
    args = parser.parse_args()

    start_time = time.time()
    modes = parse_modes(args.modes)
    shots_grid = parse_int_list(args.shots)
    query_grid = parse_int_list(args.queries)
    query_shot_pairs = parse_query_shot_pairs(args.query_shot_pairs)
    if not query_shot_pairs:
        query_shot_pairs = [(queries, shots) for queries in query_grid for shots in shots_grid]
    simulator_seeds = parse_int_list(args.simulator_seeds)
    if any(value <= 0 for value in shots_grid):
        raise ValueError("All shot counts must be positive")
    if any(value <= 0 for value in query_grid):
        raise ValueError("All query counts must be positive")
    if not simulator_seeds and any(mode.endswith("shot") for mode in modes):
        raise ValueError("At least one simulator seed is required for finite-shot modes")
    if args.require_noise and "noisy_shot" not in modes:
        raise ValueError("--require-noise was specified but noisy_shot is not in --modes")
    if args.condition_shard_count < 1:
        raise ValueError("--condition-shard-count must be positive")
    if not 0 <= args.condition_shard_index < args.condition_shard_count:
        raise ValueError("--condition-shard-index must be in [0, condition-shard-count)")

    os.environ.setdefault("QURIFT_DISABLE_DEBUG_EXPORTS", "1")
    os.environ.setdefault("QURIFT_DISABLE_CIRCUIT_EXPORTS", "1")

    repo_root = args.repo_root.resolve()
    row = read_target_row(args.targets, args.target_id)
    target_structure = structural_base(row)
    snapshot_manifest_sha256 = None
    if args.backend_snapshot is not None:
        snapshot_manifest_path = args.backend_snapshot.resolve() / "snapshot_manifest.json"
        if snapshot_manifest_path.is_file():
            snapshot_manifest_sha256 = hashlib.sha256(
                snapshot_manifest_path.read_bytes()
            ).hexdigest()
    default_model, default_attack = resolve_target_paths(row, args.run_root)
    model_path = (args.model_path or default_model).resolve()
    attack_path = (args.attack_data_path or default_attack).resolve()
    target_out = args.out_dir / args.target_id
    payload_dir = target_out / "payloads"
    target_out.mkdir(parents=True, exist_ok=True)
    payload_dir.mkdir(parents=True, exist_ok=True)
    status_path = shard_output_path(
        target_out,
        "condition_status.csv",
        shard_index=args.condition_shard_index,
        shard_count=args.condition_shard_count,
    )
    failures_path = shard_output_path(
        target_out,
        "failures.csv",
        shard_index=args.condition_shard_index,
        shard_count=args.condition_shard_count,
    )

    requested_device = str(args.device).lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    qmain = import_qurift_main(repo_root)
    dataset, feature_dim = build_dataset(qmain, row, repo_root)
    cfg = build_config(qmain, row, feature_dim, device)
    model, architecture = instantiate_model(qmain, row, cfg, device)
    load_report = load_saved_model(model, model_path, device)
    samples = select_member_nonmember_samples(
        dataset,
        n_member=args.n_member,
        n_nonmember=args.n_nonmember,
        selection_seed=args.sample_seed,
    )

    manifest = pd.DataFrame(
        {
            "target_id": args.target_id,
            "sample_id": samples.sample_ids,
            "source_split": samples.split_names,
            "source_index": samples.source_indices,
            "label": samples.labels.tolist(),
            "membership": samples.membership.tolist(),
            "membership_convention": "1=member,0=nonmember",
            "sample_selection_seed": int(args.sample_seed),
        }
    )
    atomic_csv(manifest, target_out / "sample_manifest.csv")

    sample_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    transpile_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    condition_status: List[Dict[str, Any]] = []
    backend_context: Optional[BackendNoiseContext] = None
    quantum_scope = "exact_full_model"

    exact_probs: Optional[torch.Tensor] = None
    shard_owns_exact = args.condition_shard_index == 0
    if ("exact" in modes and shard_owns_exact) or args.verify_attack_payload:
        exact_path = payload_dir / "exact.pt"
        if args.resume and exact_path.exists():
            exact_payload = load_existing_payload(exact_path)
            if list(exact_payload.get("sample_ids", [])) != list(samples.sample_ids):
                raise RuntimeError(
                    f"Exact resume sample manifest mismatch at {exact_path}"
                )
            exact_probs = exact_payload["pv"]
            status = "resumed"
        else:
            exact_probs = exact_probabilities(
                model,
                samples,
                device=device,
                batch_size=args.batch_size,
            )
            save_condition_payload(
                exact_path,
                probabilities=exact_probs,
                samples=samples,
                base={
                    "target_id": args.target_id,
                    **target_structure,
                    "mode": "exact",
                    "shots": 0,
                    "queries": 0,
                    "total_shots": 0,
                    "simulator_seed": -1,
                    "aggregation": "exact",
                    "snapshot_manifest_sha256": snapshot_manifest_sha256,
                    "calibration_timestamp": None,
                },
                metadata={"quantum_execution_scope": "exact_full_model"},
            )
            status = "ok"
        if "exact" in modes and shard_owns_exact:
            base = {
                "target_id": args.target_id,
                **target_structure,
                "mode": "exact",
                "shots": 0,
                "queries": 0,
                "total_shots": 0,
                "simulator_seed": -1,
                "aggregation": "exact",
                "snapshot_manifest_sha256": snapshot_manifest_sha256,
                "calibration_timestamp": None,
                "transpiler_seed": int(args.transpiler_seed),
                "backend_name": "none",
                "noise_model_loaded": False,
                "quantum_execution_scope": "exact_full_model",
            }
            sample_rows.extend(sample_prediction_rows(exact_probs, samples, base))
            metric_rows.extend(
                condition_metric_rows(
                    exact_probs,
                    samples.labels,
                    samples.membership,
                    samples.split_codes,
                    base,
                )
            )
            condition_status.append({**base, "status": status, "payload_path": str(exact_path)})

    verification = None
    if args.verify_attack_payload and exact_probs is not None:
        verification = verify_attack_payload(
            attack_path,
            exact_probs=exact_probs,
            samples=samples,
        )
        atomic_json(verification, target_out / "attack_payload_verification.json")

    shot_modes = [mode for mode in modes if mode in {"ideal_shot", "noisy_shot"}]
    circuits = transpiled = None
    circuit_metrics: Optional[List[Dict[str, Any]]] = None
    if shot_modes:
        needs_noise = "noisy_shot" in shot_modes
        try:
            if args.backend_snapshot is not None:
                backend_context = load_backend_noise_snapshot(
                    args.backend_snapshot,
                    require_noise=bool(needs_noise or args.require_noise),
                )
            else:
                backend_context = load_backend_noise_context(
                    args.backend_name,
                    args.noise_backend_name,
                    account_name=args.ibm_account_name,
                    require_noise=bool(args.require_noise),
                    allow_backend_mismatch=bool(args.allow_backend_mismatch),
                )
        except Exception as exc:
            failure = {
                "target_id": args.target_id,
                "stage": "backend_noise_initialization",
                "mode": "noisy_shot" if "noisy_shot" in shot_modes else "ideal_shot",
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            atomic_csv(pd.DataFrame(failures), failures_path)
            atomic_json(
                {
                    "requested_backend_name": args.backend_name,
                    "requested_noise_backend_name": args.noise_backend_name or args.backend_name,
                    "noise_model_loaded": False,
                    "noise_load_error": failure["error"],
                    "require_noise": bool(args.require_noise),
                },
                target_out / "backend_noise_metadata.json",
            )
            raise
        atomic_json(asdict(backend_context.metadata), target_out / "backend_noise_metadata.json")
        if args.backend_snapshot is None:
            write_backend_snapshot(backend_context, target_out / "backend_snapshot")

        try:
            circuits, quantum_scope = build_qiskit_circuits(
                qmain,
                model,
                architecture,
                cfg,
                samples,
                device=device,
                batch_size=args.qiskit_batch_size,
            )
            transpiled = transpile_for_backend(
                circuits,
                backend=backend_context.backend,
                basis_gates=(
                    backend_context.metadata.basis_gates
                    or backend_context.metadata.noise_basis_gates
                ),
                coupling_map=backend_context.metadata.coupling_map,
                optimization_level=args.optimization_level,
                seed_transpiler=args.transpiler_seed,
            )
            circuit_metrics = [circuit_resource_counts(circuit) for circuit in transpiled]
            for index, record in enumerate(circuit_metrics):
                transpile_rows.append(
                    {
                        "target_id": args.target_id,
                        **target_structure,
                        "sample_id": samples.sample_ids[index],
                        "source_split": samples.split_names[index],
                        "source_index": samples.source_indices[index],
                        "transpiler_seed": int(args.transpiler_seed),
                        "optimization_level": int(args.optimization_level),
                        "backend_name": backend_context.metadata.resolved_backend_name,
                        **record,
                    }
                )
            atomic_csv(pd.DataFrame(transpile_rows), target_out / "transpiled_circuit_metrics.csv")
        except Exception as exc:
            failures.append(
                {
                    "target_id": args.target_id,
                    "stage": "circuit_conversion_or_transpilation",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            atomic_csv(pd.DataFrame(failures), failures_path)
            raise

        condition_ordinal = 0
        for mode in shot_modes:
            if mode == "noisy_shot" and backend_context.noise_model is None:
                failure = {
                    "target_id": args.target_id,
                    "stage": "noise_load",
                    "mode": mode,
                    "error": backend_context.metadata.noise_load_error or "noise model unavailable",
                    "status": "skipped_no_noise_model",
                }
                failures.append(failure)
                condition_status.append(failure)
                print(
                    "[WARN] Noisy mode skipped because the IBM noise model was not loaded. "
                    "No ideal result has been substituted under the noisy label."
                )
                continue

            noise_model = backend_context.noise_model if mode == "noisy_shot" else None
            for queries, shots in query_shot_pairs:
                for simulator_seed in simulator_seeds:
                    belongs_to_shard = condition_belongs_to_shard(
                        condition_ordinal,
                        shard_index=args.condition_shard_index,
                        shard_count=args.condition_shard_count,
                    )
                    condition_ordinal += 1
                    if not belongs_to_shard:
                        continue
                    key = condition_key(mode, shots, queries, simulator_seed)
                    payload_path = payload_dir / f"{key}.pt"
                    base = {
                        "target_id": args.target_id,
                        **target_structure,
                        "mode": mode,
                        "shots": int(shots),
                        "queries": int(queries),
                        "total_shots": int(shots * queries),
                        "simulator_seed": int(simulator_seed),
                        "aggregation": API_AGGREGATION,
                        "calibration_timestamp": backend_context.metadata.calibration_timestamp,
                        "snapshot_manifest_sha256": snapshot_manifest_sha256,
                        "transpiler_seed": int(args.transpiler_seed),
                        "backend_name": backend_context.metadata.resolved_backend_name or args.backend_name,
                        "noise_model_loaded": bool(mode == "noisy_shot" and backend_context.noise_model is not None),
                        "quantum_execution_scope": quantum_scope,
                    }
                    t0 = time.time()
                    try:
                        if args.resume and payload_path.exists():
                            saved_payload = load_existing_payload(payload_path)
                            saved_meta = dict(saved_payload.get("meta", {}) or {})
                            expected_resume = {
                                "target_id": args.target_id,
                                "mode": mode,
                                "queries": int(queries),
                                "shots": int(shots),
                                "simulator_seed": int(simulator_seed),
                                "snapshot_manifest_sha256": snapshot_manifest_sha256,
                                "api_aggregation": API_AGGREGATION,
                            }
                            mismatches = {
                                key_: (saved_meta.get(key_), expected_)
                                for key_, expected_ in expected_resume.items()
                                if saved_meta.get(key_) != expected_
                            }
                            if mismatches:
                                raise RuntimeError(
                                    f"Resume metadata mismatch for {payload_path}: {mismatches}"
                                )
                            if list(saved_payload.get("sample_ids", [])) != list(samples.sample_ids):
                                raise RuntimeError(
                                    f"Resume sample manifest mismatch at {payload_path}"
                                )
                            probabilities = saved_payload["pv"]
                            query_expectations = saved_payload.get("query_expectations")
                            query_probabilities = saved_payload.get("query_pv")
                            pooled_probabilities = saved_payload.get("pooled_count_pv")
                            status = "resumed"
                        else:
                            count_runs = [
                                run_aer_counts(
                                    transpiled,
                                    shots=shots,
                                    seed_simulator=(int(simulator_seed) * 1_000_003 + query_index),
                                    noise_model=noise_model,
                                )
                                for query_index in range(int(queries))
                            ]
                            (
                                query_expectations,
                                query_probabilities,
                                probabilities,
                                pooled_probabilities,
                            ) = evaluate_query_count_runs(
                                model,
                                count_runs,
                                n_wires=int(cfg.n_wires),
                                device=device,
                            )
                            save_condition_payload(
                                payload_path,
                                probabilities=probabilities,
                                samples=samples,
                                base=base,
                                metadata={
                                    "backend_metadata": asdict(backend_context.metadata),
                                    "optimization_level": int(args.optimization_level),
                                    "calibration_timestamp": backend_context.metadata.calibration_timestamp,
                                    "backend_snapshot_source": (
                                        str(args.backend_snapshot.resolve())
                                        if args.backend_snapshot is not None
                                        else None
                                    ),
                                },
                                query_expectations=query_expectations,
                                query_probabilities=query_probabilities,
                                pooled_count_probabilities=pooled_probabilities,
                            )
                            status = "ok"

                        sample_rows.extend(
                            sample_prediction_rows(
                                probabilities,
                                samples,
                                base,
                                circuit_metrics=circuit_metrics,
                            )
                        )
                        metric_rows.extend(
                            condition_metric_rows(
                                probabilities,
                                samples.labels,
                                samples.membership,
                                samples.split_codes,
                                base,
                            )
                        )
                        if int(queries) > 1 and pooled_probabilities is not None:
                            pooled_base = {
                                **base,
                                "aggregation": POOLED_COUNT_AGGREGATION,
                            }
                            sample_rows.extend(
                                sample_prediction_rows(
                                    torch.as_tensor(pooled_probabilities).float().cpu(),
                                    samples,
                                    pooled_base,
                                    circuit_metrics=circuit_metrics,
                                )
                            )
                            metric_rows.extend(
                                condition_metric_rows(
                                    torch.as_tensor(pooled_probabilities).float().cpu(),
                                    samples.labels,
                                    samples.membership,
                                    samples.split_codes,
                                    pooled_base,
                                )
                            )
                        condition_status.append(
                            {
                                **base,
                                "status": status,
                                "seconds": round(time.time() - t0, 3),
                                "payload_path": str(payload_path),
                            }
                        )
                    except Exception as exc:
                        failure = {
                            **base,
                            "stage": "aer_execution",
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "seconds": round(time.time() - t0, 3),
                        }
                        failures.append(failure)
                        condition_status.append(failure)

                    atomic_csv(pd.DataFrame(condition_status), status_path)
                    atomic_csv(pd.DataFrame(failures), failures_path)
                    latest = condition_status[-1]
                    print(
                        f"[{args.target_id}] mode={mode} q={queries} shots={shots} "
                        f"sim={simulator_seed} status={latest.get('status')} "
                        f"sec={latest.get('seconds', 0):.1f}",
                        flush=True,
                    )

    if sample_rows:
        atomic_csv(
            pd.DataFrame(sample_rows),
            shard_output_path(
                target_out,
                "per_sample_predictions.csv",
                shard_index=args.condition_shard_index,
                shard_count=args.condition_shard_count,
            ),
        )
    raw_metrics = pd.DataFrame(metric_rows)
    if not raw_metrics.empty:
        atomic_csv(
            raw_metrics,
            shard_output_path(
                target_out,
                "condition_metrics_raw.csv",
                shard_index=args.condition_shard_index,
                shard_count=args.condition_shard_count,
            ),
        )
        if args.condition_shard_count == 1:
            summary = summarize_metrics(raw_metrics, args.bootstrap, args.bootstrap_seed)
            atomic_csv(summary, target_out / "condition_metrics_summary.csv")
    atomic_csv(pd.DataFrame(condition_status), status_path)
    atomic_csv(pd.DataFrame(failures), failures_path)

    run_metadata = {
        "target_row": target_row_to_jsonable(row),
        "model_load": load_report,
        "attack_data_path": str(attack_path),
        "attack_payload_verification": verification,
        "modes": modes,
        "shots": shots_grid,
        "queries": query_grid,
        "query_shot_pairs": [
            {"queries": queries, "shots_per_query": shots, "total_shots": queries * shots}
            for queries, shots in query_shot_pairs
        ],
        "simulator_seeds": simulator_seeds,
        "transpiler_seed": int(args.transpiler_seed),
        "optimization_level": int(args.optimization_level),
        "sample_selection_seed": int(args.sample_seed),
        "n_member": int(samples.membership.sum().item()),
        "n_nonmember": int((samples.membership == 0).sum().item()),
        "membership_convention": "1=member,0=nonmember",
        "backend": asdict(backend_context.metadata) if backend_context is not None else None,
        "backend_snapshot_dir": (
            str(args.backend_snapshot.resolve())
            if args.backend_snapshot is not None
            else (
                str((target_out / "backend_snapshot").resolve())
                if backend_context is not None
                else None
            )
        ),
        "backend_snapshot_frozen": bool(args.backend_snapshot is not None),
        "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
        "attacker_facing_aggregation": API_AGGREGATION,
        "pooled_count_aggregation_retained_as_diagnostic": True,
        "quantum_execution_scope": quantum_scope,
        "architecture_scope_note": (
            "For QCNN, the quanvolutional front-end remains exact TorchQuantum and backend noise is applied "
            "only to the downstream encoder/PQC stack. QNN and compatible HQNN use the full main quantum stack."
        ),
        "ci_method": CI_SIMULATOR,
        "bootstrap_unit": "simulator_seeds",
        "bootstrap_replicates": int(args.bootstrap),
        "elapsed_seconds": round(time.time() - start_time, 3),
        "credentials_logged": False,
        "real_hardware_execution": False,
        "condition_shard_index": int(args.condition_shard_index),
        "condition_shard_count": int(args.condition_shard_count),
    }
    atomic_json(
        run_metadata,
        shard_output_path(
            target_out,
            "run_metadata.json",
            shard_index=args.condition_shard_index,
            shard_count=args.condition_shard_count,
        ),
    )
    atomic_json(
        {
            "primary_uncertainty": "mean ± sample SD across simulator seeds for a fixed target checkpoint",
            "confidence_interval": CI_SIMULATOR,
            "record_bootstrap": "not performed in Checkpoint 4 noisy evaluator",
            "pseudo_replication_warning": (
                "Simulator seeds are repeated measurements of one fixed target checkpoint, not independent target-model seeds."
            ),
            "low_fpr_policy": "No fixed-FPR claim is generated here; Checkpoint 5/6 uses 10% primary and 5% secondary.",
            "query_protocol": (
                "Each API query is passed through the nonlinear classical head separately; "
                "the attacker-facing response is the mean of returned probability vectors. "
                "Raw per-query expectations/probabilities and a pooled-count diagnostic are retained."
            ),
        },
        target_out / "analysis_metadata.json",
    )

    errors = sum(1 for row_ in condition_status if row_.get("status") == "error")
    print(f"[OK] Output directory: {target_out.resolve()}")
    print(f"[OK] Conditions recorded: {len(condition_status)}; errors: {errors}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
