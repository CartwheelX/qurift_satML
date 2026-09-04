#!/usr/bin/env python3
"""Run prediction defenses and adaptive attacks for one trained PETS target."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "reviewer_tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from qurift.defenses.attacks import (  # noqa: E402
    adaptive_learned_metrics,
    adaptive_threshold_metrics,
    attack_signals,
    fixed_threshold_metrics,
)
from qurift.defenses.discriminator import (  # noqa: E402
    DiscriminatorFitConfig,
    fit_membership_discriminator,
)
from qurift.defenses.dynanoise import DynaNoiseOracle  # noqa: E402
from qurift.defenses.guards import (  # noqa: E402
    LatticeRoundOracle,
    LogitGuardOracle,
    MeasurementGuardOracle,
    StickyInputOracle,
)
from qurift.defenses.hamp import CalibrationSupportGenerator, HAMPOutputOracle  # noqa: E402
from qurift.defenses.memguard import MemGuardOracle  # noqa: E402
from qurift.defenses.oracle import RawOracle  # noqa: E402
from qurift.defenses.utility import classification_utility  # noqa: E402
from qurift.defenses.protocol import (  # noqa: E402
    CONFIRMATORY_CREDIT_QUOTA_PLAN,
    CONFIRMATORY_PARTITION_PROTOCOL,
    PARTITION_PROTOCOL,
    RecordRef,
    label_quotas_for_protocol,
    task_labels_from_dataset,
)
from qurift.defenses.protocol_pooled import (  # noqa: E402
    POOLED_CONFIRMATORY_PARTITION_PROTOCOL,
    POOLED_PARTITION_PROTOCOL,
    build_pooled_defense_partitions,
    pooled_partition_fingerprint,
)
from qurift_target_loader import (  # noqa: E402
    build_config,
    build_dataset,
    import_qurift_main,
    instantiate_model,
    load_saved_model,
    preprocess_like_train,
    read_target_row,
    resolve_target_paths,
)


EVALUATION_PROTOCOL = "pets_defense_evaluation_v5"


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def target_decision_threshold(model_path: Path, *, required: bool = False):
    """Load the validation-frozen binary threshold stored with a checkpoint."""

    metadata_path = model_path.with_name("training_metadata.json")
    if not metadata_path.exists():
        if required:
            raise FileNotFoundError(
                f"missing decision-rule metadata for binary PETS target: {metadata_path}"
            )
        return None
    payload = json.loads(metadata_path.read_text())
    rule = payload.get("decision_rule")
    if not isinstance(rule, Mapping):
        if required:
            raise RuntimeError(
                f"binary PETS target has no validation-frozen decision rule: {metadata_path}"
            )
        return None
    if rule.get("selection_split") != "valid" or int(rule.get("test_records_consulted", -1)) != 0:
        raise RuntimeError(f"invalid target decision-rule provenance: {metadata_path}")
    threshold = float(rule["threshold"])
    if not 0.0 < threshold < 1.0:
        raise RuntimeError(f"invalid target decision threshold: {threshold}")
    return threshold


def materialize(dataset: Mapping[str, Any], refs: Sequence[RecordRef]):
    inputs = []
    labels = []
    for ref in refs:
        item = dataset[ref.split][ref.index]
        value = item["image"]
        inputs.append(value if torch.is_tensor(value) else torch.as_tensor(value))
        label = item["digit"]
        observed_label = int(label.item()) if torch.is_tensor(label) else int(label)
        if observed_label != int(ref.task_label):
            raise RuntimeError(
                f"partition task label changed for {ref.record_id}: "
                f"manifest={ref.task_label}, dataset={observed_label}"
            )
        labels.append(observed_label)
    return (
        torch.stack(inputs),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor([ref.membership for ref in refs], dtype=torch.long),
        [ref.record_id for ref in refs],
    )


def batch_predict(oracle, inputs: torch.Tensor, ids: Sequence[str], batch_size: int):
    batches = []
    for start in range(0, len(inputs), int(batch_size)):
        batches.append(
            oracle.predict(
                inputs[start : start + batch_size],
                query_ids=ids[start : start + batch_size],
            )
        )
    if len(batches) == 1:
        return batches[0]
    from qurift.defenses.base import PredictionBatch

    diagnostic_names = set.intersection(*(set(batch.diagnostics) for batch in batches))
    return PredictionBatch(
        model_output=torch.cat([batch.model_output for batch in batches]),
        logits=torch.cat([batch.logits for batch in batches]),
        log_probabilities=torch.cat([batch.log_probabilities for batch in batches]),
        probabilities=torch.cat([batch.probabilities for batch in batches]),
        labels=torch.cat([batch.labels for batch in batches]),
        measurement=(
            None
            if batches[0].measurement is None
            else torch.cat([batch.measurement for batch in batches])
        ),
        diagnostics={
            name: torch.cat([batch.diagnostics[name] for batch in batches])
            for name in diagnostic_names
        },
        metadata=dict(batches[0].metadata),
    ).validate()


def utility_metrics(output, labels: torch.Tensor) -> Dict[str, float]:
    labels = labels.to(output.probabilities.device)
    return classification_utility(
        output.probabilities, labels, predictions=output.labels
    ) | {
        "nll": float(F.nll_loss(output.log_probabilities, labels).item()),
        "mean_entropy": float(
            (-(output.probabilities * output.log_probabilities).sum(1)).mean().item()
        ),
    }


def build_defenses(args, raw, discriminator, generator, head):
    requested = {value.strip() for value in args.defenses.split(",") if value.strip()}
    known = {
        "none",
        "dynanoise",
        "hamp_output",
        "memguard",
        "logitguard_continuous",
        "logitguard_quantized",
        "measurementguard_continuous",
        "lattice_round",
        "memgq_lattice",
        "memgq_lattice_sticky",
    }
    unknown = requested - known
    if unknown:
        raise ValueError(f"unknown defenses: {sorted(unknown)}")
    values = {}
    if "none" in requested:
        values["none"] = raw
    if "dynanoise" in requested:
        values["dynanoise"] = DynaNoiseOracle(
            raw,
            base_variance=args.dynanoise_base_variance,
            confidence_lambda=args.dynanoise_lambda,
            temperature=args.dynanoise_temperature,
            ensemble_size=args.dynanoise_ensemble,
            seed=args.seed,
        )
    if "hamp_output" in requested:
        values["hamp_output"] = HAMPOutputOracle(raw, generator)
    if "memguard" in requested:
        values["memguard"] = MemGuardOracle(
            raw,
            discriminator,
            max_iterations=args.optimizer_iterations,
            distortion_weights=(0.1, 1.0, 10.0),
        )
    if "logitguard_continuous" in requested:
        values["logitguard_continuous"] = LogitGuardOracle(
            raw, discriminator, iterations=args.optimizer_iterations
        )
    if "logitguard_quantized" in requested:
        values["logitguard_quantized"] = LogitGuardOracle(
            raw,
            discriminator,
            iterations=args.optimizer_iterations,
            quantization_step=args.logit_quantization_step,
        )
    if "measurementguard_continuous" in requested:
        values["measurementguard_continuous"] = MeasurementGuardOracle(
            raw, head, discriminator, iterations=args.optimizer_iterations
        )
    if "lattice_round" in requested:
        values["lattice_round"] = LatticeRoundOracle(raw, head, shots=args.shots)
    if "memgq_lattice" in requested or "memgq_lattice_sticky" in requested:
        memgq = MeasurementGuardOracle(
            raw,
            head,
            discriminator,
            iterations=args.optimizer_iterations,
            shots=args.shots,
        )
        if "memgq_lattice" in requested:
            values["memgq_lattice"] = memgq
        if "memgq_lattice_sticky" in requested:
            values["memgq_lattice_sticky"] = StickyInputOracle(
                memgq, resolution=args.sticky_resolution, secret=args.sticky_secret
            )
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--run-root", type=Path, default=Path("pets_runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("pets_results/defense_pilot"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--defense-per-class", type=int, default=50)
    parser.add_argument("--attack-per-class", type=int, default=50)
    parser.add_argument("--evaluation-per-class", type=int, default=100)
    parser.add_argument(
        "--evaluation-nonmember-multiplier",
        type=int,
        default=1,
        help=(
            "Widen ONLY the final-evaluation non-member side by this integer "
            "factor, drawing the extra records from the held-out test split. "
            "1 (default) reproduces the frozen partition contract bit for bit. "
            "Higher values shrink the per-block AUC standard error without "
            "changing which records are members. Write these runs to a separate "
            "--out-dir; they are not interchangeable with frozen-contract runs."
        ),
    )
    parser.add_argument("--discriminator-epochs", type=int, default=100)
    parser.add_argument("--optimizer-iterations", type=int, default=30)
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--logit-quantization-step", type=float, default=0.01)
    parser.add_argument("--sticky-resolution", type=float, default=0.01)
    parser.add_argument("--sticky-secret", default=os.environ.get("QURIFT_PETS_STICKY_SECRET", ""))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dynanoise-base-variance", type=float, default=0.3)
    parser.add_argument("--dynanoise-lambda", type=float, default=2.0)
    parser.add_argument("--dynanoise-temperature", type=float, default=10.0)
    parser.add_argument("--dynanoise-ensemble", type=int, default=1)
    parser.add_argument(
        "--defenses",
        default=(
            "none,dynanoise,hamp_output,memguard,logitguard_continuous,"
            "logitguard_quantized,measurementguard_continuous,lattice_round,"
            "memgq_lattice,memgq_lattice_sticky"
        ),
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    targets = args.targets if args.targets.is_absolute() else repo_root / args.targets
    run_root = args.run_root if args.run_root.is_absolute() else repo_root / args.run_root
    output_root = args.out_dir if args.out_dir.is_absolute() else repo_root / args.out_dir
    row = read_target_row(targets, args.target_id)
    label_quotas = label_quotas_for_protocol(row.get("confirmatory_protocol", ""))
    quota_plan_name = (
        CONFIRMATORY_CREDIT_QUOTA_PLAN if label_quotas is not None else None
    )
    requested_conditions = {
        value.strip() for value in args.defenses.split(",") if value.strip()
    }
    requested_protocol_arguments = {
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "defense_per_class": int(args.defense_per_class),
        "attack_per_class": int(args.attack_per_class),
        "evaluation_per_class": int(args.evaluation_per_class),
        "evaluation_nonmember_multiplier": int(args.evaluation_nonmember_multiplier),
        "quota_plan_name": quota_plan_name,
        "discriminator_epochs": int(args.discriminator_epochs),
        "optimizer_iterations": int(args.optimizer_iterations),
        "shots": int(args.shots),
        "logit_quantization_step": float(args.logit_quantization_step),
        "sticky_resolution": float(args.sticky_resolution),
        "sticky_secret_sha256": (
            hashlib.sha256(args.sticky_secret.encode("utf-8")).hexdigest()
            if args.sticky_secret
            else None
        ),
        "dynanoise_base_variance": float(args.dynanoise_base_variance),
        "dynanoise_lambda": float(args.dynanoise_lambda),
        "dynanoise_temperature": float(args.dynanoise_temperature),
        "dynanoise_ensemble": int(args.dynanoise_ensemble),
    }

    target_out = output_root / args.target_id
    target_out.mkdir(parents=True, exist_ok=True)
    metadata_path = target_out / "evaluation_metadata.json"
    partition_path = target_out / "partition_manifest.json"
    expected_files = (
        target_out / "adaptive_attack_metrics.csv",
        target_out / "final_predictions.csv",
        target_out / "test_utility_predictions.csv",
        metadata_path,
        partition_path,
    )
    if args.resume and metadata_path.exists():
        try:
            existing_metadata = json.loads(metadata_path.read_text())
            existing_partition = json.loads(partition_path.read_text())
        except (json.JSONDecodeError, OSError, FileNotFoundError):
            existing_metadata, existing_partition = {}, {}
        if (
            all(path.exists() for path in expected_files)
            and existing_metadata.get("protocol") == EVALUATION_PROTOCOL
            and existing_metadata.get("utility_evaluation", {}).get("scope")
            == "full_held_out_test_split"
            and set(existing_metadata.get("conditions", {})) == requested_conditions
            and existing_metadata.get("protocol_arguments")
            == requested_protocol_arguments
            and existing_partition.get("protocol")
            == (
                (
                    PARTITION_PROTOCOL
                    if args.evaluation_nonmember_multiplier == 1
                    else POOLED_PARTITION_PROTOCOL
                )
                if label_quotas is None
                else (
                    CONFIRMATORY_PARTITION_PROTOCOL
                    if args.evaluation_nonmember_multiplier == 1
                    else POOLED_CONFIRMATORY_PARTITION_PROTOCOL
                )
            )
        ):
            print(f"[SKIP] completed defense evaluation: {args.target_id}")
            return
        raise RuntimeError(
            f"{target_out} contains an incompatible earlier evaluation. Run the "
            "PETS correction archiver before using --resume."
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    qmain = import_qurift_main(repo_root)
    dataset, feature_dim = build_dataset(qmain, row, repo_root)
    cfg = build_config(qmain, row, feature_dim, device)
    model, architecture = instantiate_model(qmain, row, cfg, device)
    model_path, _ = resolve_target_paths(row, run_root)
    load_metadata = load_saved_model(model, model_path, device)
    if architecture != "qnn":
        raise NotImplementedError("PETS pilot currently requires the main QNN measurement interface")

    split_labels = task_labels_from_dataset(dataset)
    partitions = build_pooled_defense_partitions(
        train_labels=split_labels["train"],
        valid_labels=split_labels["valid"],
        test_labels=split_labels["test"],
        defense_per_class=args.defense_per_class,
        attack_per_class=args.attack_per_class,
        evaluation_per_class=args.evaluation_per_class,
        seed=args.seed,
        nonmember_multiplier=args.evaluation_nonmember_multiplier,
        label_quotas=label_quotas,
        quota_plan_name=quota_plan_name,
    )
    active_partition_protocol = partitions.protocol
    atomic_json(partition_path, partitions.to_json())
    materialized = {}
    for name in ("defense_calibration", "attack_calibration", "final_evaluation"):
        x, y, membership, ids = materialize(dataset, getattr(partitions, name))
        materialized[name] = (
            preprocess_like_train(x, device),
            y.to(device),
            membership.to(device),
            ids,
        )

    training_defense = str(row.get("training_defense", "none")).strip().lower()
    model_path, _ = resolve_target_paths(row, run_root)
    decision_threshold = target_decision_threshold(model_path, required=True)
    raw = RawOracle(model, decision_threshold=decision_threshold)
    defense_x, defense_y, defense_membership, defense_ids = materialized["defense_calibration"]
    raw_defense = batch_predict(raw, defense_x, defense_ids, args.batch_size)
    discriminator, discriminator_metadata = fit_membership_discriminator(
        raw_defense.probabilities,
        defense_membership,
        hidden_sizes=(64, 32),
        config=DiscriminatorFitConfig(
            epochs=args.discriminator_epochs,
            batch_size=min(args.batch_size, len(defense_x)),
            seed=args.seed,
        ),
    )
    generator = CalibrationSupportGenerator(
        defense_x,
        lower=torch.full_like(defense_x[0], -1.0),
        upper=torch.full_like(defense_x[0], 1.0),
        mode="empirical_marginal",
        seed=args.seed,
    )
    defenses = build_defenses(args, raw, discriminator, generator, model.linear)

    attack_x, attack_y, attack_membership, attack_ids = materialized["attack_calibration"]
    eval_x, eval_y, eval_membership, eval_ids = materialized["final_evaluation"]
    test_refs = tuple(
        RecordRef("test", index, 0, int(label))
        for index, label in enumerate(split_labels["test"])
    )
    test_x, test_y, _, test_ids = materialize(dataset, test_refs)
    test_x = preprocess_like_train(test_x, device)
    test_y = test_y.to(device)
    raw_test = batch_predict(raw, test_x, test_ids, args.batch_size)
    metric_rows = []
    prediction_rows = []
    test_prediction_rows = []
    condition_metadata = {}
    for condition, oracle in defenses.items():
        attack_started = time.monotonic()
        calibration_output = batch_predict(oracle, attack_x, attack_ids, args.batch_size)
        evaluation_output = batch_predict(oracle, eval_x, eval_ids, args.batch_size)
        attack_seconds = time.monotonic() - attack_started
        calibration_signals = attack_signals(calibration_output, attack_y)
        evaluation_signals = attack_signals(evaluation_output, eval_y)
        for attack, scores in evaluation_signals.items():
            metrics = adaptive_threshold_metrics(
                calibration_signals[attack],
                attack_membership,
                scores,
                eval_membership,
                orientation="fixed",
            )
            metric_rows.append(
                {
                    "target_id": args.target_id,
                    "block_id": row.get("block_id"),
                    "structural_cell_id": row.get("structural_cell_id"),
                    "structural_role": row.get("defense_structural_role", row.get("role")),
                    "training_defense": training_defense,
                    "defense": condition,
                    "attack": attack,
                    "attack_fit": "adaptive_defended_calibration",
                    **metrics,
                }
            )
        if condition in {"none", "dynanoise"}:
            artifact_attacks = {
                "artifact_confidence_tau0.9": (
                    evaluation_signals["maximum_probability"],
                    0.9,
                ),
                "artifact_loss_tau0.5": (evaluation_signals["loss"], -0.5),
            }
            for attack, (scores, threshold) in artifact_attacks.items():
                metric_rows.append(
                    {
                        "target_id": args.target_id,
                        "block_id": row.get("block_id"),
                        "structural_cell_id": row.get("structural_cell_id"),
                        "structural_role": row.get(
                            "defense_structural_role", row.get("role")
                        ),
                        "training_defense": training_defense,
                        "defense": condition,
                        "attack": attack,
                        "attack_fit": "dynanoise_artifact_fixed_threshold_appendix",
                        **fixed_threshold_metrics(
                            scores, eval_membership, threshold=threshold
                        ),
                    }
                )
        learned, _ = adaptive_learned_metrics(
            calibration_output,
            attack_y,
            attack_membership,
            evaluation_output,
            eval_y,
            eval_membership,
            seed=args.seed,
        )
        metric_rows.append(
            {
                "target_id": args.target_id,
                "block_id": row.get("block_id"),
                "structural_cell_id": row.get("structural_cell_id"),
                "structural_role": row.get("defense_structural_role", row.get("role")),
                "training_defense": training_defense,
                "defense": condition,
                "attack": "learned_pv_stats_logistic",
                "attack_fit": "adaptive_defended_calibration",
                **learned,
            }
        )
        utility_started = time.monotonic()
        test_output = batch_predict(oracle, test_x, test_ids, args.batch_size)
        utility_seconds = time.monotonic() - utility_started
        utility = utility_metrics(test_output, test_y)
        diagnostics = {
            name: float(value.float().mean().item())
            for name, value in evaluation_output.diagnostics.items()
        }
        condition_metadata[condition] = {
            "config": dict(oracle.config),
            "utility": utility,
            "utility_scope": "full_held_out_test_split",
            "utility_records": len(test_x),
            "runtime_seconds": attack_seconds + utility_seconds,
            "attack_runtime_seconds": attack_seconds,
            "utility_runtime_seconds": utility_seconds,
            "records_per_second": len(test_x) / max(utility_seconds, 1e-12),
            "mean_l1_probability_distortion": float(
                (test_output.probabilities - raw_test.probabilities)
                .abs()
                .sum(1)
                .mean()
                .item()
            ),
            "diagnostic_means": diagnostics,
        }
        probabilities = evaluation_output.probabilities.detach().cpu()
        for index, record_id in enumerate(eval_ids):
            prediction_rows.append(
                {
                    "target_id": args.target_id,
                    "training_defense": training_defense,
                    "defense": condition,
                    "record_id": record_id,
                    "membership": int(eval_membership[index]),
                    "true_label": int(eval_y[index]),
                    "predicted_label": int(evaluation_output.labels[index]),
                    **{
                        f"probability_{column}": float(probabilities[index, column])
                        for column in range(probabilities.shape[1])
                    },
                }
            )
        test_probabilities = test_output.probabilities.detach().cpu()
        for index, record_id in enumerate(test_ids):
            test_prediction_rows.append(
                {
                    "target_id": args.target_id,
                    "training_defense": training_defense,
                    "defense": condition,
                    "record_id": record_id,
                    "test_index": index,
                    "true_label": int(test_y[index]),
                    "predicted_label": int(test_output.labels[index]),
                    **{
                        f"probability_{column}": float(
                            test_probabilities[index, column]
                        )
                        for column in range(test_probabilities.shape[1])
                    },
                }
            )
        print(
            f"[OK] {args.target_id} defense={condition} "
            f"attack_sec={attack_seconds:.2f} utility_sec={utility_seconds:.2f} "
            f"acc={utility['accuracy']:.3f}",
            flush=True,
        )

    pd.DataFrame(metric_rows).to_csv(target_out / "adaptive_attack_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(target_out / "final_predictions.csv", index=False)
    pd.DataFrame(test_prediction_rows).to_csv(
        target_out / "test_utility_predictions.csv", index=False
    )
    atomic_json(
        target_out / "evaluation_metadata.json",
        {
            "protocol": EVALUATION_PROTOCOL,
            "target": row,
            "model_load": load_metadata,
            "partition_fingerprint": pooled_partition_fingerprint(partitions),
            "partition_protocol": active_partition_protocol,
            "quota_plan_name": quota_plan_name,
            "member_task_label_quotas": partitions.to_json().get(
                "member_task_label_quotas"
            ),
            "evaluation_nonmember_multiplier": args.evaluation_nonmember_multiplier,
            "protocol_arguments": requested_protocol_arguments,
            "defense_discriminator": discriminator_metadata,
            "hamp_generator": dict(generator.config),
            "conditions": condition_metadata,
            "adaptive_attack_rule": "refit on disjoint defended attack-calibration records",
            "utility_evaluation": {
                "scope": "full_held_out_test_split",
                "records": len(test_x),
                "member_records_included": 0,
            },
            "target_decision_threshold": decision_threshold,
            "target_label_rule": (
                "binary_probability_threshold"
                if decision_threshold is not None
                else "argmax"
            ),
            "membership_encoding": "1=member,0=nonmember",
        },
    )
    print(f"[DONE] {target_out.resolve()}")


if __name__ == "__main__":
    main()
