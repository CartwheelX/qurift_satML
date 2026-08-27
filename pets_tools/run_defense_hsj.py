#!/usr/bin/env python3
"""Run the hard-label HSJ MIA adaptively through a defended oracle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "reviewer_tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from pets_tools.run_defense_evaluation import (  # noqa: E402
    batch_predict,
    build_defenses,
    materialize,
    target_decision_threshold,
)
from qurift.defenses.attacks import adaptive_threshold_metrics  # noqa: E402
from qurift.defenses.discriminator import (  # noqa: E402
    DiscriminatorFitConfig,
    fit_membership_discriminator,
)
from qurift.defenses.hamp import CalibrationSupportGenerator  # noqa: E402
from qurift.defenses.oracle import RawOracle  # noqa: E402
from qurift.defenses.protocol import (  # noqa: E402
    PARTITION_PROTOCOL,
    build_defense_partitions,
    partition_fingerprint,
    task_labels_from_dataset,
)
from qurift_label_only_hsj import (  # noqa: E402
    hsj_boundary_distance,
    input_bounds_for_dataset,
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


def stable_seed(*values: Any) -> int:
    digest = hashlib.sha256("|".join(str(value) for value in values).encode()).hexdigest()
    return int(digest[:8], 16)


def hsj_record_seed(
    base_seed: int,
    target_id: str,
    partition_name: str,
    record_id: str,
) -> int:
    """Return the defense-independent seed for one paired HSJ record."""

    return stable_seed(base_seed, target_id, partition_name, record_id)


def balanced_prefix(refs, per_class: int):
    members = [ref for ref in refs if ref.membership == 1][: int(per_class)]
    nonmembers = [ref for ref in refs if ref.membership == 0][: int(per_class)]
    if len(members) != int(per_class) or len(nonmembers) != int(per_class):
        raise ValueError("HSJ prefix lacks the requested member/nonmember records")
    if [ref.task_label for ref in members] != [ref.task_label for ref in nonmembers]:
        raise RuntimeError("HSJ member/nonmember prefix is not task-label matched")
    return tuple(members + nonmembers)


def score_partition(
    oracle,
    dataset,
    refs,
    *,
    bounds,
    device,
    args,
    partition_name: str,
):
    raw_inputs, labels, membership, ids = materialize(dataset, refs)
    rows = []

    def query(points: torch.Tensor) -> torch.Tensor:
        prepared = preprocess_like_train(points, device)
        query_ids = [f"hsj-query-{index}" for index in range(len(prepared))]
        return batch_predict(oracle, prepared, query_ids, args.query_batch_size).labels.cpu()

    for position, (origin, label, member, record_id) in enumerate(
        zip(raw_inputs, labels, membership, ids)
    ):
        original_prediction = int(query(origin.unsqueeze(0))[0].item())
        lower = torch.full_like(origin.float(), float(bounds.lower))
        upper = torch.full_like(origin.float(), float(bounds.upper))
        result = hsj_boundary_distance(
            origin=origin.detach().cpu().float(),
            true_label=int(label),
            original_prediction=original_prediction,
            query_fn=query,
            lower=lower,
            upper=upper,
            max_queries=args.max_queries,
            init_queries=args.init_queries,
            init_batch_size=args.init_batch_size,
            iterations=args.hsj_iterations,
            gradient_samples=args.gradient_samples,
            binary_steps=args.binary_steps,
            step_search_steps=args.step_search_steps,
            gradient_delta_ratio=args.gradient_delta_ratio,
            min_gradient_delta=args.min_gradient_delta,
            # Use common random numbers across defenses.  The attack must send
            # the same stochastic HSJ probes to every defended oracle for a
            # paired comparison; including the defense name here would add an
            # avoidable attack-randomness difference to the defense contrast.
            seed=hsj_record_seed(args.seed, args.target_id, partition_name, record_id),
        )
        rows.append(
            {
                "partition": partition_name,
                "record_id": record_id,
                "membership": int(member),
                "true_label": int(label),
                "original_prediction": original_prediction,
                **result,
            }
        )
        print(
            f"[{position + 1}/{len(refs)}] {partition_name} member={int(member)} "
            f"distance={result['boundary_distance']:.5g} queries={result['boundary_queries']}",
            flush=True,
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--run-root", type=Path, default=Path("pets_runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("pets_results/defenses"))
    parser.add_argument("--defense", default="none")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--defense-per-class", type=int, default=50)
    parser.add_argument("--attack-per-class", type=int, default=50)
    parser.add_argument("--evaluation-per-class", type=int, default=100)
    parser.add_argument("--hsj-records-per-class", type=int, default=20)
    parser.add_argument("--discriminator-epochs", type=int, default=100)
    parser.add_argument("--optimizer-iterations", type=int, default=30)
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--logit-quantization-step", type=float, default=0.01)
    parser.add_argument("--sticky-resolution", type=float, default=0.01)
    parser.add_argument("--sticky-secret", default=os.environ.get("QURIFT_PETS_STICKY_SECRET", ""))
    parser.add_argument("--dynanoise-base-variance", type=float, default=0.3)
    parser.add_argument("--dynanoise-lambda", type=float, default=2.0)
    parser.add_argument("--dynanoise-temperature", type=float, default=10.0)
    parser.add_argument("--dynanoise-ensemble", type=int, default=1)
    parser.add_argument("--max-queries", type=int, default=512)
    parser.add_argument("--init-queries", type=int, default=128)
    parser.add_argument("--init-batch-size", type=int, default=32)
    parser.add_argument("--hsj-iterations", type=int, default=8)
    parser.add_argument("--gradient-samples", type=int, default=32)
    parser.add_argument("--binary-steps", type=int, default=10)
    parser.add_argument("--step-search-steps", type=int, default=10)
    parser.add_argument("--gradient-delta-ratio", type=float, default=0.1)
    parser.add_argument("--min-gradient-delta", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    targets = args.targets if args.targets.is_absolute() else repo_root / args.targets
    run_root = args.run_root if args.run_root.is_absolute() else repo_root / args.run_root
    out_root = args.out_dir if args.out_dir.is_absolute() else repo_root / args.out_dir
    result_dir = out_root / args.target_id / "hsj"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{args.defense}.csv"
    metrics_path = result_dir / f"{args.defense}_metrics.json"
    if args.resume and result_path.exists() and metrics_path.exists():
        try:
            existing = json.loads(metrics_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}
        if (
            existing.get("protocol") == "pets_defended_label_only_hsj_v3"
            and existing.get("partition_protocol") == PARTITION_PROTOCOL
            and existing.get("task_label_matched") is True
        ):
            print(f"[SKIP] {metrics_path.resolve()}")
            return
        raise RuntimeError(f"stale HSJ result must be archived before resume: {metrics_path}")

    device = torch.device(args.device)
    qmain = import_qurift_main(repo_root)
    row = read_target_row(targets, args.target_id)
    dataset, feature_dim = build_dataset(qmain, row, repo_root)
    config = build_config(qmain, row, feature_dim, device)
    model, architecture = instantiate_model(qmain, row, config, device)
    if architecture != "qnn":
        raise NotImplementedError("defended HSJ currently supports QNN")
    model_path, _ = resolve_target_paths(row, run_root)
    load_saved_model(model, model_path, device)
    training_defense = str(row.get("training_defense", "none")).strip().lower()
    decision_threshold = (
        target_decision_threshold(model_path)
        if training_defense in {"l2", "dp_qml"}
        else None
    )
    split_labels = task_labels_from_dataset(dataset)
    partitions = build_defense_partitions(
        train_labels=split_labels["train"],
        valid_labels=split_labels["valid"],
        test_labels=split_labels["test"],
        defense_per_class=args.defense_per_class,
        attack_per_class=args.attack_per_class,
        evaluation_per_class=args.evaluation_per_class,
        seed=args.seed,
    )
    defense_x, _, defense_membership, defense_ids = materialize(
        dataset, partitions.defense_calibration
    )
    defense_x = preprocess_like_train(defense_x, device)
    defense_membership = defense_membership.to(device)
    raw = RawOracle(model, decision_threshold=decision_threshold)
    raw_defense = batch_predict(raw, defense_x, defense_ids, args.batch_size)
    discriminator, _ = fit_membership_discriminator(
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
        seed=args.seed,
    )
    args.defenses = args.defense
    defenses = build_defenses(args, raw, discriminator, generator, model.linear)
    if args.defense not in defenses:
        raise RuntimeError(f"could not construct defense {args.defense!r}")
    oracle = defenses[args.defense]
    bounds = input_bounds_for_dataset(str(row.get("dataset", "")))
    attack_refs = balanced_prefix(partitions.attack_calibration, args.hsj_records_per_class)
    evaluation_refs = balanced_prefix(partitions.final_evaluation, args.hsj_records_per_class)
    calibration = score_partition(
        oracle,
        dataset,
        attack_refs,
        bounds=bounds,
        device=device,
        args=args,
        partition_name="attack_calibration",
    )
    evaluation = score_partition(
        oracle,
        dataset,
        evaluation_refs,
        bounds=bounds,
        device=device,
        args=args,
        partition_name="final_evaluation",
    )
    combined = pd.concat([calibration, evaluation], ignore_index=True)
    combined.insert(0, "target_id", args.target_id)
    combined.insert(1, "defense", args.defense)
    combined.to_csv(result_path, index=False)
    metrics = adaptive_threshold_metrics(
        torch.tensor(calibration.boundary_distance.to_numpy()),
        torch.tensor(calibration.membership.to_numpy()),
        torch.tensor(evaluation.boundary_distance.to_numpy()),
        torch.tensor(evaluation.membership.to_numpy()),
    )
    payload = {
        "protocol": "pets_defended_label_only_hsj_v3",
        "target_id": args.target_id,
        "block_id": row.get("block_id"),
        "structural_cell_id": row.get("structural_cell_id"),
        "structural_role": row.get("defense_structural_role", row.get("role")),
        "training_defense": str(row.get("training_defense", "none")),
        "defense": args.defense,
        "attack": "label_only_hsj",
        "attack_fit": "adaptive_defended_calibration",
        "membership_encoding": "1=member,0=nonmember",
        "partition_protocol": PARTITION_PROTOCOL,
        "partition_fingerprint": partition_fingerprint(partitions),
        "task_label_matched": True,
        "records_per_class_per_partition": args.hsj_records_per_class,
        "metrics": metrics,
        "mean_queries": float(evaluation.boundary_queries.mean()),
        "censored_fraction": float(evaluation.search_censored.mean()),
        "defense_config": dict(oracle.config),
        "target_decision_threshold": decision_threshold,
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[DONE] {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
