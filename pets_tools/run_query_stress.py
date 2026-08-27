#!/usr/bin/env python3
"""Adaptive nearby/repeated-query stress test for output defenses."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "reviewer_tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from pets_tools.run_defense_evaluation import batch_predict, build_defenses, materialize  # noqa: E402
from qurift.defenses.attacks import adaptive_feature_attack_metrics  # noqa: E402
from qurift.defenses.discriminator import DiscriminatorFitConfig, fit_membership_discriminator  # noqa: E402
from qurift.defenses.hamp import CalibrationSupportGenerator  # noqa: E402
from qurift.defenses.oracle import RawOracle  # noqa: E402
from qurift.defenses.protocol import (  # noqa: E402
    PARTITION_PROTOCOL,
    build_defense_partitions,
    partition_fingerprint,
    task_labels_from_dataset,
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


PROTOCOL = "pets_nearby_query_stress_v3"


def existing_result_matches(
    payload,
    *,
    defenses,
    queries: int,
    radius: float,
) -> bool:
    return bool(
        payload.get("protocol") == PROTOCOL
        and set(payload.get("defenses", [])) == set(defenses)
        and int(payload.get("queries", -1)) == int(queries)
        and float(payload.get("linf_radius", -1.0)) == float(radius)
        and payload.get("common_random_perturbations_across_defenses") is True
    )


def nearby_query_features(
    oracle,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    ids,
    *,
    queries: int,
    radius: float,
    batch_size: int,
    seed: int,
):
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    probabilities = []
    for query in range(int(queries)):
        if query == 0:
            perturbed = inputs
        else:
            noise = (
                torch.rand(inputs.shape, generator=generator, dtype=inputs.dtype) * 2.0 - 1.0
            ).to(inputs.device)
            perturbed = (inputs + float(radius) * noise).clamp(-1.0, 1.0)
        query_ids = [f"{record_id}:nearby:{query}" for record_id in ids]
        probabilities.append(
            batch_predict(oracle, perturbed, query_ids, batch_size).probabilities.detach()
        )
    stack = torch.stack(probabilities, dim=0)
    mean = stack.mean(0)
    std = stack.std(0, unbiased=False)
    minimum = stack.amin(0)
    maximum = stack.amax(0)
    entropy = -(stack * stack.clamp_min(torch.finfo(stack.dtype).tiny).log()).sum(2)
    labels = labels.to(stack.device).long()
    true_values = stack.gather(2, labels[None, :, None].expand(len(stack), -1, 1)).squeeze(2)
    base_labels = stack[0].argmax(1)
    flip_rate = (stack.argmax(2) != base_labels[None]).float().mean(0)
    features = torch.cat(
        [
            mean,
            std,
            minimum,
            maximum,
            true_values.mean(0)[:, None],
            true_values.std(0, unbiased=False)[:, None],
            entropy.mean(0)[:, None],
            entropy.std(0, unbiased=False)[:, None],
            flip_rate[:, None],
        ],
        dim=1,
    )
    diagnostics = {
        "mean_probability_std": float(std.mean().item()),
        "mean_label_flip_rate": float(flip_rate.mean().item()),
    }
    return features.cpu().numpy(), diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--run-root", type=Path, default=Path("pets_runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("pets_results/defenses"))
    parser.add_argument("--defenses", default="memgq_lattice,memgq_lattice_sticky")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--radius", type=float, default=0.005)
    parser.add_argument("--defense-per-class", type=int, default=50)
    parser.add_argument("--attack-per-class", type=int, default=50)
    parser.add_argument("--evaluation-per-class", type=int, default=100)
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--archive-existing",
        action="store_true",
        help="Archive an existing metrics file before a protocol-compatible refresh.",
    )
    args = parser.parse_args()
    if args.queries < 2 or args.radius <= 0:
        parser.error("nearby-query stress requires --queries >=2 and --radius >0")

    repo_root = args.repo_root.resolve()
    targets = args.targets if args.targets.is_absolute() else repo_root / args.targets
    run_root = args.run_root if args.run_root.is_absolute() else repo_root / args.run_root
    out_root = args.out_dir if args.out_dir.is_absolute() else repo_root / args.out_dir
    out = out_root / args.target_id / "query_stress"
    out.mkdir(parents=True, exist_ok=True)
    output = out / "metrics.json"
    requested_defenses = [
        value.strip() for value in args.defenses.split(",") if value.strip()
    ]
    if args.archive_existing and output.exists():
        try:
            current = json.loads(output.read_text())
        except (json.JSONDecodeError, OSError):
            current = {}
        if args.resume and existing_result_matches(
            current,
            defenses=requested_defenses,
            queries=args.queries,
            radius=args.radius,
        ):
            print(f"[SKIP] current common-random query result: {output.resolve()}")
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archived = out / f"metrics.before_common_random_queries.{stamp}.json"
        output.replace(archived)
        print(f"[OK] archived previous query-stress result: {archived.resolve()}")
    if args.resume and output.exists():
        print(f"[SKIP] {output.resolve()}")
        return
    device = torch.device(args.device)
    qmain = import_qurift_main(repo_root)
    row = read_target_row(targets, args.target_id)
    dataset, feature_dim = build_dataset(qmain, row, repo_root)
    config = build_config(qmain, row, feature_dim, device)
    model, architecture = instantiate_model(qmain, row, config, device)
    if architecture != "qnn" or str(row.get("dataset", "")) not in {
        "credit_default",
        "breast_cancer_wdbc",
    }:
        raise NotImplementedError("nearby-query pilot currently supports tabular QNN targets")
    model_path, _ = resolve_target_paths(row, run_root)
    load_saved_model(model, model_path, device)
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
    values = {}
    for name in ("defense_calibration", "attack_calibration", "final_evaluation"):
        x, y, membership, ids = materialize(dataset, getattr(partitions, name))
        values[name] = (
            preprocess_like_train(x, device),
            y.to(device),
            membership.to(device),
            ids,
        )
    defense_x, _, defense_membership, defense_ids = values["defense_calibration"]
    raw = RawOracle(model)
    raw_defense = batch_predict(raw, defense_x, defense_ids, args.batch_size)
    discriminator, _ = fit_membership_discriminator(
        raw_defense.probabilities,
        defense_membership,
        hidden_sizes=(64, 32),
        config=DiscriminatorFitConfig(
            epochs=args.discriminator_epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        ),
    )
    generator = CalibrationSupportGenerator(
        defense_x,
        lower=torch.full_like(defense_x[0], -1.0),
        upper=torch.full_like(defense_x[0], 1.0),
        seed=args.seed,
    )
    defenses = build_defenses(args, raw, discriminator, generator, model.linear)
    attack_x, attack_y, attack_membership, attack_ids = values["attack_calibration"]
    eval_x, eval_y, eval_membership, eval_ids = values["final_evaluation"]
    rows = []
    for name, oracle in defenses.items():
        calibration_features, calibration_diagnostics = nearby_query_features(
            oracle,
            attack_x,
            attack_y,
            attack_ids,
            queries=args.queries,
            radius=args.radius,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        evaluation_features, evaluation_diagnostics = nearby_query_features(
            oracle,
            eval_x,
            eval_y,
            eval_ids,
            queries=args.queries,
            radius=args.radius,
            batch_size=args.batch_size,
            seed=args.seed + 1,
        )
        metrics, _ = adaptive_feature_attack_metrics(
            calibration_features,
            attack_membership,
            evaluation_features,
            eval_membership,
            seed=args.seed,
        )
        rows.append(
            {
                "target_id": args.target_id,
                "block_id": row.get("block_id"),
                "structural_cell_id": row.get("structural_cell_id"),
                "structural_role": row.get("defense_structural_role", row.get("role")),
                "training_defense": str(row.get("training_defense", "none")),
                "defense": name,
                "attack": f"nearby_query_logistic_q{args.queries}",
                "attack_fit": "adaptive_defended_calibration",
                **metrics,
                **{f"evaluation_{key}": value for key, value in evaluation_diagnostics.items()},
            }
        )
        print(f"[OK] query stress defense={name} auc={metrics['auc']:.4f}", flush=True)
    output.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "queries": args.queries,
                "linf_radius": args.radius,
                "common_random_perturbations_across_defenses": True,
                "partition_protocol": PARTITION_PROTOCOL,
                "partition_fingerprint": partition_fingerprint(partitions),
                "defenses": list(defenses),
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[DONE] {output.resolve()}")


if __name__ == "__main__":
    main()
