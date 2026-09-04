#!/usr/bin/env python3
"""Score LiRA after applying the same output defense to target and references."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
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
from qurift.defenses.base import defended_batch  # noqa: E402
from qurift.defenses.discriminator import DiscriminatorFitConfig, fit_membership_discriminator  # noqa: E402
from qurift.defenses.hamp import CalibrationSupportGenerator  # noqa: E402
from qurift.defenses.oracle import RawOracle  # noqa: E402
from qurift.defenses.protocol import (  # noqa: E402
    CONFIRMATORY_CREDIT_QUOTA_PLAN,
    CONFIRMATORY_PARTITION_PROTOCOL,
    PARTITION_PROTOCOL,
    build_defense_partitions,
    label_quotas_for_protocol,
    partition_fingerprint,
    task_labels_from_dataset,
)
from qurift_lira_attack import (  # noqa: E402
    LIRA_SCORE_PROTOCOL,
    attack_scores,
    cell_id,
    load_context,
    load_reference_bank,
    reference_checkpoint_path,
    reference_distribution,
    tensor_fingerprint,
    true_class_log_odds,
)
from qurift_target_loader import (  # noqa: E402
    instantiate_model,
    load_saved_model,
    preprocess_like_train,
    read_target_row,
    resolve_target_paths,
)


def averaged_probabilities(oracle, inputs, ids, *, batch_size: int, draws: int):
    values = []
    for _ in range(int(draws)):
        values.append(batch_predict(oracle, inputs, ids, batch_size).probabilities)
    return torch.stack(values).mean(0)


def defense_monte_carlo_draws(defense: str, requested: int) -> int:
    """Avoid recomputing deterministic defenses while averaging stochastic ones."""

    return int(requested) if defense in {"dynanoise", "hamp_output"} else 1


def lira_selection_fingerprint(samples, partitions, attack, evaluation) -> str:
    """Fingerprint the records actually used by defended LiRA.

    The ordinary partition fingerprint cannot represent candidate-pool
    non-members because those records are redrawn from the reference bank.
    """

    values = ["pets_lira_actual_selection_v1"]
    for ref in partitions.defense_calibration:
        values.append(
            f"defense_calibration:{ref.record_id}:{ref.membership}:{ref.task_label}"
        )
    for name, indices in (
        ("attack_calibration", attack),
        ("final_evaluation", evaluation),
    ):
        for index in np.asarray(indices, dtype=int).tolist():
            values.append(
                f"{name}:{samples.sample_ids[index]}:"
                f"{int(samples.membership[index])}:{int(samples.labels[index])}"
            )
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def candidate_partitions(
    samples,
    partitions,
    *,
    attack_per_class: int,
    evaluation_per_class: int,
    seed: int,
):
    lookup = {
        (split, int(index)): position
        for position, (split, index) in enumerate(
            zip(samples.split_names, samples.source_indices)
        )
    }
    attack_member = [
        lookup[(ref.split, ref.index)]
        for ref in partitions.attack_calibration
        if ref.membership == 1
    ][: int(attack_per_class)]
    evaluation_member = [
        lookup[(ref.split, ref.index)]
        for ref in partitions.final_evaluation
        if ref.membership == 1
    ][: int(evaluation_per_class)]
    nonmember_by_label = {
        int(label): np.asarray(
            [
                index
                for index, (membership, observed_label) in enumerate(
                    zip(samples.membership.tolist(), samples.labels.tolist())
                )
                if int(membership) == 0 and int(observed_label) == int(label)
            ],
            dtype=int,
        )
        for label in sorted(set(samples.labels.tolist()))
    }
    if len(attack_member) != int(attack_per_class) or len(evaluation_member) != int(
        evaluation_per_class
    ):
        raise RuntimeError("LiRA candidate pool lacks the requested member partitions")
    rng = np.random.default_rng(int(seed))
    for label in nonmember_by_label:
        nonmember_by_label[label] = rng.permutation(nonmember_by_label[label])

    def matched_nonmembers(member_indices):
        chosen = []
        for member_index in member_indices:
            label = int(samples.labels[member_index])
            available = nonmember_by_label.get(label, np.empty(0, dtype=int))
            if len(available) == 0:
                raise RuntimeError(
                    f"LiRA candidate pool lacks test nonmembers with task label {label}"
                )
            chosen.append(int(available[0]))
            nonmember_by_label[label] = available[1:]
        return chosen

    attack_nonmember = matched_nonmembers(attack_member)
    evaluation_nonmember = matched_nonmembers(evaluation_member)
    attack = np.asarray(attack_member + attack_nonmember, dtype=int)
    evaluation = np.asarray(evaluation_member + evaluation_nonmember, dtype=int)
    if set(attack) & set(evaluation):
        raise RuntimeError("LiRA attack-calibration and final candidates overlap")
    return attack, evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--run-root", type=Path, default=Path("pets_runs"))
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("pets_results/defenses"))
    parser.add_argument("--defense", default="none")
    parser.add_argument("--num-references", type=int, default=16)
    parser.add_argument("--mc-samples", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
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
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    targets = args.targets if args.targets.is_absolute() else repo_root / args.targets
    run_root = args.run_root if args.run_root.is_absolute() else repo_root / args.run_root
    reference_dir = (
        args.reference_dir if args.reference_dir.is_absolute() else repo_root / args.reference_dir
    )
    out_root = args.out_dir if args.out_dir.is_absolute() else repo_root / args.out_dir
    target_row = read_target_row(targets, args.target_id)
    label_quotas = label_quotas_for_protocol(
        target_row.get("confirmatory_protocol", "")
    )
    quota_plan_name = (
        CONFIRMATORY_CREDIT_QUOTA_PLAN if label_quotas is not None else None
    )
    protocol_arguments = {
        "defense": str(args.defense),
        "num_references": int(args.num_references),
        "mc_samples": int(args.mc_samples),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "defense_per_class": int(args.defense_per_class),
        "attack_per_class": int(args.attack_per_class),
        "evaluation_per_class": int(args.evaluation_per_class),
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
        "lira_score_protocol": LIRA_SCORE_PROTOCOL,
    }

    out = out_root / args.target_id / "lira"
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / f"{args.defense}_metrics.json"
    if args.resume and metrics_path.exists():
        try:
            existing = json.loads(metrics_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}
        if (
            existing.get("protocol") == "pets_adaptive_defended_lira_v5"
            and existing.get("partition_protocol")
            == (
                CONFIRMATORY_PARTITION_PROTOCOL
                if label_quotas is not None
                else PARTITION_PROTOCOL
            )
            and existing.get("task_label_matched") is True
            and existing.get("protocol_arguments") == protocol_arguments
        ):
            print(f"[SKIP] {metrics_path.resolve()}")
            return
        raise RuntimeError(f"stale LiRA result must be archived before resume: {metrics_path}")
    device = torch.device(args.device)
    qmain, row, dataset, config, samples = load_context(
        repo_root, targets, args.target_id, device
    )
    training_defense = str(row.get("training_defense", "none")).strip().lower()
    if training_defense not in {"none", "l2", "hamp_train", "dp_qml"}:
        raise NotImplementedError(
            f"defended LiRA has no matched reference training loop for {training_defense!r}"
        )
    structural = cell_id(row)
    target_model, _ = instantiate_model(qmain, row, config, device)
    target_model_path, _ = resolve_target_paths(row, run_root)
    load_saved_model(target_model, target_model_path, device)
    split_labels = task_labels_from_dataset(dataset)
    partitions = build_defense_partitions(
        train_labels=split_labels["train"],
        valid_labels=split_labels["valid"],
        test_labels=split_labels["test"],
        defense_per_class=args.defense_per_class,
        attack_per_class=args.attack_per_class,
        evaluation_per_class=args.evaluation_per_class,
        seed=args.seed,
        label_quotas=label_quotas,
        quota_plan_name=quota_plan_name,
    )
    attack_full, evaluation_full = candidate_partitions(
        samples,
        partitions,
        attack_per_class=args.attack_per_class,
        evaluation_per_class=args.evaluation_per_class,
        seed=args.seed,
    )
    selected = np.concatenate([attack_full, evaluation_full])
    if len(np.unique(selected)) != len(selected):
        raise RuntimeError("selected defended-LiRA candidates overlap")
    selected_tensor = torch.as_tensor(selected, dtype=torch.long)
    selection_fingerprint = lira_selection_fingerprint(
        samples, partitions, attack_full, evaluation_full
    )
    attack_indices = np.arange(len(attack_full), dtype=int)
    evaluation_indices = np.arange(len(attack_full), len(selected), dtype=int)
    defense_x, _, defense_membership, defense_ids = materialize(
        dataset, partitions.defense_calibration
    )
    defense_x = preprocess_like_train(defense_x, device)
    defense_membership = defense_membership.to(device)
    target_threshold = target_decision_threshold(target_model_path, required=True)
    target_raw = RawOracle(target_model, decision_threshold=target_threshold)
    raw_defense = batch_predict(target_raw, defense_x, defense_ids, args.batch_size)
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

    def new_generator(offset: int):
        return CalibrationSupportGenerator(
            defense_x,
            lower=torch.full_like(defense_x[0], -1.0),
            upper=torch.full_like(defense_x[0], 1.0),
            seed=args.seed + offset,
        )

    args.defenses = args.defense
    target_oracle = build_defenses(
        args, target_raw, discriminator, new_generator(0), target_model.linear
    )[args.defense]
    candidate_inputs = preprocess_like_train(samples.inputs[selected_tensor], device)
    candidate_ids = [samples.sample_ids[index] for index in selected]
    draws = defense_monte_carlo_draws(args.defense, args.mc_samples)
    target_probabilities = averaged_probabilities(
        target_oracle,
        candidate_inputs,
        candidate_ids,
        batch_size=args.batch_size,
        draws=draws,
    ).detach().cpu().numpy()
    labels = samples.labels[selected_tensor].numpy()
    observed = true_class_log_odds(target_probabilities, labels)

    reference_scores_raw, inclusion, bank_metadata = load_reference_bank(
        reference_dir, structural, args.num_references
    )
    if tensor_fingerprint(samples.inputs, samples.labels) != bank_metadata["candidate_fingerprint"]:
        raise ValueError("target candidate pool differs from LiRA reference bank")
    reference_scores_raw = reference_scores_raw[:, selected]
    inclusion = inclusion[:, selected]
    defended_reference_scores = []
    for reference_id in range(args.num_references):
        checkpoint = reference_checkpoint_path(reference_dir, structural, reference_id)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"missing {checkpoint}; train references with --save-reference-checkpoints"
            )
        seed_row = dict(row)
        seed_row["model_seed"] = args.seed + reference_id + 1
        reference_model, _ = instantiate_model(qmain, seed_row, config, device)
        saved = torch.load(checkpoint, map_location=device)
        state = saved["state_dict"] if isinstance(saved, dict) and "state_dict" in saved else saved
        reference_model.load_state_dict(state, strict=False)
        reference_model.eval()
        reference_threshold = None
        if isinstance(saved, dict):
            reference_threshold = saved.get("decision_threshold")
        if target_threshold is not None and reference_threshold is None:
            raise RuntimeError(
                f"reference checkpoint {checkpoint} lacks its validation-frozen "
                "decision threshold; rebuild the reference bank"
            )
        raw_reference = RawOracle(
            reference_model,
            decision_threshold=(
                None if reference_threshold is None else float(reference_threshold)
            ),
        )
        reference_args = copy.copy(args)
        reference_args.seed = args.seed + 10_000 + reference_id
        reference_oracle = build_defenses(
            reference_args,
            raw_reference,
            discriminator,
            new_generator(reference_id + 1),
            reference_model.linear,
        )[args.defense]
        probabilities = averaged_probabilities(
            reference_oracle,
            candidate_inputs,
            candidate_ids,
            batch_size=args.batch_size,
            draws=draws,
        ).detach().cpu().numpy()
        defended_reference_scores.append(true_class_log_odds(probabilities, labels))
        print(f"[{reference_id + 1}/{args.num_references}] defended reference scored", flush=True)
    reference_scores = np.stack(defended_reference_scores)
    if reference_scores.shape != reference_scores_raw.shape:
        raise RuntimeError("defended reference score matrix shape changed")
    distribution = reference_distribution(reference_scores, inclusion)
    scores = attack_scores(observed, distribution)
    membership = samples.membership[selected_tensor].numpy().astype(int)
    rows = []
    sample_frame = pd.DataFrame(
        {
            "sample_id": candidate_ids,
            "membership": membership,
            "true_label": labels,
            "partition": ["attack_calibration"] * len(attack_indices)
            + ["final_evaluation"] * len(evaluation_indices),
        }
    )
    for attack, score in scores.items():
        # Every score has a predeclared member-higher orientation (likelihood
        # ratio, one-sided OUT tail, OUT-density surprise, or true-class log odds).
        # Pin that orientation rather than relearning it from 50 calibration
        # members independently in every block.
        metrics = adaptive_threshold_metrics(
            torch.tensor(score[attack_indices]),
            torch.tensor(membership[attack_indices]),
            torch.tensor(score[evaluation_indices]),
            torch.tensor(membership[evaluation_indices]),
            orientation="fixed",
        )
        rows.append(
            {
                "target_id": args.target_id,
                "block_id": row.get("block_id"),
                "structural_cell_id": row.get("structural_cell_id"),
                "structural_role": row.get("defense_structural_role", row.get("role")),
                "training_defense": training_defense,
                "defense": args.defense,
                "attack": attack,
                "attack_fit": "adaptive_defended_reference_models",
                **metrics,
            }
        )
        sample_frame[attack] = score
    sample_frame.to_csv(out / f"{args.defense}_sample_scores.csv", index=False)
    payload = {
        "protocol": "pets_adaptive_defended_lira_v5",
        "target_id": args.target_id,
        "defense": args.defense,
        "training_defense": training_defense,
        "num_references": args.num_references,
        "mc_samples": args.mc_samples,
        "actual_oracle_draws": draws,
        "full_candidate_pool": len(samples.labels),
        "scored_candidates": len(selected),
        "candidate_fingerprint": bank_metadata["candidate_fingerprint"],
        "partition_protocol": partitions.protocol,
        "partition_fingerprint": selection_fingerprint,
        "partition_fingerprint_scheme": "pets_lira_actual_selection_v1",
        "source_partition_fingerprint": partition_fingerprint(partitions),
        "quota_plan_name": quota_plan_name,
        "member_task_label_quotas": partitions.to_json().get(
            "member_task_label_quotas"
        ),
        "task_label_matched": True,
        "candidate_partition": {
            "attack_calibration": {
                "members": int(args.attack_per_class),
                "nonmembers": int(args.attack_per_class),
                "member_source": "target_train",
                "nonmember_source": "target_test_lira_candidate_pool",
            },
            "final_evaluation": {
                "members": int(args.evaluation_per_class),
                "nonmembers": int(args.evaluation_per_class),
                "member_source": "target_train",
                "nonmember_source": "target_test_lira_candidate_pool",
            },
            "disjoint": True,
        },
        "reference_outputs_defended": True,
        "target_output_defended": True,
        "defense_config": dict(target_oracle.config),
        "protocol_arguments": protocol_arguments,
        "lira_score_protocol": LIRA_SCORE_PROTOCOL,
        "membership_encoding": "1=member,0=nonmember",
        "target_decision_threshold": target_threshold,
        "target_label_rule": (
            "binary_probability_threshold" if target_threshold is not None else "argmax"
        ),
        "rows": rows,
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[DONE] {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
