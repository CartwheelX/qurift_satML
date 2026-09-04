#!/usr/bin/env python3
"""Reference-model LiRA baselines for completed QuRiFT targets.

The implementation follows the core protocol used by LiRA implementations:

* train same-architecture reference models with explicit per-record IN/OUT
  inclusion;
* score the true class with the numerically stable log-odds
  log(p_y) - log(1 - p_y);
* fit per-record Gaussian IN and OUT distributions; and
* evaluate online/offline likelihood scores, including fixed-variance variants.

This file is an independent QuRiFT integration.  It does not import or vendor
third-party repository code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

# Support both direct script execution and package imports used by unittest.
_REVIEWER_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _REVIEWER_TOOLS_DIR not in sys.path:
    sys.path.insert(0, _REVIEWER_TOOLS_DIR)

# cuBLAS reads this before its first CUDA context is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from scipy.special import log_ndtr
from sklearn.metrics import roc_auc_score

from qurift_target_loader import (
    build_config,
    build_dataset,
    exact_probabilities,
    import_qurift_main,
    instantiate_model,
    load_saved_model,
    preprocess_like_train,
    read_target_row,
    resolve_target_paths,
    select_member_nonmember_samples,
    set_all_seeds,
)
from reviewer_common import (
    CI_RECORD,
    atomic_write_csv,
    atomic_write_json,
    cross_fitted_threshold_metrics,
    stable_seed,
    stratified_bootstrap_auc,
    tpr_at_resolvable_fpr,
    write_analysis_metadata,
)


REFERENCE_COMMITS = {
    "tensorflow/privacy/research/mi_lira_2021": (
        "c5a3633e84f1a296c7e362e7c1926aead4063359"
    ),
    "orientino/lira-pytorch": "50dc2a3fc5e66628d48bf07e05c8c33f9703c789",
    "antibloch/mia_attacks": "9c09fe9d9be982e203df303fb450375f2333987b",
    "Pierre-Joly/Membership-Inference-Attacks": "9182ed809d9fa3d5141d50816b7e83a06590371b",
}
LIRA_SCORE_PROTOCOL = "carlini_eq4_one_sided_with_released_density_aux_v1"
LIRA_ATTACK_NAMES = frozenset(
    {
        "lira_online",
        "lira_online_fixed_variance",
        "lira_offline",
        "lira_offline_fixed_variance",
        "lira_offline_one_sided_z",
        "lira_offline_one_sided_z_fixed_variance",
        "lira_offline_density_surprise",
        "lira_offline_density_surprise_fixed_variance",
        "global_true_class_log_odds",
    }
)


def safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text or "unnamed"


def row_number(row: Mapping[str, Any], key: str, default: float) -> float:
    value = row.get(key, default)
    try:
        return float(default) if pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return float(default)


def row_integer(row: Mapping[str, Any], key: str, default: int) -> int:
    return int(round(row_number(row, key, default)))


def training_mode(row: Mapping[str, Any]) -> str:
    mode = str(row.get("training_defense", "none")).strip().lower()
    return "none" if mode in {"", "nan", "none"} else mode


def training_signature(row: Mapping[str, Any]) -> str:
    """Identify the complete reference-training mechanism used by one bank."""

    mode = training_mode(row)
    if mode == "none":
        return "none"
    if mode == "l2":
        return f"l2_wd{row_number(row, 'weight_decay', 0.0):g}"
    if mode == "hamp_train":
        return (
            f"hamp_g{row_number(row, 'hamp_gamma', 0.95):g}"
            f"_a{row_number(row, 'hamp_alpha', 0.001):g}"
            f"_wd{row_number(row, 'weight_decay', 0.0):g}"
            f"_b{row_integer(row, 'batch_size', 16)}"
            f"_e{row_integer(row, 'epochs', 100)}"
            f"_lr{row_number(row, 'learning_rate', 0.05):g}"
        )
    if mode == "dp_qml":
        return (
            f"dp_eps{row_number(row, 'dp_target_epsilon', 4.0):g}"
            f"_del{row_number(row, 'dp_delta', 1e-5):g}"
            f"_c{row_number(row, 'dp_max_grad_norm', 1.0):g}"
            f"_b{row_integer(row, 'dp_batch_size', 32)}"
            f"_e{row_integer(row, 'dp_epochs', 30)}"
            f"_lr{row_number(row, 'dp_learning_rate', 0.05):g}"
            f"_wd{row_number(row, 'weight_decay', 0.0):g}"
        )
    raise ValueError(f"unsupported reference training defense {mode!r}")


def cell_id(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("structural_cell_id", "")).strip()
    block = str(row.get("block_id", "")).strip()
    block_suffix = "" if block.lower() in {"", "nan", "none"} else f"_block{block}"
    mode = training_mode(row)
    # Preserve the established none/L2 bank names so completed rebuttal banks
    # remain readable. HAMP and DP previously collided with the ordinary wd=0
    # bank, so only those newer mechanisms need an explicit signature suffix.
    if mode in {"none", "l2"}:
        weight_decay = row_number(row, "weight_decay", 0.0)
        training_suffix = f"_wd{weight_decay:g}"
    else:
        training_suffix = f"_train_{training_signature(row)}"
    if explicit and explicit.lower() not in {"nan", "none"}:
        base = explicit.split("|", 1)[0]
        return safe_name(f"{base}{training_suffix}{block_suffix}")
    base = (
        f"{row.get('architecture', 'qnn')}_{row.get('fm_kind', 'unknown')}"
        f"_r{int(float(row.get('reps', 0)))}_d{int(float(row.get('depth', 0)))}"
    )
    return safe_name(f"{base}{training_suffix}{block_suffix}")


def reference_pairing_id(row: Mapping[str, Any]) -> str:
    """Common-random identity shared by all defense arms of one target block."""

    explicit = str(row.get("structural_cell_id", "")).strip()
    if explicit and explicit.lower() not in {"nan", "none"}:
        structural = explicit.split("|", 1)[0]
    else:
        structural = (
            f"{row.get('architecture', 'qnn')}_{row.get('fm_kind', 'unknown')}"
            f"_r{int(float(row.get('reps', 0)))}_d{int(float(row.get('depth', 0)))}"
        )
    block = str(row.get("block_id", "")).strip()
    return safe_name(
        f"{row.get('dataset', 'dataset')}_{structural}_block{block}"
    )


def reference_training_spec(
    row: Mapping[str, Any], *, epochs_override: int | None = None
) -> dict[str, Any]:
    mode = training_mode(row)
    if mode == "dp_qml":
        spec = {
            "mode": mode,
            "epochs": row_integer(row, "dp_epochs", 30),
            "batch_size": row_integer(row, "dp_batch_size", 32),
            "learning_rate": row_number(row, "dp_learning_rate", 0.05),
            "optimizer": "rmsprop",
            "scheduler": "none",
        }
    else:
        spec = {
            "mode": mode,
            "epochs": row_integer(row, "epochs", 100),
            "batch_size": row_integer(row, "batch_size", 16),
            "learning_rate": row_number(row, "learning_rate", 0.05),
            "optimizer": "adam",
            "scheduler": "cosine_annealing",
        }
    if epochs_override is not None:
        spec["epochs"] = int(epochs_override)
    if spec["epochs"] <= 0 or spec["batch_size"] <= 0 or spec["learning_rate"] <= 0:
        raise ValueError(f"invalid reference training specification: {spec}")
    return spec


def tensor_fingerprint(inputs: torch.Tensor, labels: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(inputs.detach().cpu().contiguous().numpy().tobytes())
    digest.update(labels.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def balanced_inclusion_matrix(
    num_references: int, num_candidates: int, seed: int
) -> np.ndarray:
    """Return a deterministic half-IN design balanced by model and record.

    When ``num_candidates`` is divisible by ``num_references`` (as in the
    QuRiFT 400-record pool), every row contains exactly half the records and
    every column appears in exactly half the reference models.
    """
    if num_references < 4 or num_references % 2:
        raise ValueError("--num-references must be an even integer of at least 4")
    half = num_references // 2
    shifts = np.resize(np.arange(num_references, dtype=int), num_candidates)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(shifts)
    design = np.zeros((num_references, num_candidates), dtype=bool)
    for reference_id in range(num_references):
        design[reference_id] = (
            (reference_id - shifts) % num_references
        ) < half
    if not np.all(design.sum(axis=0) == half):
        raise RuntimeError("Internal error: reference inclusion is not column-balanced")
    return design


def true_class_log_odds(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    p_true = probabilities[np.arange(len(labels)), labels]
    p_true = np.clip(p_true, 1e-12, 1.0 - 1e-12)
    return np.log(p_true) - np.log1p(-p_true)


def normal_logpdf(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.maximum(np.asarray(std, dtype=float), 1e-6)
    value = np.asarray(value, dtype=float)
    mean = np.asarray(mean, dtype=float)
    return -0.5 * ((value - mean) / std) ** 2 - np.log(std) - 0.5 * math.log(
        2.0 * math.pi
    )


def reference_distribution(
    scores: np.ndarray, inclusion: np.ndarray
) -> dict[str, np.ndarray | float]:
    n_models, n_candidates = scores.shape
    if inclusion.shape != scores.shape:
        raise ValueError(
            f"Reference shape mismatch: scores={scores.shape}, inclusion={inclusion.shape}"
        )
    in_values: list[np.ndarray] = []
    out_values: list[np.ndarray] = []
    for index in range(n_candidates):
        in_score = scores[inclusion[:, index], index]
        out_score = scores[~inclusion[:, index], index]
        if len(in_score) < 2 or len(out_score) < 2:
            raise ValueError(
                f"Candidate {index} has only {len(in_score)} IN and {len(out_score)} OUT "
                f"reference observations; use at least four reference models"
            )
        in_values.append(in_score)
        out_values.append(out_score)
    in_array = np.stack(in_values)
    out_array = np.stack(out_values)
    mean_in = np.median(in_array, axis=1)
    mean_out = np.median(out_array, axis=1)
    std_in = np.std(in_array, axis=1, ddof=1)
    std_out = np.std(out_array, axis=1, ddof=1)
    fixed_in = float(np.sqrt(np.mean((in_array - mean_in[:, None]) ** 2)))
    fixed_out = float(np.sqrt(np.mean((out_array - mean_out[:, None]) ** 2)))
    return {
        "mean_in": mean_in,
        "mean_out": mean_out,
        "std_in": np.maximum(std_in, 1e-6),
        "std_out": np.maximum(std_out, 1e-6),
        "fixed_std_in": max(fixed_in, 1e-6),
        "fixed_std_out": max(fixed_out, 1e-6),
        "n_models": np.asarray(n_models),
    }


def attack_scores(
    observed: np.ndarray, distribution: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    mean_in = np.asarray(distribution["mean_in"])
    mean_out = np.asarray(distribution["mean_out"])
    std_in = np.asarray(distribution["std_in"])
    std_out = np.asarray(distribution["std_out"])
    fixed_in = np.full_like(mean_in, float(distribution["fixed_std_in"]))
    fixed_out = np.full_like(mean_out, float(distribution["fixed_std_out"]))
    online = normal_logpdf(observed, mean_in, std_in) - normal_logpdf(
        observed, mean_out, std_out
    )
    online_fixed = normal_logpdf(observed, mean_in, fixed_in) - normal_logpdf(
        observed, mean_out, fixed_out
    )
    # Carlini et al., Eq. (4), changes offline LiRA to a one-sided hypothesis
    # test using only the OUT distribution.  log Phi(z) is the numerically
    # stable log-CDF form of that test.  It is strictly monotone in z, so the
    # explicit z-score aliases below have identical ROC/AUC rankings.
    offline_one_sided = (observed - mean_out) / std_out
    offline_one_sided_fixed = (observed - mean_out) / fixed_out
    offline = log_ndtr(offline_one_sided)
    offline_fixed = log_ndtr(offline_one_sided_fixed)

    # The authors' released TensorFlow Privacy artifact instead evaluates OUT
    # density surprise (-log PDF), which is two-sided.  Preserve that useful
    # implementation comparator, but do not label it as the paper's one-sided
    # offline LiRA definition.
    offline_density = -normal_logpdf(observed, mean_out, std_out)
    offline_density_fixed = -normal_logpdf(observed, mean_out, fixed_out)
    return {
        "lira_online": online,
        "lira_online_fixed_variance": online_fixed,
        "lira_offline": offline,
        "lira_offline_fixed_variance": offline_fixed,
        "lira_offline_one_sided_z": offline_one_sided,
        "lira_offline_one_sided_z_fixed_variance": offline_one_sided_fixed,
        "lira_offline_density_surprise": offline_density,
        "lira_offline_density_surprise_fixed_variance": offline_density_fixed,
        "global_true_class_log_odds": observed,
    }


def reference_path(out_dir: Path, structural_cell: str, reference_id: int) -> Path:
    return (
        out_dir
        / "reference_models"
        / safe_name(structural_cell)
        / f"reference_{int(reference_id):03d}.npz"
    )


def reference_checkpoint_path(
    out_dir: Path, structural_cell: str, reference_id: int
) -> Path:
    return reference_path(out_dir, structural_cell, reference_id).with_suffix(".pt")


class CandidateDataset(torch.utils.data.Dataset):
    """Dataset view whose indices exactly match the recorded LiRA candidates."""

    def __init__(self, inputs: torch.Tensor, labels: torch.Tensor):
        if len(inputs) != len(labels):
            raise ValueError("LiRA candidate inputs and labels differ in length")
        self.inputs = inputs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {"image": self.inputs[index], "digit": self.labels[index]}


def load_context(
    repo_root: Path, targets: Path, target_id: str, device: torch.device
):
    qmain = import_qurift_main(repo_root)
    row = read_target_row(targets, target_id)
    architecture = str(row.get("architecture", "qnn")).strip().lower()
    if architecture != "qnn":
        raise NotImplementedError(
            "The rebuttal LiRA reference bank currently supports the primary QNN "
            f"factorial only; got architecture={architecture!r}"
        )
    dataset, feature_dim = build_dataset(qmain, row, repo_root)
    config = build_config(qmain, row, feature_dim, device)
    target_train_size = int(float(row.get("vector_train", len(dataset["train"]))))
    samples = select_member_nonmember_samples(
        dataset,
        n_member=target_train_size,
        n_nonmember=target_train_size,
        selection_seed=int(float(row.get("data_seed", 43))),
    )
    return qmain, row, dataset, config, samples


def _initialize_reference_dp(
    row: Mapping[str, Any],
    model,
    optimizer,
    train_dataset,
    *,
    batch_size: int,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    """Build the DP-SGD session a dp_qml reference model trains under.

    The reference must be calibrated to the same target epsilon as the model it
    calibrates, over its own training-set size, or the OUT distribution it
    contributes is drawn from a different mechanism than the target's.
    """

    import math as _math

    from qurift.defenses.dp_training import (
        DPConfig,
        DPTrainingSession,
        PoissonIndexSampler,
        calibrate_noise_multiplier,
    )

    population = len(train_dataset)
    if population <= 0:
        raise ValueError("dp_qml reference training needs a non-empty training subset")
    sample_rate = float(batch_size) / float(population)
    target_epsilon = row_number(row, "dp_target_epsilon", 4.0)
    delta = row_number(row, "dp_delta", 1e-5)
    steps_per_epoch = int(_math.ceil(population / float(batch_size)))
    try:
        from opacus.accountants.utils import get_noise_multiplier
    except ImportError as error:
        raise RuntimeError("DP reference calibration requires opacus==1.5.4") from error
    initial = float(
        get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant="rdp",
        )
    )
    noise_multiplier, _ = calibrate_noise_multiplier(
        target_epsilon=target_epsilon,
        delta=delta,
        sample_rate=sample_rate,
        steps=steps_per_epoch * int(epochs),
        initial_noise_multiplier=initial,
    )
    config = DPConfig(
        max_grad_norm=row_number(row, "dp_max_grad_norm", 1.0),
        noise_multiplier=noise_multiplier,
        sample_rate=sample_rate,
        delta=delta,
        expected_batch_size=int(batch_size),
    )
    # Sampling and noise draw independent OS entropy, matching how the target's
    # DP session is constructed; the privacy analysis needs them independent, so
    # the reference seed is deliberately not threaded through here.
    return {
        "session": DPTrainingSession(model, optimizer, config),
        "sampler": PoissonIndexSampler(population, sample_rate),
        "dataset": train_dataset,
        "steps_per_epoch": steps_per_epoch,
    }


def _reference_epoch(
    mode: str,
    model,
    dataflow,
    device,
    optimizer,
    qmain,
    *,
    row: Mapping[str, Any],
    dp_state: Mapping[str, Any] | None,
) -> tuple[float, float]:
    """Run one reference-training epoch under the target's training defense.

    LiRA calibration assumes the reference models are drawn from the same
    training procedure as the target.  A reference trained with plain NLL cannot
    calibrate a HAMP or DP-SGD target: its loss distribution is shaped by a
    different objective, so the likelihood ratio is measured against the wrong
    null.  Each arm therefore trains through its own loop here.
    """

    import torch.nn.functional as F

    from qurift.defenses.dp_training import DPTrainingSession, PoissonIndexSampler
    from qurift.defenses.hamp import hamp_training_loss, hamp_true_probability_from_gamma

    if mode in {"none", "l2"}:
        return qmain.train_one_epoch(dataflow, model, device, optimizer)

    if mode == "hamp_train":
        num_classes = int(getattr(model.cfg, "num_classes", 2))
        true_probability = hamp_true_probability_from_gamma(
            row_number(row, "hamp_gamma", 0.95), num_classes
        )
        entropy_weight = row_number(row, "hamp_alpha", 0.001)
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for feed in dataflow["train"]:
            inputs = preprocess_like_train(feed["image"], device)
            labels = feed["digit"].to(device).long()
            output = model(inputs)
            loss, _ = hamp_training_loss(
                output,
                labels,
                true_class_probability=true_probability,
                entropy_weight=entropy_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * len(labels)
            correct += int((output.argmax(1) == labels).sum().item())
            total += len(labels)
        return total_loss / max(total, 1), correct / max(total, 1)

    if mode == "dp_qml":
        if dp_state is None:
            raise ValueError("dp_qml reference training requires an initialized DP session")
        session: DPTrainingSession = dp_state["session"]
        sampler: PoissonIndexSampler = dp_state["sampler"]
        dataset = dp_state["dataset"]
        model.train()
        for _ in range(int(dp_state["steps_per_epoch"])):
            indices = sampler.sample()
            if len(indices) == 0:
                session.step(torch.empty(0, device=device))
                continue
            batch = [dataset[int(index)] for index in indices.tolist()]
            inputs = preprocess_like_train(
                torch.stack([item["image"] for item in batch]), device
            )
            labels = torch.stack(
                [torch.as_tensor(item["digit"]) for item in batch]
            ).to(device).long()
            losses = F.nll_loss(model(inputs), labels, reduction="none")
            session.step(losses)
        return float("nan"), float("nan")

    raise ValueError(f"unsupported reference training defense {mode!r}")


@torch.no_grad()
def reference_decision_rule(
    model,
    valid_dataset,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any] | None:
    """Select the reference model's label rule from validation only."""

    from qurift.defenses.utility import select_binary_decision_threshold

    loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
    )
    probabilities = []
    labels = []
    model.eval()
    for feed in loader:
        inputs = preprocess_like_train(feed["image"], device)
        probabilities.append(model(inputs).exp().detach().cpu())
        labels.append(feed["digit"].detach().cpu().long())
    probability = torch.cat(probabilities)
    label = torch.cat(labels)
    if probability.shape[1] != 2:
        return None
    rule = select_binary_decision_threshold(
        probability.numpy(), label.numpy()
    )
    rule.update(
        {
            "selection_split": "valid",
            "score": "class_1_probability",
            "test_records_consulted": 0,
            "attack_metrics_consulted": False,
        }
    )
    return rule


def train_reference(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    device = torch.device(args.device)
    qmain, row, dataset, config, samples = load_context(
        repo_root, args.targets, args.target_id, device
    )
    structural_cell = cell_id(row)
    output = reference_path(args.out_dir, structural_cell, args.reference_id)
    checkpoint_output = reference_checkpoint_path(
        args.out_dir, structural_cell, args.reference_id
    )
    spec = reference_training_spec(row, epochs_override=args.epochs)
    signature = training_signature(row)
    strict_v2 = str(row.get("confirmatory_protocol", "")).strip() == (
        "pets_credit_three_regime_v2"
    )
    pairing_id = reference_pairing_id(row) if strict_v2 else structural_cell
    candidate_fingerprint = tensor_fingerprint(samples.inputs, samples.labels)
    if args.resume and output.exists():
        try:
            with np.load(output, allow_pickle=False) as saved:
                schedule_complete = True
                signature_complete = (
                    str(saved["training_signature"]) == signature
                    if "training_signature" in saved.files
                    else not strict_v2
                )
                candidate_complete = (
                    str(saved["candidate_fingerprint"]) == candidate_fingerprint
                )
                if strict_v2:
                    schedule_complete = (
                        int(saved["batch_size"]) == int(spec["batch_size"])
                        and math.isclose(
                            float(saved["learning_rate"]),
                            float(spec["learning_rate"]),
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                        and str(saved["optimizer"]) == str(spec["optimizer"])
                        and str(saved["scheduler"]) == str(spec["scheduler"])
                        and str(saved["reference_pairing_id"]) == pairing_id
                    )
                score_complete = (
                    int(saved["num_references"]) == args.num_references
                    and int(saved["reference_id"]) == args.reference_id
                    and int(saved["epochs"]) == int(spec["epochs"])
                    and signature_complete
                    and schedule_complete
                    and candidate_complete
                )
                checkpoint_complete = (
                    not args.save_checkpoint
                    or (checkpoint_output.exists() and checkpoint_output.stat().st_size > 0)
                )
                if score_complete and checkpoint_complete:
                    print(f"[SKIP] complete reference exists: {output.resolve()}")
                    return
        except Exception:
            pass

    reference_seed = stable_seed(
        args.seed, pairing_id, args.reference_id, "qurift_lira_reference"
    )
    design = balanced_inclusion_matrix(
        args.num_references, len(samples.labels), stable_seed(args.seed, pairing_id)
    )
    inclusion = design[args.reference_id]
    expected_train = int(float(row.get("vector_train", inclusion.sum())))
    if int(inclusion.sum()) != expected_train:
        raise ValueError(
            f"Reference model would train on {int(inclusion.sum())} records, but the "
            f"target trains on {expected_train}. Choose a reference count that divides "
            f"the {len(inclusion)}-record candidate pool."
        )

    # Inclusion rows are defined over ``samples``.  Build the reference
    # training subset from those exact candidate tensors; using positional
    # indices into train+full-test would select the wrong nonmembers whenever
    # the target has a larger test pool (as in the Credit low-FPR protocol).
    candidate_dataset = CandidateDataset(samples.inputs, samples.labels)
    train_dataset = torch.utils.data.Subset(candidate_dataset, np.flatnonzero(inclusion).tolist())
    seed_row = dict(row)
    seed_row["model_seed"] = int(reference_seed)
    set_all_seeds(reference_seed)
    model, _ = instantiate_model(qmain, seed_row, config, device)
    learning_rate = float(spec["learning_rate"])
    weight_decay = row_number(row, "weight_decay", 0.0)
    epochs = int(spec["epochs"])
    batch_size = int(spec["batch_size"])
    generator = torch.Generator()
    generator.manual_seed(reference_seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset["valid"], batch_size=batch_size, shuffle=False, num_workers=0
    )
    dataflow = {"train": train_loader, "valid": valid_loader}
    if spec["optimizer"] == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = None
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    mode = str(spec["mode"])
    dp_state = None
    if mode == "dp_qml":
        dp_state = _initialize_reference_dp(
            row,
            model,
            optimizer,
            train_dataset,
            batch_size=batch_size,
            epochs=epochs,
            seed=reference_seed,
        )
    final_train_loss = final_train_acc = float("nan")
    final_valid_loss = final_valid_acc = float("nan")
    for epoch in range(1, epochs + 1):
        final_train_loss, final_train_acc = _reference_epoch(
            mode, model, dataflow, device, optimizer, qmain, row=row, dp_state=dp_state
        )
        final_valid_loss, final_valid_acc = qmain.evaluate(
            dataflow, "valid", model, device
        )
        if scheduler is not None:
            scheduler.step()
        if epoch == 1 or epoch == epochs or epoch % max(epochs // 10, 1) == 0:
            print(
                f"[reference {args.reference_id:03d}] epoch={epoch}/{epochs} "
                f"train_acc={final_train_acc:.3f} valid_acc={final_valid_acc:.3f}",
                flush=True,
            )

    decision_rule = reference_decision_rule(
        model,
        dataset["valid"],
        device=device,
        batch_size=batch_size,
    )
    decision_threshold = (
        None if decision_rule is None else float(decision_rule["threshold"])
    )
    privacy_report = None
    if dp_state is not None:
        privacy_report = dict(dp_state["session"].privacy_report())
        privacy_report["target_epsilon"] = row_number(
            row, "dp_target_epsilon", 4.0
        )

    probabilities = exact_probabilities(
        model, samples, device=device, batch_size=batch_size
    ).numpy()
    labels = samples.labels.numpy()
    scores = true_class_log_odds(probabilities, labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            scores=scores.astype(np.float32),
            inclusion=inclusion.astype(np.uint8),
            labels=labels.astype(np.int64),
            membership=samples.membership.numpy().astype(np.uint8),
            sample_ids=np.asarray(samples.sample_ids),
            candidate_fingerprint=np.asarray(
                candidate_fingerprint
            ),
            structural_cell=np.asarray(structural_cell),
            reference_pairing_id=np.asarray(pairing_id),
            training_signature=np.asarray(signature),
            training_defense=np.asarray(mode),
            target_template=np.asarray(args.target_id),
            reference_id=np.asarray(args.reference_id),
            num_references=np.asarray(args.num_references),
            reference_seed=np.asarray(reference_seed),
            epochs=np.asarray(epochs),
            batch_size=np.asarray(batch_size),
            learning_rate=np.asarray(learning_rate),
            optimizer=np.asarray(str(spec["optimizer"])),
            scheduler=np.asarray(str(spec["scheduler"])),
            weight_decay=np.asarray(weight_decay),
            train_size=np.asarray(int(inclusion.sum())),
            final_train_loss=np.asarray(final_train_loss),
            final_train_acc=np.asarray(final_train_acc),
            final_valid_loss=np.asarray(final_valid_loss),
            final_valid_acc=np.asarray(final_valid_acc),
            decision_threshold=np.asarray(
                np.nan if decision_threshold is None else decision_threshold
            ),
        )
    temporary.replace(output)
    if args.save_checkpoint:
        checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_temporary = checkpoint_output.with_suffix(".pt.tmp")
        torch.save(
            {
                "state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "reference_id": int(args.reference_id),
                "num_references": int(args.num_references),
                "reference_seed": int(reference_seed),
                "structural_cell": structural_cell,
                "reference_pairing_id": pairing_id,
                "training_signature": signature,
                "training_defense": mode,
                "target_template": args.target_id,
                "candidate_fingerprint": candidate_fingerprint,
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "learning_rate": float(learning_rate),
                "optimizer": str(spec["optimizer"]),
                "scheduler": str(spec["scheduler"]),
                "decision_rule": decision_rule,
                "decision_threshold": decision_threshold,
                "privacy": privacy_report,
            },
            checkpoint_temporary,
        )
        checkpoint_temporary.replace(checkpoint_output)
    print(f"[OK] reference -> {output.resolve()}")


def load_reference_bank(
    out_dir: Path, structural_cell: str, expected: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    directory = out_dir / "reference_models" / safe_name(structural_cell)
    paths = sorted(directory.glob("reference_*.npz"))
    if len(paths) != expected:
        raise FileNotFoundError(
            f"Expected {expected} reference files in {directory}, found {len(paths)}"
        )
    score_rows = []
    inclusion_rows = []
    fingerprint = None
    epochs = set()
    seeds = []
    for expected_id, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as saved:
            reference_id = int(saved["reference_id"])
            if reference_id != expected_id:
                raise ValueError(
                    f"Reference IDs are not contiguous: expected {expected_id}, got "
                    f"{reference_id} in {path}"
                )
            if int(saved["num_references"]) != expected:
                raise ValueError(f"Reference-count mismatch in {path}")
            current_fingerprint = str(saved["candidate_fingerprint"])
            if fingerprint is None:
                fingerprint = current_fingerprint
            elif current_fingerprint != fingerprint:
                raise ValueError("Reference bank contains different candidate pools")
            score_rows.append(np.asarray(saved["scores"], dtype=float))
            inclusion_rows.append(np.asarray(saved["inclusion"], dtype=bool))
            epochs.add(int(saved["epochs"]))
            seeds.append(int(saved["reference_seed"]))
    return (
        np.stack(score_rows),
        np.stack(inclusion_rows),
        {
            "candidate_fingerprint": fingerprint,
            "epochs": sorted(epochs),
            "reference_seeds": seeds,
            "paths": [str(path.resolve()) for path in paths],
        },
    )


def score_target(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    device = torch.device(args.device)
    qmain, row, _, config, samples = load_context(
        repo_root, args.targets, args.target_id, device
    )
    structural_cell = cell_id(row)
    output = args.out_dir / "target_scores" / f"{safe_name(args.target_id)}.csv"
    if args.resume and output.exists():
        existing = pd.read_csv(output)
        if (
            set(existing.get("lira_score_protocol", pd.Series(dtype=str)).astype(str))
            == {LIRA_SCORE_PROTOCOL}
            and set(existing.get("attack", pd.Series(dtype=str)).astype(str))
            == LIRA_ATTACK_NAMES
        ):
            print(f"[SKIP] target score exists: {output.resolve()}")
            return
        raise RuntimeError(
            f"stale LiRA target score must be archived before resume: {output}"
        )

    model, _ = instantiate_model(qmain, row, config, device)
    model_path, _ = resolve_target_paths(row, args.run_root)
    load_saved_model(model, model_path, device)
    batch_size = int(float(row.get("batch_size", 16)))
    target_probabilities = exact_probabilities(
        model, samples, device=device, batch_size=batch_size
    ).numpy()
    labels = samples.labels.numpy()
    membership = samples.membership.numpy().astype(int)
    observed = true_class_log_odds(target_probabilities, labels)
    reference_scores, inclusion, bank_meta = load_reference_bank(
        args.reference_dir, structural_cell, args.num_references
    )
    current_fingerprint = tensor_fingerprint(samples.inputs, samples.labels)
    if current_fingerprint != bank_meta["candidate_fingerprint"]:
        raise ValueError(
            "Target candidate pool does not match the reference bank. Check data_seed "
            "and dataset construction."
        )
    distribution = reference_distribution(reference_scores, inclusion)
    scores_by_attack = attack_scores(observed, distribution)

    sample_output = (
        args.out_dir / "sample_scores" / f"{safe_name(args.target_id)}.npz"
    )
    sample_output.parent.mkdir(parents=True, exist_ok=True)
    with sample_output.open("wb") as handle:
        np.savez_compressed(
            handle,
            membership=membership.astype(np.uint8),
            labels=labels.astype(np.int64),
            sample_ids=np.asarray(samples.sample_ids),
            observed_log_odds=observed.astype(np.float32),
            lira_score_protocol=np.asarray(LIRA_SCORE_PROTOCOL),
            **{
                attack: values.astype(np.float32)
                for attack, values in scores_by_attack.items()
            },
        )

    rows: list[dict[str, Any]] = []
    for attack, score in scores_by_attack.items():
        auc = float(roc_auc_score(membership, score))
        low, high, valid = stratified_bootstrap_auc(
            membership,
            score,
            args.bootstrap,
            stable_seed(args.seed, args.target_id, attack, "record_bootstrap"),
        )
        tpr5, attained5 = tpr_at_resolvable_fpr(membership, score, 0.05)
        tpr10, attained10 = tpr_at_resolvable_fpr(membership, score, 0.10)
        record = {
            "target_id": args.target_id,
            "experiment": row.get("experiment", ""),
            "dataset": row.get("dataset", ""),
            "architecture": row.get("architecture", ""),
            "role": row.get("role", ""),
            "seed": int(float(row.get("seed", row.get("model_seed", 0)))),
            "model_seed": int(float(row.get("model_seed", row.get("seed", 0)))),
            "data_seed": int(float(row.get("data_seed", 0))),
            "structural_cell_id": structural_cell,
            "fm_kind": row.get("fm_kind", ""),
            "reps": int(float(row.get("reps", 0))),
            "depth": int(float(row.get("depth", 0))),
            "attack": attack,
            "lira_score_protocol": LIRA_SCORE_PROTOCOL,
            "auc": auc,
            "auc_record_ci95_low": low,
            "auc_record_ci95_high": high,
            "valid_record_bootstrap_replicates": valid,
            "tpr_at_0_05_fpr": tpr5,
            "attained_fpr_for_0_05": attained5,
            "tpr_at_0_10_fpr": tpr10,
            "attained_fpr_for_0_10": attained10,
            "n_member": int((membership == 1).sum()),
            "n_nonmember": int((membership == 0).sum()),
            "num_reference_models": args.num_references,
            "reference_epochs": json.dumps(bank_meta["epochs"]),
            "reference_candidate_protocol": (
                f"{len(membership)} target-candidate records; each candidate IN in exactly half "
                f"of references; each reference trains on {len(membership) // 2} records"
            ),
            "score_statistic": "true-class log-odds",
            "ci_method": CI_RECORD,
            "sample_score_file": str(sample_output.resolve()),
        }
        record.update(
            cross_fitted_threshold_metrics(
                membership,
                score,
                5,
                stable_seed(args.seed, args.target_id, attack, "crossfit"),
            )
        )
        rows.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(pd.DataFrame(rows), output)
    print(f"[OK] {len(rows)} attacks -> {output.resolve()}")


def aggregate(args: argparse.Namespace) -> None:
    paths = sorted((args.out_dir / "target_scores").glob("*.csv"))
    if not paths:
        raise SystemExit(f"No per-target scores found under {args.out_dir / 'target_scores'}")
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if "lira_score_protocol" not in raw or set(raw.lira_score_protocol.astype(str)) != {
        LIRA_SCORE_PROTOCOL
    }:
        raise RuntimeError(
            "LiRA aggregation found stale or mixed score semantics; archive and "
            "rescore the affected targets"
        )
    attack_sets = raw.groupby("target_id").attack.agg(lambda values: set(map(str, values)))
    incomplete = attack_sets[attack_sets.map(lambda values: values != LIRA_ATTACK_NAMES)]
    if len(incomplete):
        raise RuntimeError(
            "LiRA aggregation found incomplete attack families for targets: "
            + ", ".join(map(str, incomplete.index[:10]))
        )
    raw_path = args.out_dir / "lira_reference_mia_raw.csv"
    atomic_write_csv(raw, raw_path)
    group_columns = [
        column
        for column in (
            "attack",
            "dataset",
            "architecture",
            "fm_kind",
            "reps",
            "depth",
        )
        if column in raw.columns
    ]
    metric_columns = [
        "auc",
        "tpr_at_0_05_fpr",
        "tpr_at_0_10_fpr",
        "balanced_accuracy_crossfit",
        "membership_advantage_crossfit",
    ]
    summary = (
        raw.groupby(group_columns, dropna=False)[metric_columns]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary["error_bar_type"] = "mean ± sample SD across target-model seeds"
    summary_path = args.out_dir / "lira_reference_mia_summary.csv"
    atomic_write_csv(summary, summary_path)
    metadata_path = args.out_dir / "analysis_metadata.json"
    write_analysis_metadata(
        metadata_path,
        script=Path(__file__).name,
        inputs=[str(args.targets), str(args.run_root), str(args.reference_dir)],
        outputs=[str(raw_path), str(summary_path)],
        ci_method=CI_RECORD,
        bootstrap_unit="member/non-member records within target; target seed for summary SD",
        bootstrap_replicates=args.bootstrap,
        error_bar_type="mean ± sample SD across target-model seeds",
        notes=(
            "Online/offline LiRA with per-record balanced IN/OUT reference models. "
            "Paper-defined offline LiRA uses the one-sided OUT log-CDF; explicit z-score "
            "aliases preserve the same ranking. The released-artifact negative OUT "
            "log-density score is retained under a distinct auxiliary name. "
            "References train on the union of the target-training members and an equal-size "
            "deterministic target-test candidate subset, with half included per model. This is calibrated "
            "reference-model evidence, not a claim of target/reference distribution identity."
        ),
    )
    atomic_write_json(
        {
            "reference_implementations_studied": REFERENCE_COMMITS,
            "implementation": "independent QuRiFT integration",
            "lira_score_protocol": LIRA_SCORE_PROTOCOL,
            "raw_rows": len(raw),
            "target_files": len(paths),
        },
        args.out_dir / "reference_provenance.json",
    )
    print(f"[OK] raw={len(raw)} summary={len(summary)} -> {args.out_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--repo-root", type=Path, default=Path("."))
        subparser.add_argument("--targets", type=Path, required=True)
        subparser.add_argument("--run-root", type=Path, default=Path("reviewer_runs"))
        subparser.add_argument(
            "--out-dir", type=Path, default=Path("reviewer_results/lira_reference_mia")
        )
        subparser.add_argument(
            "--reference-dir",
            type=Path,
            default=Path("reviewer_results/lira_reference_mia"),
        )
        subparser.add_argument("--num-references", type=int, default=16)
        subparser.add_argument("--seed", type=int, default=2026)
        subparser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
        subparser.add_argument("--resume", action="store_true")

    train = subparsers.add_parser("train-reference")
    common(train)
    train.add_argument("--target-id", required=True)
    train.add_argument("--reference-id", type=int, required=True)
    train.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override target-table epochs; omit for the matched full protocol.",
    )
    train.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Retain the trained reference weights for matched noisy inference.",
    )
    train.set_defaults(function=train_reference)

    score = subparsers.add_parser("score-target")
    common(score)
    score.add_argument("--target-id", required=True)
    score.add_argument("--bootstrap", type=int, default=5000)
    score.set_defaults(function=score_target)

    collect = subparsers.add_parser("aggregate")
    common(collect)
    collect.add_argument("--bootstrap", type=int, default=5000)
    collect.set_defaults(function=aggregate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_references < 4 or args.num_references % 2:
        raise SystemExit("--num-references must be even and at least 4")
    if hasattr(args, "reference_id") and not (
        0 <= args.reference_id < args.num_references
    ):
        raise SystemExit("--reference-id must be in [0, --num-references)")
    args.function(args)


if __name__ == "__main__":
    main()
