#!/usr/bin/env python3
"""Train one standard, L2, HAMP, or formally accounted DP-QML target."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, Mapping, Optional

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "reviewer_tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from qurift.defenses.dp_training import (  # noqa: E402
    DPConfig,
    DPTrainingSession,
    PoissonIndexSampler,
    calibrate_noise_multiplier,
)
from qurift.defenses.hamp import hamp_training_loss, hamp_true_probability_from_gamma  # noqa: E402
from qurift.defenses.utility import (  # noqa: E402
    calibrated_binary_utility,
    classification_utility,
    select_binary_decision_threshold,
)
from qurift_target_loader import (  # noqa: E402
    build_config,
    build_dataset,
    import_qurift_main,
    instantiate_model,
    preprocess_like_train,
    read_target_row,
    sample_dataset_split,
)


def number(row: Mapping[str, Any], key: str, default: float) -> float:
    value = row.get(key, default)
    try:
        return float(default) if pd.isna(value) else float(value)
    except Exception:
        return float(default)


def integer(row: Mapping[str, Any], key: str, default: int) -> int:
    return int(round(number(row, key, default)))


def save_model(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value for key, value in model.state_dict().items() if "q_device" not in key}
    torch.save(state, path)


def make_loader(dataset, *, batch_size: int, shuffle: bool, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        drop_last=False,
        generator=generator if shuffle else None,
    )


@torch.no_grad()
def predict_split(model, loader, device: torch.device):
    model.eval()
    outputs = []
    labels = []
    for feed in loader:
        inputs = preprocess_like_train(feed["image"], device)
        target = feed["digit"].to(device).long()
        outputs.append(model(inputs))
        labels.append(target)
    output = torch.cat(outputs)
    label = torch.cat(labels)
    return output.exp(), label, float(F.nll_loss(output, label).item())


def utility_with_decision_rule(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    loss: float,
    *,
    decision_threshold: Optional[float],
) -> Dict[str, float]:
    metrics = classification_utility(probabilities, labels) | {
        "loss": float(loss),
        "records": len(labels),
    }
    if decision_threshold is not None:
        metrics.update(
            calibrated_binary_utility(
                probabilities.detach().cpu().numpy(),
                labels.detach().cpu().numpy(),
                decision_threshold,
            )
        )
        metrics["calibrated_decision_threshold"] = float(decision_threshold)
    return metrics


def standard_epoch(
    model,
    loader,
    optimizer,
    device,
    *,
    mode: str,
    hamp_true_probability: float,
    hamp_entropy_weight: float,
):
    model.train()
    total_loss = 0.0
    total = 0
    for feed in loader:
        inputs = preprocess_like_train(feed["image"], device)
        labels = feed["digit"].to(device).long()
        output = model(inputs)
        if mode == "hamp_train":
            loss, _ = hamp_training_loss(
                output,
                labels,
                true_class_probability=hamp_true_probability,
                entropy_weight=hamp_entropy_weight,
            )
        else:
            loss = F.nll_loss(output, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item()) * len(labels)
        total += len(labels)
    return total_loss / max(total, 1)


def dp_epoch(
    model,
    dataset,
    session: DPTrainingSession,
    sampler: PoissonIndexSampler,
    *,
    steps: int,
    device: torch.device,
):
    model.train()
    diagnostics = []
    for _ in range(int(steps)):
        indices = sampler.sample()
        if len(indices) == 0:
            diagnostics.append(session.step(torch.empty(0, device=device)))
            continue
        inputs, labels = sample_dataset_split(dataset, indices.tolist())
        inputs = preprocess_like_train(inputs, device)
        labels = labels.to(device)
        losses = F.nll_loss(model(inputs), labels, reduction="none")
        diagnostics.append(session.step(losses))
    return {
        "mean_sampled_batch": float(np.mean([value["batch_size"] for value in diagnostics])),
        "mean_clipped_fraction": float(
            np.mean([value.get("clipped_fraction", 0.0) for value in diagnostics])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--run-root", type=Path, default=Path("pets_runs"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--epochs-override",
        type=int,
        default=None,
        help="Smoke-test override; production runs use the frozen target-table value.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    targets = args.targets if args.targets.is_absolute() else repo_root / args.targets
    run_root = args.run_root if args.run_root.is_absolute() else repo_root / args.run_root
    row = read_target_row(targets, args.target_id)
    output_dir = run_root / str(row.get("experiment")) / args.target_id
    model_path = output_dir / "target_model.pt"
    metadata_path = output_dir / "training_metadata.json"
    if args.resume and model_path.exists() and metadata_path.exists():
        print(f"[SKIP] completed target: {args.target_id}")
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    model_seed = integer(row, "model_seed", 43)
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)

    qmain = import_qurift_main(repo_root)
    dataset, feature_dim = build_dataset(qmain, row, repo_root)
    cfg = build_config(qmain, row, feature_dim, device)
    model, architecture = instantiate_model(qmain, row, cfg, device)
    if architecture != "qnn":
        raise NotImplementedError("PETS training pilot currently supports QNN targets")
    mode = str(row.get("training_defense", "none")).strip().lower()
    if mode not in {"none", "l2", "hamp_train", "dp_qml"}:
        raise ValueError(f"unsupported training defense {mode!r}")
    if mode == "dp_qml":
        batch_size = integer(row, "dp_batch_size", 32)
        configured_epochs = integer(row, "dp_epochs", 30)
        learning_rate = number(row, "dp_learning_rate", 0.05)
        optimizer_name = str(row.get("dp_optimizer", "rmsprop")).strip().lower()
        scheduler_name = str(row.get("dp_scheduler", "none")).strip().lower()
        if optimizer_name != "rmsprop" or scheduler_name != "none":
            raise ValueError(
                "faithful Watkins DP-QML requires RMSprop with no learning-rate scheduler"
            )
    else:
        batch_size = integer(row, "batch_size", 16)
        configured_epochs = integer(row, "epochs", 100)
        learning_rate = number(row, "learning_rate", 0.05)
        optimizer_name = "adam"
        scheduler_name = "cosine_annealing"
    epochs = configured_epochs if args.epochs_override is None else args.epochs_override
    if epochs <= 0:
        parser.error("epoch count must be positive")
    weight_decay = number(row, "weight_decay", 0.0)
    if mode == "dp_qml":
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = None
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_loader = make_loader(
        dataset["train"], batch_size=batch_size, shuffle=True, seed=model_seed
    )
    evaluation_loaders = {
        split: make_loader(dataset[split], batch_size=batch_size, shuffle=False, seed=model_seed)
        for split in ("train", "valid", "test")
    }
    history = []
    privacy_report = None
    num_classes = int(cfg.num_classes)
    hamp_gamma = number(row, "hamp_gamma", 0.95)
    hamp_alpha = number(row, "hamp_alpha", 0.001)
    hamp_true_probability = hamp_true_probability_from_gamma(hamp_gamma, num_classes)
    if mode == "dp_qml":
        sample_rate = batch_size / len(dataset["train"])
        target_epsilon = number(row, "dp_target_epsilon", 4.0)
        if target_epsilon <= 0:
            raise ValueError("dp_target_epsilon must be positive")
        try:
            from opacus.accountants.utils import get_noise_multiplier
        except ImportError as error:
            raise RuntimeError(
                "DP target-epsilon calibration requires opacus==1.5.4"
            ) from error
        initial_noise_multiplier = float(
            get_noise_multiplier(
                target_epsilon=target_epsilon,
                target_delta=number(row, "dp_delta", 1e-5),
                sample_rate=sample_rate,
                epochs=epochs,
                accountant="rdp",
            )
        )
        steps_per_epoch = math.ceil(len(dataset["train"]) / batch_size)
        noise_multiplier, calibrated_epsilon = calibrate_noise_multiplier(
            target_epsilon=target_epsilon,
            delta=number(row, "dp_delta", 1e-5),
            sample_rate=sample_rate,
            steps=steps_per_epoch * epochs,
            initial_noise_multiplier=initial_noise_multiplier,
        )
        dp_config = DPConfig(
            max_grad_norm=number(row, "dp_max_grad_norm", 1.0),
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            delta=number(row, "dp_delta", 1e-5),
            expected_batch_size=batch_size,
        )
        session = DPTrainingSession(model, optimizer, dp_config)
        sampler = PoissonIndexSampler(
            len(dataset["train"]), sample_rate
        )
        for epoch in range(1, epochs + 1):
            diagnostics = dp_epoch(
                model,
                dataset["train"],
                session,
                sampler,
                steps=steps_per_epoch,
                device=device,
            )
            history.append({"epoch": epoch, **diagnostics})
            print(f"epoch={epoch}/{epochs} sampled={diagnostics['mean_sampled_batch']:.2f}", flush=True)
        privacy_report = session.privacy_report()
        privacy_report["target_epsilon"] = target_epsilon
        privacy_report["calibrated_epsilon_before_training"] = calibrated_epsilon
        privacy_report["noise_multiplier_source"] = (
            "exact_step_rdp_binary_search_initialized_by_opacus_get_noise_multiplier"
        )
    else:
        for epoch in range(1, epochs + 1):
            loss = standard_epoch(
                model,
                train_loader,
                optimizer,
                device,
                mode=mode,
                hamp_true_probability=hamp_true_probability,
                hamp_entropy_weight=hamp_alpha,
            )
            if scheduler is not None:
                scheduler.step()
            history.append({"epoch": epoch, "training_objective": loss})
            print(f"epoch={epoch}/{epochs} objective={loss:.6f}", flush=True)

    split_outputs = {
        split: predict_split(model, loader, device)
        for split, loader in evaluation_loaders.items()
    }
    decision_rule = None
    decision_threshold = None
    valid_probabilities, valid_labels, _ = split_outputs["valid"]
    if valid_probabilities.shape[1] == 2:
        decision_rule = select_binary_decision_threshold(
            valid_probabilities.detach().cpu().numpy(),
            valid_labels.detach().cpu().numpy(),
        )
        decision_rule.update(
            {
                "selection_split": "valid",
                "score": "class_1_probability",
                "test_records_consulted": 0,
                "attack_metrics_consulted": False,
            }
        )
        decision_threshold = float(decision_rule["threshold"])
    metrics = {
        split: utility_with_decision_rule(
            probabilities,
            labels,
            loss,
            decision_threshold=decision_threshold,
        )
        for split, (probabilities, labels, loss) in split_outputs.items()
    }
    save_model(model, model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "protocol": "pets_defended_target_training_v2",
        "target_id": args.target_id,
        "training_defense": mode,
        "membership_encoding": "1=member,0=nonmember in PETS analyses",
        "model_seed": model_seed,
        "data_seed": integer(row, "data_seed", 43),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer_name,
        "scheduler": scheduler_name,
        "weight_decay": weight_decay,
        "hamp_gamma": hamp_gamma,
        "hamp_alpha": hamp_alpha,
        "hamp_true_probability_derived": hamp_true_probability,
        "dp_target_epsilon": number(row, "dp_target_epsilon", 4.0),
        "metrics": metrics,
        "privacy": privacy_report,
        "decision_rule": decision_rule,
        "history": history,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[DONE] target={args.target_id} model={model_path.resolve()}")


if __name__ == "__main__":
    main()
