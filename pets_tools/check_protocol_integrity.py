#!/usr/bin/env python3
"""Verify that trained models, manifests, and attack artifacts agree.

A defense evaluation is only meaningful if every arm is a single, consistently
specified condition and the manifest actually describes the checkpoints on disk.
Both properties are easy to lose across months of incremental runs, and neither
is checked anywhere else in the pipeline: training reads the manifest row and
writes whatever it was given, so a manifest edited after a run leaves no trace.

Four failure classes are reported.

  manifest_drift      A trained model's recorded hyperparameters differ from the
                      manifest row that claims to describe it. Anyone
                      reproducing from the manifest gets a different model.
  arm_inconsistent    One training defense uses different hyperparameters in
                      different blocks, so pooling those blocks averages over
                      two distinct conditions and reports them as one.
  reference_mismatch  A LiRA reference bank is missing, or holds a different
                      number of references than its siblings. LiRA calibration
                      assumes references are drawn like the target.
  missing_artifact    A manifest row has no trained model, or a trained model
                      has no evaluation output.

Exit status is non-zero when any blocking issue is found, so this can gate a
protocol freeze.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "reviewer_tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from qurift_lira_attack import (  # noqa: E402
    LIRA_ATTACK_NAMES,
    LIRA_SCORE_PROTOCOL,
    cell_id,
    reference_pairing_id,
    reference_training_spec,
    training_signature,
)
from qurift.defenses.protocol import (  # noqa: E402
    CONFIRMATORY_CREDIT_LABEL_QUOTAS,
    CONFIRMATORY_CREDIT_QUOTA_PLAN,
    CONFIRMATORY_PARTITION_PROTOCOL,
)
from qurift.defenses.protocol_pooled import (  # noqa: E402
    POOLED_CONFIRMATORY_PARTITION_PROTOCOL,
)


V2_CONFIRMATORY_PROTOCOL = "pets_credit_three_regime_v2"
V2_EVALUATION_PROTOCOL = "pets_defense_evaluation_v5"
V2_HSJ_PROTOCOL = "pets_defended_label_only_hsj_v5"
V2_LIRA_PROTOCOL = "pets_adaptive_defended_lira_v5"
V2_QUERY_PROTOCOL = "pets_nearby_query_stress_v5"
V2_BASE_PARTITION_PROTOCOL = CONFIRMATORY_PARTITION_PROTOCOL
V2_POOLED_PARTITION_PROTOCOL = POOLED_CONFIRMATORY_PARTITION_PROTOCOL
V2_EVALUATION_NONMEMBER_MULTIPLIER = 10
V2_QUOTA_PLAN = CONFIRMATORY_CREDIT_QUOTA_PLAN
V2_LABEL_QUOTAS = {
    partition: {int(label): int(count) for label, count in quota.items()}
    for partition, quota in CONFIRMATORY_CREDIT_LABEL_QUOTAS.items()
}
V2_BASE_ATTACKS = {
    "loss",
    "confidence",
    "maximum_probability",
    "entropy",
    "margin",
    "correctness",
    "learned_pv_stats_logistic",
}
V2_LIRA_ATTACKS = set(LIRA_ATTACK_NAMES)
V2_OUTPUT_CONDITIONS = {
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
V2_HSJ_OUTPUT_CONDITIONS = {
    "none",
    "dynanoise",
    "lattice_round",
    "memgq_lattice_sticky",
}
V2_QUERY_CONDITIONS = {
    "none",
    "logitguard_continuous",
    "logitguard_quantized",
    "measurementguard_continuous",
    "lattice_round",
    "memgq_lattice",
    "memgq_lattice_sticky",
}
V2_LIRA_OUTPUT_CONDITIONS = {
    "none",
    "dynanoise",
    "memguard",
    "logitguard_continuous",
    "logitguard_quantized",
    "measurementguard_continuous",
    "lattice_round",
    "memgq_lattice",
    "memgq_lattice_sticky",
}


# Fields that define the trained condition. A divergence in any of these means
# the checkpoint is not the model the manifest describes.
TRAINING_FIELDS = (
    "training_defense",
    "weight_decay",
    "learning_rate",
    "batch_size",
    "epochs",
    "optimizer",
    "scheduler",
    "model_seed",
    "data_seed",
    "hamp_gamma",
    "hamp_alpha",
    "dp_target_epsilon",
    "dp_max_grad_norm",
    "dp_delta",
)

# Fields that must agree across every block within one training-defense arm.
ARM_FIELDS = (
    "weight_decay",
    "learning_rate",
    "batch_size",
    "epochs",
    "optimizer",
    "scheduler",
    "hamp_gamma",
    "hamp_alpha",
    "dp_target_epsilon",
    "dp_max_grad_norm",
    "dp_delta",
)


def comparable(value: Any) -> Any:
    """Normalize so 0.01 and '0.01' and 1e-2 compare equal."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return round(float(value), 12)
    except (TypeError, ValueError):
        return str(value).strip().lower()


def load_metadata(run_root: Path, experiment: str, target_id: str) -> Dict[str, Any] | None:
    path = run_root / experiment / target_id / "training_metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


# The DP arm reads its own schedule columns, so a dp_qml checkpoint recording
# batch_size=32 is faithful to a manifest whose dp_batch_size is 32 even though
# its batch_size column says 16. Compare against the column the trainer actually
# consults, or every DP row reports a drift that does not exist.
DP_FIELD_SOURCE = {
    "batch_size": "dp_batch_size",
    "epochs": "dp_epochs",
    "learning_rate": "dp_learning_rate",
    "optimizer": "dp_optimizer",
    "scheduler": "dp_scheduler",
}


def manifest_value_for(row: Mapping[str, Any], field: str, training_defense: str) -> Any:
    if training_defense == "dp_qml":
        alias = DP_FIELD_SOURCE.get(field)
        if alias and alias in row:
            return row[alias]
    return row.get(field)


def check_manifest_drift(targets: pd.DataFrame, run_root: Path) -> List[Dict[str, Any]]:
    issues = []
    for _, row in targets.iterrows():
        target_id = str(row["target_id"])
        metadata = load_metadata(run_root, str(row.get("experiment", "")), target_id)
        if metadata is None:
            issues.append(
                {
                    "kind": "missing_artifact",
                    "blocking": True,
                    "target_id": target_id,
                    "detail": "manifest row has no trained model",
                }
            )
            continue
        training_defense = str(
            metadata.get("training_defense", row.get("training_defense", ""))
        ).strip().lower()
        for field in TRAINING_FIELDS:
            if field not in metadata:
                continue
            raw_manifest = manifest_value_for(row, field, training_defense)
            if raw_manifest is None:
                continue
            manifest_value = comparable(raw_manifest)
            trained_value = comparable(metadata[field])
            if manifest_value is None or trained_value is None:
                continue
            if manifest_value != trained_value:
                issues.append(
                    {
                        "kind": "manifest_drift",
                        "blocking": True,
                        "target_id": target_id,
                        "detail": (
                            f"{field}: manifest={manifest_value!r} but the trained "
                            f"model used {trained_value!r}"
                        ),
                    }
                )
    return issues


def check_arm_consistency(targets: pd.DataFrame, run_root: Path) -> List[Dict[str, Any]]:
    """Every block of one arm must be the same condition."""

    observed: Dict[tuple, Dict[Any, List[str]]] = defaultdict(lambda: defaultdict(list))
    for _, row in targets.iterrows():
        target_id = str(row["target_id"])
        metadata = load_metadata(run_root, str(row.get("experiment", "")), target_id)
        if metadata is None:
            continue
        arm = str(metadata.get("training_defense", row.get("training_defense", "")))
        role = str(row.get("defense_structural_role", row.get("role", "")))
        for field in ARM_FIELDS:
            if field not in metadata:
                continue
            value = comparable(metadata[field])
            if value is None:
                continue
            observed[(arm, role, field)][value].append(target_id)

    issues = []
    for (arm, role, field), by_value in sorted(observed.items()):
        if len(by_value) <= 1:
            continue
        spread = "; ".join(
            f"{value!r} in {sorted(targets_)}" for value, targets_ in sorted(by_value.items())
        )
        issues.append(
            {
                "kind": "arm_inconsistent",
                "blocking": True,
                "target_id": f"{arm}/{role}",
                "detail": (
                    f"{field} takes {len(by_value)} different values within one arm: {spread}"
                ),
            }
        )
    return issues


def check_reference_banks(
    targets: pd.DataFrame,
    reference_root: Path,
    expected_references: int | None,
) -> List[Dict[str, Any]]:
    """LiRA banks must exist for scored arms and hold a uniform reference count."""

    issues = []
    models_root = reference_root / "reference_models"
    strict_reference_metadata = (
        "confirmatory_protocol" in targets
        and set(targets.confirmatory_protocol.astype(str))
        == {V2_CONFIRMATORY_PROTOCOL}
    )
    if not models_root.exists():
        if expected_references is not None:
            issues.append(
                {
                    "kind": "reference_mismatch",
                    "blocking": True,
                    "target_id": "lira_reference_models",
                    "detail": f"reference root is absent: {models_root}",
                }
            )
        return issues
    counts: Dict[str, int] = {}
    expected_banks = {
        cell_id(row): (
            training_signature(row),
            reference_training_spec(row),
            reference_pairing_id(row),
        )
        for _, row in targets.iterrows()
    }
    for name, (signature, spec, pairing_id) in sorted(expected_banks.items()):
        bank = models_root / name
        npz_paths = sorted(bank.glob("reference_*.npz")) if bank.exists() else []
        npz_count = len(npz_paths)
        checkpoint_count = len(list(bank.glob("reference_*.pt"))) if bank.exists() else 0
        counts[name] = npz_count
        if expected_references is not None and (
            npz_count != expected_references or checkpoint_count != expected_references
        ):
            issues.append(
                {
                    "kind": "reference_mismatch",
                    "blocking": True,
                    "target_id": name,
                    "detail": (
                        f"expected {expected_references} score/checkpoint references, "
                        f"found npz={npz_count}, pt={checkpoint_count}"
                    ),
                }
            )
        if expected_references is not None and npz_count == expected_references:
            try:
                import numpy as np

                for reference_id, path in enumerate(npz_paths):
                    with np.load(path, allow_pickle=False) as saved:
                        if int(saved["reference_id"]) != reference_id:
                            raise ValueError(
                                f"non-contiguous reference id in {path.name}"
                            )
                        if int(saved["num_references"]) != expected_references:
                            raise ValueError(
                                f"reference-count metadata mismatch in {path.name}"
                            )
                        if strict_reference_metadata:
                            if str(saved["training_signature"]) != signature:
                                raise ValueError(
                                    f"training signature mismatch in {path.name}"
                                )
                            if str(saved["reference_pairing_id"]) != pairing_id:
                                raise ValueError(
                                    f"reference pairing mismatch in {path.name}"
                                )
                            if int(saved["epochs"]) != int(spec["epochs"]):
                                raise ValueError(f"epoch mismatch in {path.name}")
                            if int(saved["batch_size"]) != int(spec["batch_size"]):
                                raise ValueError(f"batch-size mismatch in {path.name}")
                            if abs(
                                float(saved["learning_rate"])
                                - float(spec["learning_rate"])
                            ) > 1e-12:
                                raise ValueError(f"learning-rate mismatch in {path.name}")
                            if str(saved["optimizer"]) != str(spec["optimizer"]):
                                raise ValueError(f"optimizer mismatch in {path.name}")
                            if str(saved["scheduler"]) != str(spec["scheduler"]):
                                raise ValueError(f"scheduler mismatch in {path.name}")
                            threshold = float(saved["decision_threshold"])
                            if not 0.0 < threshold < 1.0:
                                raise ValueError(
                                    f"invalid reference decision threshold in {path.name}"
                                )
            except Exception as error:
                issues.append(
                    {
                        "kind": "reference_mismatch",
                        "blocking": True,
                        "target_id": name,
                        "detail": f"invalid reference metadata: {type(error).__name__}: {error}",
                    }
                )
    if not counts:
        return issues
    sizes = sorted(set(counts.values()))
    if len(sizes) > 1:
        spread = "; ".join(
            f"{size} references in {sorted(n for n, c in counts.items() if c == size)}"
            for size in sizes
        )
        issues.append(
            {
                "kind": "reference_mismatch",
                "blocking": True,
                "target_id": "lira_reference_models",
                "detail": f"reference banks differ in size: {spread}",
            }
        )
    else:
        issues.append(
            {
                "kind": "reference_mismatch",
                "blocking": False,
                "target_id": "lira_reference_models",
                "detail": f"all {len(counts)} banks hold {sizes[0]} references",
            }
        )
    return issues


def check_evaluation_artifacts(
    targets: pd.DataFrame,
    run_root: Path,
    results_root: Path,
    *,
    allow_missing: bool,
) -> List[Dict[str, Any]]:
    issues = []
    if not results_root.exists():
        return issues
    for _, row in targets.iterrows():
        target_id = str(row["target_id"])
        if load_metadata(run_root, str(row.get("experiment", "")), target_id) is None:
            continue
        metrics = results_root / target_id / "adaptive_attack_metrics.csv"
        if not metrics.exists():
            issues.append(
                {
                    "kind": "missing_artifact",
                    "blocking": not allow_missing,
                    "target_id": target_id,
                    "detail": "trained model has no adaptive_attack_metrics.csv",
                }
            )
    return issues


def check_decision_rules(
    targets: pd.DataFrame, run_root: Path, results_root: Path
) -> List[Dict[str, Any]]:
    """All binary consumers must use the checkpoint's frozen label rule."""

    issues: List[Dict[str, Any]] = []
    for _, row in targets.iterrows():
        target_id = str(row["target_id"])
        metadata = load_metadata(run_root, str(row.get("experiment", "")), target_id)
        if metadata is None:
            continue
        rule = metadata.get("decision_rule")
        if not isinstance(rule, Mapping):
            issues.append(
                {
                    "kind": "decision_rule_mismatch",
                    "blocking": True,
                    "target_id": target_id,
                    "detail": "binary checkpoint has no validation-frozen decision rule",
                }
            )
            continue
        threshold = float(rule.get("threshold", float("nan")))
        if not 0.0 < threshold < 1.0:
            issues.append(
                {
                    "kind": "decision_rule_mismatch",
                    "blocking": True,
                    "target_id": target_id,
                    "detail": f"invalid decision threshold {threshold}",
                }
            )
            continue
        if rule.get("selection_split") != "valid" or int(
            rule.get("test_records_consulted", -1)
        ) != 0:
            issues.append(
                {
                    "kind": "decision_rule_mismatch",
                    "blocking": True,
                    "target_id": target_id,
                    "detail": "decision threshold provenance is not validation-only",
                }
            )
        target_dir = results_root / target_id
        evaluation_path = target_dir / "evaluation_metadata.json"
        if evaluation_path.exists():
            evaluation = json.loads(evaluation_path.read_text())
            recorded = evaluation.get("target_decision_threshold")
            if recorded is None or abs(float(recorded) - threshold) > 1e-12:
                issues.append(
                    {
                        "kind": "decision_rule_mismatch",
                        "blocking": True,
                        "target_id": target_id,
                        "detail": (
                            f"evaluation threshold={recorded!r}, checkpoint threshold={threshold}"
                        ),
                    }
                )
            for filename in ("final_predictions.csv", "test_utility_predictions.csv"):
                prediction_path = target_dir / filename
                if not prediction_path.exists():
                    continue
                predictions = pd.read_csv(prediction_path)
                if not {"probability_1", "predicted_label"}.issubset(predictions):
                    continue
                expected = (predictions.probability_1.astype(float) >= threshold).astype(int)
                mismatches = int((expected != predictions.predicted_label.astype(int)).sum())
                if mismatches:
                    issues.append(
                        {
                            "kind": "decision_rule_mismatch",
                            "blocking": True,
                            "target_id": target_id,
                            "detail": f"{filename} has {mismatches} labels inconsistent with threshold",
                        }
                    )
        for artifact in (
            *sorted((target_dir / "hsj").glob("*_metrics.json")),
            *sorted((target_dir / "lira").glob("*_metrics.json")),
            target_dir / "query_stress" / "metrics.json",
        ):
            if not artifact.exists():
                continue
            payload = json.loads(artifact.read_text())
            recorded = payload.get("target_decision_threshold")
            if recorded is None or abs(float(recorded) - threshold) > 1e-12:
                issues.append(
                    {
                        "kind": "decision_rule_mismatch",
                        "blocking": True,
                        "target_id": target_id,
                        "detail": f"{artifact.name} does not use the checkpoint threshold",
                    }
                )
    return issues


def _blocking(target_id: str, detail: str) -> Dict[str, Any]:
    return {
        "kind": "confirmatory_incomplete",
        "blocking": True,
        "target_id": target_id,
        "detail": detail,
    }


def _load_json(path: Path, *, target_id: str, issues: List[Dict[str, Any]]):
    if not path.exists():
        issues.append(_blocking(target_id, f"missing artifact {path}"))
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        issues.append(_blocking(target_id, f"unreadable {path}: {type(error).__name__}"))
        return None


def _expected_evaluation_conditions(training_defense: str) -> set[str]:
    if training_defense == "none":
        return set(V2_OUTPUT_CONDITIONS)
    if training_defense == "hamp_train":
        return {"none", "hamp_output"}
    return {"none"}


def _expected_lira_conditions(training_defense: str) -> set[str]:
    if training_defense == "none":
        return set(V2_LIRA_OUTPUT_CONDITIONS)
    if training_defense == "hamp_train":
        return {"none", "hamp_output"}
    return {"none"}


def check_confirmatory_v2_completeness(
    targets: pd.DataFrame,
    results_root: Path,
    *,
    expected_references: int | None,
) -> List[Dict[str, Any]]:
    """Require every predeclared target/defense/attack artifact before analysis."""

    if "confirmatory_protocol" not in targets:
        return []
    protocols = set(targets.confirmatory_protocol.astype(str))
    if protocols != {V2_CONFIRMATORY_PROTOCOL}:
        return []

    issues: List[Dict[str, Any]] = []
    if len(targets) != 96:
        issues.append(_blocking("manifest", f"expected 96 target rows, found {len(targets)}"))
    if targets.block_id.nunique() != 8:
        issues.append(
            _blocking(
                "manifest",
                f"expected 8 independent blocks, found {targets.block_id.nunique()}",
            )
        )
    if expected_references != 16:
        issues.append(
            _blocking(
                "lira_reference_models",
                f"v2 headline protocol requires expected-references=16, got {expected_references}",
            )
        )

    for _, row in targets.iterrows():
        target_id = str(row["target_id"])
        training = str(row.get("training_defense", "none")).strip().lower()
        target_dir = results_root / target_id

        evaluation_path = target_dir / "evaluation_metadata.json"
        evaluation = _load_json(evaluation_path, target_id=target_id, issues=issues)
        if evaluation is not None:
            if evaluation.get("protocol") != V2_EVALUATION_PROTOCOL:
                issues.append(
                    _blocking(target_id, f"evaluation protocol is {evaluation.get('protocol')!r}")
                )
            if evaluation.get("partition_protocol") != V2_POOLED_PARTITION_PROTOCOL:
                issues.append(_blocking(target_id, "evaluation is not the widened pooled protocol"))
            if int(evaluation.get("evaluation_nonmember_multiplier", -1)) != (
                V2_EVALUATION_NONMEMBER_MULTIPLIER
            ):
                issues.append(
                    _blocking(
                        target_id,
                        "evaluation non-member multiplier is not the frozen value 10",
                    )
                )
            observed = set(evaluation.get("conditions", {}))
            expected = _expected_evaluation_conditions(training)
            if observed != expected:
                issues.append(
                    _blocking(
                        target_id,
                        f"evaluation conditions differ: missing={sorted(expected-observed)}, "
                        f"extra={sorted(observed-expected)}",
                    )
                )

        metrics_path = target_dir / "adaptive_attack_metrics.csv"
        if metrics_path.exists():
            try:
                metrics = pd.read_csv(metrics_path)
                for condition in _expected_evaluation_conditions(training):
                    observed_attacks = set(
                        metrics.loc[
                            metrics.defense.astype(str).eq(condition), "attack"
                        ].astype(str)
                    )
                    missing = V2_BASE_ATTACKS - observed_attacks
                    if missing:
                        issues.append(
                            _blocking(
                                target_id,
                                f"condition {condition} lacks attacks {sorted(missing)}",
                            )
                        )
            except Exception as error:
                issues.append(
                    _blocking(target_id, f"could not validate adaptive attacks: {error}")
                )

        expected_hsj = V2_HSJ_OUTPUT_CONDITIONS if training == "none" else {"none"}
        for condition in expected_hsj:
            path = target_dir / "hsj" / f"{condition}_metrics.json"
            payload = _load_json(path, target_id=target_id, issues=issues)
            if payload is not None and payload.get("protocol") != V2_HSJ_PROTOCOL:
                issues.append(_blocking(target_id, f"stale HSJ protocol for {condition}"))

        if training == "none":
            path = target_dir / "query_stress" / "metrics.json"
            payload = _load_json(path, target_id=target_id, issues=issues)
            if payload is not None:
                if payload.get("protocol") != V2_QUERY_PROTOCOL:
                    issues.append(_blocking(target_id, "stale query-stress protocol"))
                observed = {str(item.get("defense")) for item in payload.get("rows", [])}
                if observed != V2_QUERY_CONDITIONS:
                    issues.append(
                        _blocking(
                            target_id,
                            f"query conditions differ: missing={sorted(V2_QUERY_CONDITIONS-observed)}, "
                            f"extra={sorted(observed-V2_QUERY_CONDITIONS)}",
                        )
                    )

        for condition in _expected_lira_conditions(training):
            path = target_dir / "lira" / f"{condition}_metrics.json"
            payload = _load_json(path, target_id=target_id, issues=issues)
            if payload is None:
                continue
            if payload.get("protocol") != V2_LIRA_PROTOCOL:
                issues.append(_blocking(target_id, f"stale LiRA protocol for {condition}"))
            if payload.get("lira_score_protocol") != LIRA_SCORE_PROTOCOL:
                issues.append(
                    _blocking(target_id, f"stale LiRA score semantics for {condition}")
                )
            if int(payload.get("num_references", -1)) != 16:
                issues.append(_blocking(target_id, f"{condition} LiRA did not use 16 references"))
            observed = {str(item.get("attack")) for item in payload.get("rows", [])}
            if observed != V2_LIRA_ATTACKS:
                issues.append(
                    _blocking(
                        target_id,
                        f"{condition} LiRA attacks differ: missing={sorted(V2_LIRA_ATTACKS-observed)}, "
                        f"extra={sorted(observed-V2_LIRA_ATTACKS)}",
                    )
                )
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("pets_targets/credit_defense_training_targets.csv"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("pets_runs"))
    parser.add_argument("--results-root", type=Path, default=Path("pets_results/defenses"))
    parser.add_argument(
        "--reference-root", type=Path, default=Path("pets_results/lira_references")
    )
    parser.add_argument("--expected-references", type=int, default=None)
    parser.add_argument("--allow-missing-evaluations", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("pets_results/protocol_integrity.csv"))
    args = parser.parse_args()

    targets = pd.read_csv(args.targets)
    issues: List[Dict[str, Any]] = []
    issues += check_manifest_drift(targets, args.run_root)
    issues += check_arm_consistency(targets, args.run_root)
    issues += check_reference_banks(
        targets, args.reference_root, args.expected_references
    )
    issues += check_evaluation_artifacts(
        targets,
        args.run_root,
        args.results_root,
        allow_missing=args.allow_missing_evaluations,
    )
    issues += check_decision_rules(targets, args.run_root, args.results_root)
    if not args.allow_missing_evaluations:
        issues += check_confirmatory_v2_completeness(
            targets,
            args.results_root,
            expected_references=args.expected_references,
        )

    frame = pd.DataFrame(issues) if issues else pd.DataFrame(
        columns=["kind", "blocking", "target_id", "detail"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    blocking = frame[frame.blocking] if len(frame) else frame
    print(f"targets checked: {len(targets)}   issues: {len(frame)}   blocking: {len(blocking)}\n")
    for kind in (
        "manifest_drift",
        "arm_inconsistent",
        "reference_mismatch",
        "missing_artifact",
        "decision_rule_mismatch",
        "confirmatory_incomplete",
    ):
        subset = frame[frame.kind.eq(kind)] if len(frame) else frame
        if not len(subset):
            continue
        print(f"--- {kind} ({len(subset)}) ---")
        for _, issue in subset.iterrows():
            flag = "BLOCKING" if issue.blocking else "note"
            print(f"  [{flag}] {issue.target_id}: {issue.detail}")
        print()

    print(f"wrote {args.out}")
    if len(blocking):
        print("\nPROTOCOL IS NOT SAFE TO FREEZE: resolve the blocking issues above.")
        sys.exit(1)
    print("\nNo blocking issues. Manifest, checkpoints, and artifacts agree.")


if __name__ == "__main__":
    main()
