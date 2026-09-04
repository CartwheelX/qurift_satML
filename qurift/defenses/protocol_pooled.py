"""Opt-in evaluation partition that widens the non-member side.

The frozen contract in :mod:`qurift.defenses.protocol` forces the final
evaluation partition to hold equally many members and non-members, and members
are drawn from the 200-record training split that the three partitions share.
That caps a single block's AUC standard error at roughly 0.041, which is coarser
than the effects a defense evaluation needs to resolve.

Members cannot be increased without shrinking calibration or retraining, but
non-members are cheap: they come from the 2000-record held-out test split, of
which the frozen contract uses 100.  Widening only that side lowers the standard
error to about 0.030 at no cost in training and no change to which records are
members.

This module never mutates the frozen contract.  With ``nonmember_multiplier=1``
it delegates to :func:`qurift.defenses.protocol.build_defense_partitions` and
returns exactly what the frozen path returns.  Above 1 it reproduces the frozen
random-number consumption order so that the member set, the defense-calibration
partition, and the attack-calibration partition stay bit-identical, and only the
evaluation non-member draw grows.  The enlarged run is therefore a strict
superset of the frozen run rather than a different experiment.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Dict, Sequence

import numpy as np

from .protocol import (
    CONFIRMATORY_PARTITION_PROTOCOL,
    PARTITION_PROTOCOL,
    DefensePartitions,
    RecordRef,
    _allocate_from_remaining,
    _class_pools,
    _paired_partition,
    _take_by_quota,
    build_defense_partitions,
)


POOLED_PARTITION_PROTOCOL = "pets_label_matched_defense_attack_pooled_v3"
POOLED_CONFIRMATORY_PARTITION_PROTOCOL = (
    "pets_label_matched_defense_attack_pooled_common_quota_v4"
)


@dataclass(frozen=True)
class PooledDefensePartitions:
    """Frozen-compatible partitions whose evaluation non-members may be wider."""

    defense_calibration: tuple[RecordRef, ...]
    attack_calibration: tuple[RecordRef, ...]
    final_evaluation: tuple[RecordRef, ...]
    seed: int
    nonmember_multiplier: int
    protocol: str = POOLED_PARTITION_PROTOCOL
    quota_plan_name: str | None = None

    def validate(self) -> "PooledDefensePartitions":
        groups = {
            "defense_calibration": self.defense_calibration,
            "attack_calibration": self.attack_calibration,
            "final_evaluation": self.final_evaluation,
        }
        sets = {name: {item.record_id for item in values} for name, values in groups.items()}
        for name, values in groups.items():
            if not values:
                raise ValueError(f"{name} is empty")
            if {item.membership for item in values} != {0, 1}:
                raise ValueError(f"{name} must contain member and non-member records")
            if len(sets[name]) != len(values):
                raise ValueError(f"{name} contains duplicate records")
            members = Counter(item.task_label for item in values if item.membership == 1)
            nonmembers = Counter(item.task_label for item in values if item.membership == 0)
            # Calibration partitions stay exactly matched; only the evaluation
            # partition may scale its non-member side, and then only by an exact
            # integer factor applied uniformly to every task label. That keeps the
            # label distribution identical across membership, which is what the
            # label-matching requirement is actually protecting.
            factor = self.nonmember_multiplier if name == "final_evaluation" else 1
            expected = Counter({label: count * factor for label, count in members.items()})
            if nonmembers != expected:
                raise ValueError(
                    f"{name} is not label matched at multiplier {factor}: "
                    f"members={dict(members)}, nonmembers={dict(nonmembers)}"
                )
        names = list(groups)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                overlap = sets[left] & sets[right]
                if overlap:
                    raise ValueError(f"partitions {left} and {right} overlap")
        return self

    def to_json(self) -> Dict[str, Any]:
        self.validate()
        if self.nonmember_multiplier == 1:
            # Emit the frozen manifest byte for byte, with no extra keys, so a
            # default run stays indistinguishable from one produced before this
            # module existed.
            return DefensePartitions(
                self.defense_calibration,
                self.attack_calibration,
                self.final_evaluation,
                self.seed,
                protocol=(
                    CONFIRMATORY_PARTITION_PROTOCOL
                    if self.quota_plan_name is not None
                    else PARTITION_PROTOCOL
                ),
                quota_plan_name=self.quota_plan_name,
            ).to_json()
        payload = {
            "protocol": self.protocol,
            "membership_encoding": "1=member,0=nonmember",
            "task_label_matching": (
                "exact for calibration; proportional at the declared "
                "multiplier for final evaluation"
            ),
            "evaluation_nonmember_multiplier": int(self.nonmember_multiplier),
            "seed": self.seed,
            "partitions": {
                name: [
                    {**asdict(item), "record_id": item.record_id}
                    for item in getattr(self, name)
                ]
                for name in (
                    "defense_calibration",
                    "attack_calibration",
                    "final_evaluation",
                )
            },
        }
        if self.quota_plan_name is not None:
            payload["quota_plan_name"] = self.quota_plan_name
            payload["member_task_label_quotas"] = {
                name: dict(
                    sorted(
                        Counter(
                            item.task_label
                            for item in getattr(self, name)
                            if item.membership == 1
                        ).items()
                    )
                )
                for name in (
                    "defense_calibration",
                    "attack_calibration",
                    "final_evaluation",
                )
            }
        return payload


def _widened_partition(
    member_pools: Dict[int, list[int]],
    nonmember_pools: Dict[int, list[int]],
    *,
    member_split: str,
    nonmember_split: str,
    count: int,
    multiplier: int,
    rng: np.random.Generator,
    quota: Dict[int, int] | None = None,
) -> tuple[RecordRef, ...]:
    quota = (
        _allocate_from_remaining(member_pools, int(count))
        if quota is None
        else {int(label): int(amount) for label, amount in quota.items()}
    )
    if any(amount < 0 for amount in quota.values()) or sum(quota.values()) != int(count):
        raise ValueError(
            f"explicit quota must contain {int(count)} non-negative records, got {quota}"
        )
    # The binding constraint is per task label, not the total pool size: the
    # majority class exhausts its non-member supply well before the split does.
    # Check it here so the caller gets the largest usable multiplier instead of
    # a quota error raised several frames down.
    feasible = min(
        len(nonmember_pools.get(label, [])) // amount
        for label, amount in quota.items()
        if amount > 0
    )
    if multiplier > feasible:
        raise ValueError(
            f"evaluation non-member multiplier {multiplier} exceeds the per-label "
            f"capacity of the {nonmember_split} split; the largest feasible value "
            f"for these partition sizes is {feasible}"
        )
    members = _take_by_quota(member_pools, quota, member_split, 1)
    widened = {label: int(amount) * int(multiplier) for label, amount in quota.items()}
    nonmembers = _take_by_quota(nonmember_pools, widened, nonmember_split, 0)
    # Draw the member permutation first and with the same shape the frozen path
    # uses, so the member ordering is unchanged and any prefix of the member list
    # still matches the frozen run (the HSJ subset depends on this).
    ordering = rng.permutation(len(members)).astype(int).tolist()
    members = [members[index] for index in ordering]
    nonmember_order = rng.permutation(len(nonmembers)).astype(int).tolist()
    nonmembers = [nonmembers[index] for index in nonmember_order]
    return tuple(members + nonmembers)


def build_pooled_defense_partitions(
    *,
    train_labels: Sequence[int],
    valid_labels: Sequence[int],
    test_labels: Sequence[int],
    defense_per_class: int,
    attack_per_class: int,
    evaluation_per_class: int,
    seed: int,
    nonmember_multiplier: int = 1,
    label_quotas: Dict[str, Dict[int, int]] | None = None,
    quota_plan_name: str | None = None,
) -> PooledDefensePartitions:
    """Build partitions, optionally widening only the evaluation non-members."""

    multiplier = int(nonmember_multiplier)
    if multiplier < 1:
        raise ValueError("nonmember_multiplier must be at least 1")
    if multiplier == 1:
        frozen = build_defense_partitions(
            train_labels=train_labels,
            valid_labels=valid_labels,
            test_labels=test_labels,
            defense_per_class=defense_per_class,
            attack_per_class=attack_per_class,
            evaluation_per_class=evaluation_per_class,
            seed=seed,
            label_quotas=label_quotas,
            quota_plan_name=quota_plan_name,
        )
        return PooledDefensePartitions(
            frozen.defense_calibration,
            frozen.attack_calibration,
            frozen.final_evaluation,
            frozen.seed,
            1,
            protocol=frozen.protocol,
            quota_plan_name=frozen.quota_plan_name,
        ).validate()

    train = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    valid = np.asarray(valid_labels, dtype=np.int64).reshape(-1)
    test = np.asarray(test_labels, dtype=np.int64).reshape(-1)
    counts = [int(defense_per_class), int(attack_per_class), int(evaluation_per_class)]
    if min(counts) <= 0:
        raise ValueError("all per-membership partition sizes must be positive")
    if sum(counts) > len(train):
        raise ValueError("member partitions exceed target training records")
    if counts[0] + counts[1] > len(valid):
        raise ValueError("calibration non-member partitions exceed validation records")
    if counts[2] * multiplier > len(test):
        raise ValueError(
            f"evaluation needs {counts[2] * multiplier} non-members but the test "
            f"split holds {len(test)}; lower the evaluation non-member multiplier"
        )

    # Reproduce the frozen consumption order exactly up to the evaluation draw.
    rng = np.random.default_rng(int(seed))
    train_pools = _class_pools(train, rng)
    valid_pools = _class_pools(valid, rng)
    test_pools = _class_pools(test, rng)
    quota_names = (
        "defense_calibration",
        "attack_calibration",
        "final_evaluation",
    )
    if label_quotas is not None and set(label_quotas) != set(quota_names):
        raise ValueError(
            "explicit label quotas must define defense_calibration, "
            "attack_calibration, and final_evaluation"
        )
    if label_quotas is not None:
        classes = set(np.unique(train).astype(int).tolist())
        for name, expected_count in zip(quota_names, counts):
            quota = {
                int(label): int(amount)
                for label, amount in label_quotas[name].items()
            }
            if set(quota) != classes:
                raise ValueError(
                    f"{name} explicit quota classes differ from the dataset classes"
                )
            if (
                any(amount < 0 for amount in quota.values())
                or sum(quota.values()) != expected_count
            ):
                raise ValueError(
                    f"{name} explicit quota must sum to {expected_count}: {quota}"
                )
    if label_quotas is None and quota_plan_name is not None:
        raise ValueError("quota_plan_name requires explicit label_quotas")
    defense = _paired_partition(
        train_pools, valid_pools,
        member_split="train", nonmember_split="valid", count=counts[0], rng=rng,
        quota=None if label_quotas is None else label_quotas["defense_calibration"],
    )
    attack = _paired_partition(
        train_pools, valid_pools,
        member_split="train", nonmember_split="valid", count=counts[1], rng=rng,
        quota=None if label_quotas is None else label_quotas["attack_calibration"],
    )
    evaluation = _widened_partition(
        train_pools, test_pools,
        member_split="train", nonmember_split="test",
        count=counts[2], multiplier=multiplier, rng=rng,
        quota=None if label_quotas is None else label_quotas["final_evaluation"],
    )
    return PooledDefensePartitions(
        defense,
        attack,
        evaluation,
        int(seed),
        multiplier,
        protocol=(
            POOLED_PARTITION_PROTOCOL
            if label_quotas is None
            else POOLED_CONFIRMATORY_PARTITION_PROTOCOL
        ),
        quota_plan_name=quota_plan_name,
    ).validate()


def pooled_partition_fingerprint(partitions: PooledDefensePartitions) -> str:
    record_ids = []
    for name in ("defense_calibration", "attack_calibration", "final_evaluation"):
        record_ids.extend(f"{name}:{item.record_id}" for item in getattr(partitions, name))
    return hashlib.sha256("\n".join(record_ids).encode("utf-8")).hexdigest()
