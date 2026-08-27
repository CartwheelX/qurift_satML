"""Data-partition contracts for leakage-free defense evaluation."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Dict, Mapping, Sequence

import numpy as np


PARTITION_PROTOCOL = "pets_label_matched_defense_attack_final_v2"


@dataclass(frozen=True)
class RecordRef:
    split: str
    index: int
    membership: int
    task_label: int

    @property
    def record_id(self) -> str:
        value = f"{self.split}|{self.index}".encode("utf-8")
        return hashlib.sha256(value).hexdigest()[:20]


@dataclass(frozen=True)
class DefensePartitions:
    defense_calibration: tuple[RecordRef, ...]
    attack_calibration: tuple[RecordRef, ...]
    final_evaluation: tuple[RecordRef, ...]
    seed: int

    def validate(self) -> "DefensePartitions":
        groups = {
            "defense_calibration": self.defense_calibration,
            "attack_calibration": self.attack_calibration,
            "final_evaluation": self.final_evaluation,
        }
        sets = {name: {item.record_id for item in values} for name, values in groups.items()}
        for name, values in groups.items():
            if not values:
                raise ValueError(f"{name} is empty")
            memberships = {item.membership for item in values}
            if memberships != {0, 1}:
                raise ValueError(f"{name} must contain member and non-member records")
            if len(sets[name]) != len(values):
                raise ValueError(f"{name} contains duplicate records")
            member_labels = Counter(
                item.task_label for item in values if item.membership == 1
            )
            nonmember_labels = Counter(
                item.task_label for item in values if item.membership == 0
            )
            if member_labels != nonmember_labels:
                raise ValueError(
                    f"{name} task-label counts differ by membership: "
                    f"members={dict(member_labels)}, nonmembers={dict(nonmember_labels)}"
                )
        names = list(groups)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                overlap = sets[left] & sets[right]
                if overlap:
                    raise ValueError(
                        f"partitions {left} and {right} overlap: {sorted(overlap)[:3]}"
                    )
        return self

    def to_json(self) -> Dict[str, Any]:
        self.validate()
        return {
            "protocol": PARTITION_PROTOCOL,
            "membership_encoding": "1=member,0=nonmember",
            "task_label_matching": "exact within every member/nonmember partition",
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


def task_labels_from_dataset(dataset: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Read integer task labels without depending on a concrete dataset class."""

    result: Dict[str, np.ndarray] = {}
    for split in ("train", "valid", "test"):
        values = dataset[split]
        targets = getattr(values, "targets", None)
        if targets is not None:
            if hasattr(targets, "detach"):
                targets = targets.detach().cpu().numpy()
            labels = np.asarray(targets, dtype=np.int64).reshape(-1)
        else:
            labels = np.asarray(
                [int(values[index]["digit"]) for index in range(len(values))],
                dtype=np.int64,
            )
        if len(labels) != len(values):
            raise ValueError(f"{split} labels do not align with dataset records")
        result[split] = labels
    return result


def _class_pools(labels: np.ndarray, rng: np.random.Generator) -> Dict[int, list[int]]:
    return {
        int(label): rng.permutation(np.flatnonzero(labels == label)).astype(int).tolist()
        for label in sorted(np.unique(labels).tolist())
    }


def _allocate_from_remaining(pools: Mapping[int, Sequence[int]], count: int) -> Dict[int, int]:
    """Largest-remainder allocation respecting the remaining class capacities."""

    capacities = {int(label): len(indices) for label, indices in pools.items()}
    total = sum(capacities.values())
    if count <= 0 or count > total:
        raise ValueError(f"cannot allocate {count} records from a pool of {total}")
    expected = {label: count * capacity / total for label, capacity in capacities.items()}
    allocated = {
        label: min(capacities[label], int(np.floor(value)))
        for label, value in expected.items()
    }
    remaining = count - sum(allocated.values())
    order = sorted(
        capacities,
        key=lambda label: (-(expected[label] - np.floor(expected[label])), label),
    )
    while remaining:
        progressed = False
        for label in order:
            if allocated[label] < capacities[label]:
                allocated[label] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("class allocation exhausted before reaching requested size")
    return allocated


def _take_by_quota(
    pools: Dict[int, list[int]], quota: Mapping[int, int], split: str, membership: int
) -> list[RecordRef]:
    selected: list[RecordRef] = []
    for label in sorted(quota):
        amount = int(quota[label])
        if amount > len(pools.get(label, [])):
            raise ValueError(
                f"{split} lacks task-label {label} records for exact membership matching: "
                f"requested={amount}, available={len(pools.get(label, []))}"
            )
        indices = pools[label][:amount]
        del pools[label][:amount]
        selected.extend(
            RecordRef(split, int(index), int(membership), int(label))
            for index in indices
        )
    return selected


def _paired_partition(
    member_pools: Dict[int, list[int]],
    nonmember_pools: Dict[int, list[int]],
    *,
    member_split: str,
    nonmember_split: str,
    count: int,
    rng: np.random.Generator,
) -> tuple[RecordRef, ...]:
    quota = _allocate_from_remaining(member_pools, int(count))
    members = _take_by_quota(member_pools, quota, member_split, 1)
    nonmembers = _take_by_quota(nonmember_pools, quota, nonmember_split, 0)
    # One common permutation keeps every member/nonmember prefix label-matched,
    # including the smaller HSJ subset.
    ordering = rng.permutation(len(members)).astype(int).tolist()
    members = [members[index] for index in ordering]
    nonmembers = [nonmembers[index] for index in ordering]
    return tuple(members + nonmembers)


def build_defense_partitions(
    *,
    train_labels: Sequence[int],
    valid_labels: Sequence[int],
    test_labels: Sequence[int],
    defense_per_class: int,
    attack_per_class: int,
    evaluation_per_class: int,
    seed: int,
) -> DefensePartitions:
    """Create disjoint, membership-balanced, exactly task-label-matched pools."""

    train = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    valid = np.asarray(valid_labels, dtype=np.int64).reshape(-1)
    test = np.asarray(test_labels, dtype=np.int64).reshape(-1)
    if any(len(values) == 0 for values in (train, valid, test)):
        raise ValueError("train, validation, and test label arrays must be non-empty")
    classes = set(np.unique(train).tolist())
    if set(np.unique(valid).tolist()) != classes or set(np.unique(test).tolist()) != classes:
        raise ValueError("train, validation, and test splits must contain the same classes")
    counts = [int(defense_per_class), int(attack_per_class), int(evaluation_per_class)]
    if min(counts) <= 0:
        raise ValueError("all per-membership partition sizes must be positive")
    if sum(counts) > len(train):
        raise ValueError("member partitions exceed target training records")
    if counts[0] + counts[1] > len(valid):
        raise ValueError("calibration non-member partitions exceed validation records")
    if counts[2] > len(test):
        raise ValueError("final non-member partition exceeds test records")

    rng = np.random.default_rng(int(seed))
    train_pools = _class_pools(train, rng)
    valid_pools = _class_pools(valid, rng)
    test_pools = _class_pools(test, rng)
    defense = _paired_partition(
        train_pools,
        valid_pools,
        member_split="train",
        nonmember_split="valid",
        count=counts[0],
        rng=rng,
    )
    attack = _paired_partition(
        train_pools,
        valid_pools,
        member_split="train",
        nonmember_split="valid",
        count=counts[1],
        rng=rng,
    )
    evaluation = _paired_partition(
        train_pools,
        test_pools,
        member_split="train",
        nonmember_split="test",
        count=counts[2],
        rng=rng,
    )
    return DefensePartitions(defense, attack, evaluation, int(seed)).validate()


def partition_fingerprint(partitions: DefensePartitions) -> str:
    record_ids = []
    for name in ("defense_calibration", "attack_calibration", "final_evaluation"):
        record_ids.extend(f"{name}:{item.record_id}" for item in getattr(partitions, name))
    return hashlib.sha256("\n".join(record_ids).encode("utf-8")).hexdigest()
