from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Sequence

from neural_network.archery_ml.contracts import DatasetSample


TUNING_SPLITS = ("train", "val", "test")


def assign_splits(
    samples: Sequence[DatasetSample],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> list[DatasetSample]:
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    if any(value < 0 for value in ratios.values()):
        raise ValueError("split ratios must be non-negative")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1.0")

    groups: dict[str, list[DatasetSample]] = {}
    for sample in samples:
        groups.setdefault(sample.group_id or sample.id, []).append(sample)

    assignments: dict[str, str] = {}
    mutable_groups: list[tuple[str, list[DatasetSample]]] = []
    for group_id, members in groups.items():
        mobile_flags = [sample.split == "mobile_real_test" for sample in members]
        if any(mobile_flags):
            if not all(mobile_flags):
                raise ValueError(
                    f"group {group_id} mixes mobile_real_test with tuning samples"
                )
            assignments[group_id] = "mobile_real_test"
        else:
            mutable_groups.append((group_id, members))

    total = sum(len(members) for _, members in mutable_groups)
    targets = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: 0 for name in TUNING_SPLITS}
    ordered_groups = sorted(
        mutable_groups,
        key=lambda item: _stable_order(seed, item[0]),
    )

    for group_id, members in ordered_groups:
        size = len(members)
        selected = min(
            TUNING_SPLITS,
            key=lambda candidate: (
                _assignment_error(counts, targets, candidate, size),
                TUNING_SPLITS.index(candidate),
            ),
        )
        assignments[group_id] = selected
        counts[selected] += size

    output = [
        replace(sample, split=assignments[sample.group_id or sample.id])
        for sample in samples
    ]
    assert_no_group_leakage(output)
    return output


def assert_no_group_leakage(samples: Sequence[DatasetSample]) -> None:
    group_splits: dict[str, set[str]] = {}
    for sample in samples:
        group_splits.setdefault(sample.group_id or sample.id, set()).add(sample.split)
    leaking = sorted(group_id for group_id, splits in group_splits.items() if len(splits) > 1)
    if leaking:
        raise ValueError(f"group leakage across splits: {leaking}")


def _stable_order(seed: int, group_id: str) -> str:
    return hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()


def _assignment_error(
    counts: dict[str, int],
    targets: dict[str, float],
    candidate: str,
    size: int,
) -> float:
    error = 0.0
    for split in TUNING_SPLITS:
        count = counts[split] + (size if split == candidate else 0)
        scale = max(targets[split], 1.0)
        error += ((count - targets[split]) / scale) ** 2
    return error
