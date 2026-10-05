from __future__ import annotations

from collections.abc import Iterable


def expanded_sample_indices(
    base_indices: Iterable[int],
    base_len: int,
    num_aug: int,
    include_augmented: bool,
) -> list[int]:
    indices = list(base_indices)
    if not include_augmented:
        return indices

    expanded = []
    for aug_idx in range(1 + num_aug):
        expanded.extend(base_idx + aug_idx * base_len for base_idx in indices)
    return expanded
