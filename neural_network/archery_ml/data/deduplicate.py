from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from PIL import Image

from neural_network.archery_ml.contracts import DatasetSample


@dataclass(frozen=True)
class DeduplicationResult:
    samples: tuple[DatasetSample, ...]
    exact_duplicates_removed: tuple[str, ...]
    near_duplicate_groups: tuple[tuple[str, ...], ...]

    @property
    def duplicate_rate(self) -> float:
        total = len(self.samples) + len(self.exact_duplicates_removed)
        return 0.0 if total == 0 else len(self.exact_duplicates_removed) / total


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


class _BKNode:
    def __init__(self, value: int, index: int):
        self.value = value
        self.indices = [index]
        self.children: dict[int, _BKNode] = {}


class _BKTree:
    def __init__(self):
        self.root: _BKNode | None = None

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = _BKNode(value, index)
            return
        node = self.root
        while True:
            distance = _hamming(value, node.value)
            if distance == 0:
                node.indices.append(index)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(value, index)
                return
            node = child

    def query(self, value: int, max_distance: int) -> list[int]:
        if self.root is None:
            return []
        matches: list[int] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = _hamming(value, node.value)
            if distance <= max_distance:
                matches.extend(node.indices)
            lower = distance - max_distance
            upper = distance + max_distance
            stack.extend(
                child
                for edge, child in node.children.items()
                if lower <= edge <= upper
            )
        return matches


def deduplicate_samples(
    samples: Sequence[DatasetSample],
    perceptual_threshold: int = 4,
) -> DeduplicationResult:
    if perceptual_threshold < 0:
        raise ValueError("perceptual_threshold must be non-negative")

    exact_by_hash: dict[str, DatasetSample] = {}
    retained: list[DatasetSample] = []
    removed: list[str] = []

    for sample in sorted(samples, key=lambda item: item.id):
        digest = _sha256(sample.image_path)
        representative = exact_by_hash.get(digest)
        if representative is None:
            exact_by_hash[digest] = sample
            retained.append(sample)
            continue
        if (
            representative.annotations != sample.annotations
            or representative.target_metadata != sample.target_metadata
        ):
            raise ValueError(
                f"exact duplicate images have conflicting annotations: "
                f"{representative.id}, {sample.id}"
            )
        removed.append(sample.id)

    union_find = _UnionFind(len(retained))
    groups_by_id: dict[str, int] = {}
    for index, sample in enumerate(retained):
        key = sample.group_id or sample.id
        previous = groups_by_id.get(key)
        if previous is not None:
            union_find.union(previous, index)
        else:
            groups_by_id[key] = index

    tree = _BKTree()
    for index, sample in enumerate(retained):
        perceptual_hash = _dhash(sample.image_path)
        for match in tree.query(perceptual_hash, perceptual_threshold):
            union_find.union(match, index)
        tree.add(perceptual_hash, index)

    components: dict[int, list[int]] = {}
    for index in range(len(retained)):
        components.setdefault(union_find.find(index), []).append(index)

    updated = list(retained)
    near_groups: list[tuple[str, ...]] = []
    for members in components.values():
        member_samples = [retained[index] for index in members]
        if len(member_samples) > 1:
            near_groups.append(tuple(sorted(sample.id for sample in member_samples)))

        existing_group_ids = {sample.group_id for sample in member_samples if sample.group_id}
        if len(existing_group_ids) == 1:
            merged_group_id = next(iter(existing_group_ids))
        elif len(member_samples) == 1:
            merged_group_id = member_samples[0].group_id or member_samples[0].id
        else:
            identity = "|".join(sorted(existing_group_ids or {sample.id for sample in member_samples}))
            merged_group_id = "dedup:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]

        for index in members:
            updated[index] = replace(retained[index], group_id=merged_group_id)

    return DeduplicationResult(
        samples=tuple(sorted(updated, key=lambda sample: sample.id)),
        exact_duplicates_removed=tuple(sorted(removed)),
        near_duplicate_groups=tuple(sorted(near_groups)),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dhash(path: Path) -> int:
    with Image.open(path) as image:
        grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(grayscale.getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value <<= 1
            if pixels[offset + column] > pixels[offset + column + 1]:
                value |= 1
    return value


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()
