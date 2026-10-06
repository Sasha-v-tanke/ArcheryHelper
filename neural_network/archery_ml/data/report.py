from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from neural_network.archery_ml.contracts import DatasetSample
from neural_network.archery_ml.data.deduplicate import deduplicate_samples


@dataclass(frozen=True)
class DatasetReport:
    images: int
    impacts: int
    source_distribution: dict[str, int]
    target_distribution: dict[str, int]
    split_sizes: dict[str, int]
    exact_duplicate_count: int
    near_duplicate_group_count: int
    duplicate_rate: float

    def to_dict(self) -> dict:
        return {
            "images": self.images,
            "impacts": self.impacts,
            "source_distribution": self.source_distribution,
            "target_distribution": self.target_distribution,
            "split_sizes": self.split_sizes,
            "exact_duplicate_count": self.exact_duplicate_count,
            "near_duplicate_group_count": self.near_duplicate_group_count,
            "duplicate_rate": self.duplicate_rate,
        }


def build_report(
    samples: Sequence[DatasetSample],
    perceptual_threshold: int = 4,
) -> DatasetReport:
    sources = Counter(sample.source_id or "UNKNOWN" for sample in samples)
    splits = Counter(sample.split for sample in samples)
    targets = Counter(_target_key(sample) for sample in samples)
    deduplication = deduplicate_samples(samples, perceptual_threshold=perceptual_threshold)

    return DatasetReport(
        images=len(samples),
        impacts=sum(len(sample.annotations) for sample in samples),
        source_distribution=dict(sorted(sources.items())),
        target_distribution=dict(sorted(targets.items())),
        split_sizes=dict(sorted(splits.items())),
        exact_duplicate_count=len(deduplication.exact_duplicates_removed),
        near_duplicate_group_count=len(deduplication.near_duplicate_groups),
        duplicate_rate=deduplication.duplicate_rate,
    )


def _target_key(sample: DatasetSample) -> str:
    metadata = sample.target_metadata
    if metadata is None:
        return "UNKNOWN"
    layout = "" if metadata.triple_layout is None else f":{metadata.triple_layout}"
    return (
        f"{metadata.format}{layout}:"
        f"{metadata.ten_ring_mode}:min-{metadata.minimum_scoring_zone}"
    )
