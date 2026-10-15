from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from neural_network.archery_ml.contracts import DatasetSample, TargetFormatName, load_manifest


@dataclass(frozen=True)
class DatasetView:
    samples: tuple[DatasetSample, ...]

    @staticmethod
    def load(path: str | Path) -> "DatasetView":
        return DatasetView(tuple(load_manifest(Path(path))))

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[DatasetSample]:
        return iter(self.samples)

    def select(
        self,
        *,
        splits: Iterable[str] | None = None,
        source_ids: Iterable[str] | None = None,
        target_formats: Iterable[TargetFormatName] | None = None,
        require_canonical_impacts: bool = False,
        require_raw_impacts: bool = False,
        require_geometry: bool = False,
    ) -> "DatasetView":
        split_filter = None if splits is None else set(splits)
        source_filter = None if source_ids is None else set(source_ids)
        target_filter = None if target_formats is None else set(target_formats)

        selected = []
        for sample in self.samples:
            if split_filter is not None and sample.split not in split_filter:
                continue
            if source_filter is not None and sample.source_id not in source_filter:
                continue
            if target_filter is not None:
                if sample.target_metadata is None or sample.target_metadata.format not in target_filter:
                    continue
            if require_canonical_impacts and not sample.annotations:
                continue
            if require_raw_impacts and not sample.raw_annotations:
                continue
            if require_geometry and not sample.geometry_annotations:
                continue
            selected.append(sample)
        return DatasetView(tuple(selected))

    def source_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted({sample.source_id for sample in self.samples if sample.source_id is not None})
        )

    def image_paths(self) -> tuple[Path, ...]:
        return tuple(sample.image_path for sample in self.samples)

    def sample_ids(self) -> tuple[str, ...]:
        return tuple(sample.id for sample in self.samples)

    @staticmethod
    def from_samples(samples: Sequence[DatasetSample]) -> "DatasetView":
        return DatasetView(tuple(samples))
