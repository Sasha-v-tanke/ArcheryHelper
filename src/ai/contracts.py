from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence


SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class DatasetSample:
    id: str
    image_path: Path
    annotation_path: Path
    split: SplitName = "train"
    source: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["image_path"] = str(self.image_path)
        data["annotation_path"] = str(self.annotation_path)
        return data

    @staticmethod
    def from_dict(data: dict) -> "DatasetSample":
        return DatasetSample(
            id=data["id"],
            image_path=Path(data["image_path"]),
            annotation_path=Path(data["annotation_path"]),
            split=data.get("split", "train"),
            source=data.get("source"),
        )


@dataclass(frozen=True)
class ShotAnnotation:
    x_norm: float
    y_norm: float

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "ShotAnnotation":
        return ShotAnnotation(x_norm=float(data["x_norm"]), y_norm=float(data["y_norm"]))


@dataclass(frozen=True)
class ModelInputSpec:
    width: int
    height: int
    color_space: Literal["RGB"] = "RGB"
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "ModelInputSpec":
        return ModelInputSpec(
            width=int(data["width"]),
            height=int(data["height"]),
            color_space=data.get("color_space", "RGB"),
            mean=tuple(float(v) for v in data.get("mean", (0.0, 0.0, 0.0))),
            std=tuple(float(v) for v in data.get("std", (1.0, 1.0, 1.0))),
        )


@dataclass(frozen=True)
class ModelMetadata:
    contract_version: int
    width: int
    height: int
    output: str
    max_shots: int
    coordinate_system: str
    confidence_threshold: float

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "ModelMetadata":
        return ModelMetadata(
            contract_version=int(data["contract_version"]),
            width=int(data["width"]),
            height=int(data["height"]),
            output=data["output"],
            max_shots=int(data["max_shots"]),
            coordinate_system=data["coordinate_system"],
            confidence_threshold=float(data["confidence_threshold"]),
        )


@dataclass(frozen=True)
class ValidationReport:
    missing_images: tuple[str, ...] = ()
    missing_annotations: tuple[str, ...] = ()
    duplicate_ids: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.missing_images and not self.missing_annotations and not self.duplicate_ids

    def format_errors(self) -> str:
        parts = []
        if self.missing_images:
            parts.append(f"missing images: {', '.join(self.missing_images)}")
        if self.missing_annotations:
            parts.append(f"missing annotations: {', '.join(self.missing_annotations)}")
        if self.duplicate_ids:
            parts.append(f"duplicate ids: {', '.join(self.duplicate_ids)}")
        return "; ".join(parts)


def load_manifest(path: Path) -> list[DatasetSample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [DatasetSample.from_dict(item) for item in data["samples"]]


def save_manifest(path: Path, samples: Sequence[DatasetSample]) -> None:
    path.write_text(
        json.dumps({"samples": [sample.to_dict() for sample in samples]}, indent=2),
        encoding="utf-8",
    )


def validate_manifest(samples: Sequence[DatasetSample]) -> ValidationReport:
    seen = set()
    duplicates = set()
    missing_images = []
    missing_annotations = []

    for sample in samples:
        if sample.id in seen:
            duplicates.add(sample.id)
        seen.add(sample.id)
        if not sample.image_path.exists():
            missing_images.append(sample.id)
        if not sample.annotation_path.exists():
            missing_annotations.append(sample.id)

    return ValidationReport(
        missing_images=tuple(sorted(missing_images)),
        missing_annotations=tuple(sorted(missing_annotations)),
        duplicate_ids=tuple(sorted(duplicates)),
    )


def build_manifest_from_dirs(data_dir: Path, json_dir: Path, split: SplitName = "train") -> list[DatasetSample]:
    image_paths = []
    for pattern in ("*.jpeg", "*.jpg", "*.png"):
        image_paths.extend(data_dir.glob(pattern))

    images_by_id = _group_by_stem(image_paths)
    annotations_by_id = _group_by_stem(json_dir.glob("*.json"))
    duplicate_ids = sorted(
        sample_id
        for sample_id, paths in {**images_by_id, **annotations_by_id}.items()
        if len(paths) > 1
    )
    missing_annotations = sorted(set(images_by_id) - set(annotations_by_id))
    missing_images = sorted(set(annotations_by_id) - set(images_by_id))

    report = ValidationReport(
        missing_images=tuple(missing_images),
        missing_annotations=tuple(missing_annotations),
        duplicate_ids=tuple(duplicate_ids),
    )
    if not report.is_valid:
        raise ValueError(report.format_errors())

    return [
        DatasetSample(sample_id, images_by_id[sample_id][0], annotations_by_id[sample_id][0], split)
        for sample_id in sorted(images_by_id)
    ]


def _group_by_stem(paths) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(path.stem, []).append(path)
    return grouped
