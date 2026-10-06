from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence


SCHEMA_VERSION = 2
SplitName = Literal["train", "val", "test"]
TargetFormatName = Literal["SINGLE", "TRIPLE"]
TripleLayoutName = Literal["VERTICAL", "TRIANGULAR"]
TenRingModeName = Literal["RECURVE", "COMPOUND"]


@dataclass(frozen=True)
class ImpactAnnotation:
    x_norm: float
    y_norm: float
    face_index: int = 0
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.x_norm) or not math.isfinite(self.y_norm):
            raise ValueError("impact coordinates must be finite")
        if self.face_index < 0:
            raise ValueError("face_index must be non-negative")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "ImpactAnnotation":
        confidence = data.get("confidence")
        return ImpactAnnotation(
            x_norm=float(data["x_norm"]),
            y_norm=float(data["y_norm"]),
            face_index=int(data.get("face_index", 0)),
            confidence=None if confidence is None else float(confidence),
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
class TargetMetadata:
    format: TargetFormatName
    minimum_scoring_zone: int
    ten_ring_mode: TenRingModeName
    triple_layout: TripleLayoutName | None = None
    face_diameter_mm: int | None = None

    def __post_init__(self) -> None:
        if self.minimum_scoring_zone not in (1, 5, 6):
            raise ValueError("minimum_scoring_zone must be 1, 5, or 6")
        if self.format == "SINGLE" and self.triple_layout is not None:
            raise ValueError("single target must not define triple_layout")
        if self.format == "TRIPLE" and self.triple_layout is None:
            raise ValueError("triple target requires triple_layout")
        if self.face_diameter_mm is not None and self.face_diameter_mm <= 0:
            raise ValueError("face_diameter_mm must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "TargetMetadata":
        return TargetMetadata(
            format=data["format"],
            minimum_scoring_zone=int(data["minimum_scoring_zone"]),
            ten_ring_mode=data["ten_ring_mode"],
            triple_layout=data.get("triple_layout"),
            face_diameter_mm=None if data.get("face_diameter_mm") is None else int(data["face_diameter_mm"]),
        )


@dataclass(frozen=True)
class DatasetSample:
    id: str
    image_path: Path
    annotation_path: Path | None
    split: SplitName = "train"
    source_id: str | None = None
    group_id: str | None = None
    annotations: tuple[ImpactAnnotation, ...] = ()
    target_metadata: TargetMetadata | None = None

    @property
    def source(self) -> str | None:
        return self.source_id

    def to_dict(self) -> dict:
        data = {
            "id": self.id,
            "image_path": str(self.image_path),
            "split": self.split,
            "source_id": self.source_id,
            "group_id": self.group_id,
            "annotations": [annotation.to_dict() for annotation in self.annotations],
            "target_metadata": None if self.target_metadata is None else self.target_metadata.to_dict(),
        }
        if self.annotation_path is not None:
            data["annotation_path"] = str(self.annotation_path)
        return data

    @staticmethod
    def from_dict(data: dict) -> "DatasetSample":
        annotation_path = data.get("annotation_path")
        target_metadata = data.get("target_metadata")
        return DatasetSample(
            id=data["id"],
            image_path=Path(data["image_path"]),
            annotation_path=None if annotation_path is None else Path(annotation_path),
            split=data.get("split", "train"),
            source_id=data.get("source_id", data.get("source")),
            group_id=data.get("group_id"),
            annotations=tuple(ImpactAnnotation.from_dict(item) for item in data.get("annotations", ())),
            target_metadata=None if target_metadata is None else TargetMetadata.from_dict(target_metadata),
        )


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
class ModelOutputSpec:
    type: str
    stride: int

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "ModelOutputSpec":
        return ModelOutputSpec(type=data["type"], stride=int(data["stride"]))


@dataclass(frozen=True)
class ModelPostprocessSpec:
    confidence_threshold: float
    nms_radius: int

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "ModelPostprocessSpec":
        return ModelPostprocessSpec(
            confidence_threshold=float(data["confidence_threshold"]),
            nms_radius=int(data["nms_radius"]),
        )


@dataclass(frozen=True)
class ModelMetadata:
    contract_version: int
    model_version: str
    input: ModelInputSpec
    output: ModelOutputSpec
    postprocess: ModelPostprocessSpec

    def __post_init__(self) -> None:
        if self.contract_version != 2:
            raise ValueError("ModelMetadata requires contract_version 2")

    def to_dict(self) -> dict:
        return {
            "contract_version": self.contract_version,
            "model_version": self.model_version,
            "input": self.input.to_dict(),
            "output": self.output.to_dict(),
            "postprocess": self.postprocess.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict) -> "ModelMetadata":
        return ModelMetadata(
            contract_version=int(data["contract_version"]),
            model_version=data["model_version"],
            input=ModelInputSpec.from_dict(data["input"]),
            output=ModelOutputSpec.from_dict(data["output"]),
            postprocess=ModelPostprocessSpec.from_dict(data["postprocess"]),
        )


@dataclass(frozen=True)
class LegacyModelMetadataV1:
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
    def from_dict(data: dict) -> "LegacyModelMetadataV1":
        return LegacyModelMetadataV1(
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
    schema_version = int(data.get("schema_version", 1))
    if schema_version not in (1, SCHEMA_VERSION):
        raise ValueError(f"unsupported dataset schema version: {schema_version}")
    return [DatasetSample.from_dict(item) for item in data["samples"]]


def save_manifest(path: Path, samples: Sequence[DatasetSample]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "samples": [sample.to_dict() for sample in samples],
            },
            indent=2,
        ),
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
        if sample.annotation_path is not None and not sample.annotation_path.exists():
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
        DatasetSample(
            id=sample_id,
            image_path=images_by_id[sample_id][0],
            annotation_path=annotations_by_id[sample_id][0],
            split=split,
            group_id=sample_id,
        )
        for sample_id in sorted(images_by_id)
    ]


def _group_by_stem(paths) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(path.stem, []).append(path)
    return grouped
