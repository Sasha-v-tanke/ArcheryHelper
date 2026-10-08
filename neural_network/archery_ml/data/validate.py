from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from neural_network.archery_ml.contracts import DatasetSample
from neural_network.archery_ml.data.registry import DatasetRegistry


ALLOWED_SPLITS = {"train", "val", "test", "mobile_real_test"}
MAX_CANONICAL_AXIS = 1.05
SUPPORTED_ANNOTATION_SUFFIXES = {".json", ".txt"}


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    sample_id: str | None
    message: str


@dataclass(frozen=True)
class DatasetValidationReport:
    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def format_errors(self) -> str:
        return "\n".join(
            f"[{issue.code}] {issue.sample_id + ': ' if issue.sample_id else ''}{issue.message}"
            for issue in self.issues
        )


def validate_dataset(
    samples: Sequence[DatasetSample],
    registry: DatasetRegistry | None = None,
) -> DatasetValidationReport:
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()
    seen_images: dict[Path, str] = {}
    registry_ids = {source.id for source in registry.sources} if registry else None

    for sample in samples:
        if sample.id in seen_ids:
            issues.append(ValidationIssue("duplicate_id", sample.id, "sample id is duplicated"))
        seen_ids.add(sample.id)

        if not sample.image_path.exists():
            issues.append(ValidationIssue("missing_image", sample.id, str(sample.image_path)))
        else:
            resolved = sample.image_path.resolve()
            previous = seen_images.get(resolved)
            if previous is not None and previous != sample.id:
                issues.append(
                    ValidationIssue(
                        "duplicate_image_path",
                        sample.id,
                        f"same image path is already used by {previous}",
                    )
                )
            seen_images[resolved] = sample.id

        if sample.annotation_path is not None:
            if not sample.annotation_path.exists():
                issues.append(ValidationIssue("missing_annotation", sample.id, str(sample.annotation_path)))
            elif sample.annotation_path.suffix.lower() not in SUPPORTED_ANNOTATION_SUFFIXES:
                issues.append(
                    ValidationIssue(
                        "unsupported_annotation_file",
                        sample.id,
                        str(sample.annotation_path),
                    )
                )

        if sample.source_id is None:
            issues.append(ValidationIssue("missing_source_id", sample.id, "source_id is required"))
        elif registry_ids is not None and sample.source_id not in registry_ids:
            issues.append(
                ValidationIssue(
                    "unknown_source_id",
                    sample.id,
                    f"{sample.source_id} is not present in registry",
                )
            )

        if not sample.group_id:
            issues.append(ValidationIssue("missing_group_id", sample.id, "group_id is required"))

        if sample.split not in ALLOWED_SPLITS:
            issues.append(ValidationIssue("invalid_split", sample.id, str(sample.split)))

        for index, annotation in enumerate(sample.annotations):
            if abs(annotation.x_norm) > MAX_CANONICAL_AXIS or abs(annotation.y_norm) > MAX_CANONICAL_AXIS:
                issues.append(
                    ValidationIssue(
                        "invalid_coordinate",
                        sample.id,
                        f"annotation {index} is outside canonical image bounds: "
                        f"({annotation.x_norm}, {annotation.y_norm})",
                    )
                )
            metadata = sample.target_metadata
            if metadata is not None:
                if metadata.format == "SINGLE" and annotation.face_index != 0:
                    issues.append(
                        ValidationIssue(
                            "invalid_face_index",
                            sample.id,
                            f"single target annotation {index} has face_index={annotation.face_index}",
                        )
                    )
                if metadata.format == "TRIPLE" and annotation.face_index not in {0, 1, 2}:
                    issues.append(
                        ValidationIssue(
                            "invalid_face_index",
                            sample.id,
                            f"triple target annotation {index} has face_index={annotation.face_index}",
                        )
                    )

        for index, annotation in enumerate(sample.raw_annotations):
            if not 0.0 <= annotation.x_fraction <= 1.0 or not 0.0 <= annotation.y_fraction <= 1.0:
                issues.append(
                    ValidationIssue(
                        "invalid_raw_coordinate",
                        sample.id,
                        f"raw annotation {index} is outside image bounds: "
                        f"({annotation.x_fraction}, {annotation.y_fraction})",
                    )
                )

        for index, annotation in enumerate(sample.geometry_annotations):
            for point_index, (x_fraction, y_fraction) in enumerate(annotation.points):
                if not 0.0 <= x_fraction <= 1.0 or not 0.0 <= y_fraction <= 1.0:
                    issues.append(
                        ValidationIssue(
                            "invalid_geometry_coordinate",
                            sample.id,
                            f"geometry annotation {index} point {point_index} is outside image bounds: "
                            f"({x_fraction}, {y_fraction})",
                        )
                    )

    return DatasetValidationReport(tuple(issues))
