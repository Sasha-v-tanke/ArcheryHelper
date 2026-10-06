from __future__ import annotations

from pathlib import Path

from neural_network.archery_ml.contracts import DatasetSample, ImpactAnnotation
from neural_network.archery_ml.data.importers.common import (
    class_role,
    group_id,
    image_files,
    image_to_canonical,
    portable_path,
    sample_id,
    split_name,
    target_metadata,
)
from neural_network.archery_ml.data.registry import DatasetSource


def import_yolo(root: Path, source: DatasetSource) -> list[DatasetSample]:
    samples: list[DatasetSample] = []
    metadata = target_metadata(source)

    for image_path in image_files(root):
        label_path = _label_path(root, image_path)
        if label_path is None or not label_path.exists():
            continue
        annotations: list[ImpactAnnotation] = []
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"invalid YOLO annotation {label_path}:{line_number}")
            class_id = parts[0]
            if class_role(source, class_id) != "impact":
                continue
            if len(parts) >= 5:
                x_fraction = float(parts[1])
                y_fraction = float(parts[2])
            else:
                raise ValueError(
                    f"YOLO importer expects class cx cy width height at {label_path}:{line_number}"
                )
            x_norm, y_norm = image_to_canonical(x_fraction, y_fraction)
            annotations.append(ImpactAnnotation(x_norm=x_norm, y_norm=y_norm))

        samples.append(
            DatasetSample(
                id=sample_id(source, root, image_path),
                image_path=portable_path(image_path),
                annotation_path=portable_path(label_path),
                split=split_name(source, image_path),
                source_id=source.id,
                group_id=group_id(source, root, image_path),
                annotations=tuple(annotations),
                target_metadata=metadata,
            )
        )
    return samples


def _label_path(root: Path, image_path: Path) -> Path | None:
    relative = image_path.relative_to(root)
    parts = list(relative.parts)
    if "images" in parts:
        index = parts.index("images")
        parts[index] = "labels"
        return (root.joinpath(*parts)).with_suffix(".txt")
    candidate = image_path.with_suffix(".txt")
    return candidate if candidate.exists() else None
