from __future__ import annotations

from pathlib import Path

from neural_network.archery_ml.contracts import (
    DatasetSample,
    ImageGeometryAnnotation,
    ImagePointAnnotation,
    ImpactAnnotation,
)
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
        raw_annotations: list[ImagePointAnnotation] = []
        geometry_annotations: list[ImageGeometryAnnotation] = []
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(
                    f"YOLO importer expects class cx cy width height at {label_path}:{line_number}"
                )
            class_id = parts[0]
            role = class_role(source, class_id)
            if role == "ignore":
                continue
            cx = float(parts[1])
            cy = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            if role == "impact":
                x_fraction, y_fraction = _impact_point(source, parts, cx, cy, label_path, line_number)
                if source.annotation_space == "image":
                    raw_annotations.append(
                        ImagePointAnnotation(
                            x_fraction=x_fraction,
                            y_fraction=y_fraction,
                            source_label=class_id,
                            bbox=(
                                cx - width / 2.0,
                                cy - height / 2.0,
                                width,
                                height,
                            ),
                        )
                    )
                else:
                    x_norm, y_norm = image_to_canonical(x_fraction, y_fraction)
                    annotations.append(ImpactAnnotation(x_norm=x_norm, y_norm=y_norm))
                continue

            if role == "target_center":
                geometry_annotations.append(
                    ImageGeometryAnnotation(
                        kind="target_center",
                        points=((cx, cy),),
                        source_label=class_id,
                    )
                )
            else:
                x = cx - width / 2.0
                y = cy - height / 2.0
                geometry_annotations.append(
                    ImageGeometryAnnotation(
                        kind=role,
                        points=(
                            (x, y),
                            (x + width, y),
                            (x + width, y + height),
                            (x, y + height),
                        ),
                        source_label=class_id,
                    )
                )

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
                raw_annotations=tuple(raw_annotations),
                geometry_annotations=tuple(geometry_annotations),
            )
        )
    return samples


def _impact_point(
    source: DatasetSource,
    parts: list[str],
    cx: float,
    cy: float,
    label_path: Path,
    line_number: int,
) -> tuple[float, float]:
    mode = str(source.options.get("impact_point", "bbox_center"))
    if mode == "bbox_center":
        return cx, cy
    if mode != "keypoint":
        raise ValueError(f"unsupported impact_point mode for {source.id}: {mode}")

    keypoint_index = int(source.options.get("keypoint_index", 0))
    dimensions = int(source.options.get("keypoint_dimensions", 3))
    if dimensions not in {2, 3}:
        raise ValueError("keypoint_dimensions must be 2 or 3")
    start = 5 + keypoint_index * dimensions
    if len(parts) < start + 2:
        raise ValueError(
            f"missing keypoint {keypoint_index} at {label_path}:{line_number}"
        )
    if dimensions == 3 and len(parts) > start + 2 and float(parts[start + 2]) <= 0.0:
        raise ValueError(
            f"keypoint {keypoint_index} is not visible at {label_path}:{line_number}"
        )
    return float(parts[start]), float(parts[start + 1])


def _label_path(root: Path, image_path: Path) -> Path | None:
    relative = image_path.relative_to(root)
    parts = list(relative.parts)
    if "images" in parts:
        index = parts.index("images")
        parts[index] = "labels"
        return (root.joinpath(*parts)).with_suffix(".txt")
    candidate = image_path.with_suffix(".txt")
    return candidate if candidate.exists() else None
