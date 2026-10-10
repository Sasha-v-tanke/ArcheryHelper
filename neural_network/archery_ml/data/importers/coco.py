from __future__ import annotations

import json
from collections import defaultdict
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
    image_to_canonical,
    portable_path,
    sample_id,
    split_name,
    target_metadata,
)
from neural_network.archery_ml.data.registry import DatasetSource


def import_coco(root: Path, source: DatasetSource) -> list[DatasetSample]:
    samples: list[DatasetSample] = []
    for annotation_path in _find_annotation_files(root, source):
        samples.extend(_import_coco_file(root, annotation_path, source))
    return samples


def _import_coco_file(
    root: Path,
    annotation_path: Path,
    source: DatasetSource,
) -> list[DatasetSample]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {
        int(item["id"]): str(item.get("name", item["id"]))
        for item in payload.get("categories", ())
    }
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in payload.get("annotations", ()):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    metadata = target_metadata(source)
    samples: list[DatasetSample] = []
    for image in sorted(payload.get("images", ()), key=lambda item: str(item["file_name"])):
        image_path = _find_image(root, annotation_path.parent, str(image["file_name"]))
        width = float(image["width"])
        height = float(image["height"])
        impacts: list[ImpactAnnotation] = []
        raw_impacts: list[ImagePointAnnotation] = []
        geometry: list[ImageGeometryAnnotation] = []
        for annotation in annotations_by_image.get(int(image["id"]), ()):
            category_id = int(annotation["category_id"])
            category_name = categories.get(category_id, str(category_id))
            role = class_role(source, str(category_id), category_name)
            if role == "ignore":
                continue
            if role == "impact":
                x, y = _annotation_point(annotation, source)
                x_fraction = x / width
                y_fraction = y / height
                if source.annotation_space == "image":
                    raw_impacts.append(
                        ImagePointAnnotation(
                            x_fraction=x_fraction,
                            y_fraction=y_fraction,
                            source_label=category_name,
                            bbox=_normalized_bbox(annotation.get("bbox"), width, height),
                        )
                    )
                else:
                    x_norm, y_norm = image_to_canonical(x_fraction, y_fraction)
                    impacts.append(ImpactAnnotation(x_norm=x_norm, y_norm=y_norm))
                continue
            geometry.extend(
                _geometry_annotations(
                    role,
                    category_name,
                    annotation,
                    width,
                    height,
                )
            )

        samples.append(
            DatasetSample(
                id=sample_id(source, root, image_path),
                image_path=portable_path(image_path),
                annotation_path=portable_path(annotation_path),
                split=split_name(source, image_path),
                source_id=source.id,
                group_id=group_id(source, root, image_path),
                annotations=tuple(impacts),
                target_metadata=metadata,
                raw_annotations=tuple(raw_impacts),
                geometry_annotations=tuple(geometry),
            )
        )
    return samples


def _find_annotation_files(root: Path, source: DatasetSource) -> tuple[Path, ...]:
    configured = source.options.get("annotation_file")
    if configured:
        path = root / str(configured)
        if not path.exists():
            raise FileNotFoundError(path)
        return (path,)

    candidates = sorted(root.rglob("*.json"))
    annotation_files = tuple(
        candidate
        for candidate in candidates
        if "coco" in candidate.name.lower() or candidate.name == "_annotations.json"
    )
    if annotation_files:
        return annotation_files
    if len(candidates) == 1:
        return (candidates[0],)
    raise ValueError(f"could not identify COCO annotation file under {root}")


def _find_image(root: Path, annotation_dir: Path, file_name: str) -> Path:
    for base in (annotation_dir, root):
        direct = base / file_name
        if direct.exists():
            return direct
    matches = sorted(root.rglob(Path(file_name).name))
    if len(matches) != 1:
        raise FileNotFoundError(f"could not uniquely resolve COCO image {file_name}")
    return matches[0]


def _annotation_point(annotation: dict, source: DatasetSource | None = None) -> tuple[float, float]:
    mode = "auto" if source is None else str(source.options.get("impact_point", "auto"))
    if mode == "keypoint":
        return _keypoint(annotation, source)
    if mode != "auto":
        raise ValueError(f"unsupported impact_point mode for {source.id if source else 'coco'}: {mode}")
    point = _visible_keypoint(annotation, 0)
    if point is not None:
        return point
    bbox = annotation.get("bbox")
    if bbox and len(bbox) >= 4:
        return float(bbox[0]) + float(bbox[2]) / 2.0, float(bbox[1]) + float(bbox[3]) / 2.0
    raise ValueError(f"unsupported COCO annotation id={annotation.get('id')}: expected keypoint or bbox")


def _keypoint(annotation: dict, source: DatasetSource) -> tuple[float, float]:
    keypoint_index = int(source.options.get("keypoint_index", 0))
    point = _visible_keypoint(annotation, keypoint_index)
    if point is None:
        raise ValueError(
            f"missing visible keypoint {keypoint_index} for COCO annotation id={annotation.get('id')}"
        )
    return point


def _visible_keypoint(annotation: dict, keypoint_index: int) -> tuple[float, float] | None:
    keypoints = annotation.get("keypoints")
    if not keypoints:
        return None
    start = keypoint_index * 3
    if len(keypoints) < start + 3:
        return None
    if float(keypoints[start + 2]) <= 0:
        return None
    return float(keypoints[start]), float(keypoints[start + 1])


def _normalized_bbox(
    bbox: list | tuple | None,
    width: float,
    height: float,
) -> tuple[float, float, float, float] | None:
    if not bbox or len(bbox) < 4:
        return None
    return (
        float(bbox[0]) / width,
        float(bbox[1]) / height,
        float(bbox[2]) / width,
        float(bbox[3]) / height,
    )


def _geometry_annotations(
    role: str,
    source_label: str,
    annotation: dict,
    width: float,
    height: float,
) -> list[ImageGeometryAnnotation]:
    if role == "target_center":
        x, y = _annotation_point(annotation)
        return [
            ImageGeometryAnnotation(
                kind="target_center",
                points=((x / width, y / height),),
                source_label=source_label,
            )
        ]

    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, list):
        polygons = []
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2 != 0:
                continue
            points = tuple(
                (float(polygon[index]) / width, float(polygon[index + 1]) / height)
                for index in range(0, len(polygon), 2)
            )
            polygons.append(
                ImageGeometryAnnotation(
                    kind=role,
                    points=points,
                    source_label=source_label,
                )
            )
        if polygons:
            return polygons

    bbox = _normalized_bbox(annotation.get("bbox"), width, height)
    if bbox is None:
        raise ValueError(
            f"geometry annotation id={annotation.get('id')} requires polygon or bbox"
        )
    x, y, box_width, box_height = bbox
    return [
        ImageGeometryAnnotation(
            kind=role,
            points=(
                (x, y),
                (x + box_width, y),
                (x + box_width, y + box_height),
                (x, y + box_height),
            ),
            source_label=source_label,
        )
    ]
