from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from neural_network.archery_ml.contracts import DatasetSample, ImpactAnnotation
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
    annotation_path = _find_annotation_file(root, source)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {int(item["id"]): str(item.get("name", item["id"])) for item in payload.get("categories", ())}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in payload.get("annotations", ()):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    metadata = target_metadata(source)
    samples: list[DatasetSample] = []
    for image in sorted(payload.get("images", ()), key=lambda item: str(item["file_name"])):
        image_path = _find_image(root, str(image["file_name"]))
        width = float(image["width"])
        height = float(image["height"])
        impacts: list[ImpactAnnotation] = []
        for annotation in annotations_by_image.get(int(image["id"]), ()):
            category_id = int(annotation["category_id"])
            role = class_role(source, str(category_id), categories.get(category_id))
            if role != "impact":
                continue
            x, y = _annotation_point(annotation)
            x_norm, y_norm = image_to_canonical(x / width, y / height)
            impacts.append(ImpactAnnotation(x_norm=x_norm, y_norm=y_norm))

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
            )
        )
    return samples


def _find_annotation_file(root: Path, source: DatasetSource) -> Path:
    configured = source.options.get("annotation_file")
    if configured:
        path = root / str(configured)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    candidates = sorted(root.rglob("*.json"))
    for candidate in candidates:
        name = candidate.name.lower()
        if "coco" in name or name == "_annotations.json":
            return candidate
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"could not identify COCO annotation file under {root}")


def _find_image(root: Path, file_name: str) -> Path:
    direct = root / file_name
    if direct.exists():
        return direct
    matches = sorted(root.rglob(Path(file_name).name))
    if len(matches) != 1:
        raise FileNotFoundError(f"could not uniquely resolve COCO image {file_name}")
    return matches[0]


def _annotation_point(annotation: dict) -> tuple[float, float]:
    keypoints = annotation.get("keypoints")
    if keypoints and len(keypoints) >= 3 and float(keypoints[2]) > 0:
        return float(keypoints[0]), float(keypoints[1])
    bbox = annotation.get("bbox")
    if bbox and len(bbox) >= 4:
        return float(bbox[0]) + float(bbox[2]) / 2.0, float(bbox[1]) + float(bbox[3]) / 2.0
    raise ValueError(f"unsupported COCO annotation id={annotation.get('id')}: expected keypoint or bbox")
