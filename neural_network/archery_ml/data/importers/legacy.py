from __future__ import annotations

import json
import math
from pathlib import Path

from neural_network.archery_ml.contracts import DatasetSample, ImpactAnnotation
from neural_network.archery_ml.data.importers.common import (
    group_id,
    image_files,
    portable_path,
    sample_id,
    split_name,
    target_metadata,
)
from neural_network.archery_ml.data.registry import DatasetSource


def import_legacy(root: Path, source: DatasetSource) -> list[DatasetSample]:
    if source.annotation_space != "canonical":
        raise ValueError("legacy importer only supports canonical annotations")

    samples: list[DatasetSample] = []
    metadata = target_metadata(source)

    for image_path in image_files(root):
        annotation_path = image_path.with_suffix(".json")
        if not annotation_path.exists():
            continue
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        annotations = tuple(_impact_from_legacy(item) for item in payload.get("shots", ()))
        samples.append(
            DatasetSample(
                id=sample_id(source, root, image_path),
                image_path=portable_path(image_path),
                annotation_path=portable_path(annotation_path),
                split=split_name(source, image_path),
                source_id=source.id,
                group_id=group_id(source, root, image_path),
                annotations=annotations,
                target_metadata=metadata,
            )
        )
    return samples


def _impact_from_legacy(data: dict) -> ImpactAnnotation:
    if "x_norm" in data and "y_norm" in data:
        x_norm = float(data["x_norm"])
        y_norm = float(data["y_norm"])
    elif "r_norm" in data and "theta_deg" in data:
        radius = float(data["r_norm"])
        angle = math.radians(float(data["theta_deg"]))
        x_norm = radius * math.cos(angle)
        y_norm = radius * math.sin(angle)
    else:
        raise ValueError("legacy annotation must contain x_norm/y_norm or r_norm/theta_deg")
    return ImpactAnnotation(
        x_norm=x_norm,
        y_norm=y_norm,
        face_index=int(data.get("face_index", 0)),
        confidence=None if data.get("confidence") is None else float(data["confidence"]),
    )
