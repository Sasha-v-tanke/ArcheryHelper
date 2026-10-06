from __future__ import annotations

from pathlib import Path

from neural_network.archery_ml.contracts import DatasetSample
from neural_network.archery_ml.data.importers.coco import import_coco
from neural_network.archery_ml.data.importers.yolo import import_yolo
from neural_network.archery_ml.data.registry import DatasetSource


def import_roboflow(root: Path, source: DatasetSource) -> list[DatasetSample]:
    export_format = str(source.options.get("export_format", "")).lower()
    if export_format in {"coco", "coco-json", "coco_json"}:
        return import_coco(root, source)
    if export_format in {"yolo", "yolov8", "yolov11"}:
        return import_yolo(root, source)

    if any(path.name.lower().endswith(".json") and "coco" in path.name.lower() for path in root.rglob("*.json")):
        return import_coco(root, source)
    if any(path.name == "data.yaml" for path in root.rglob("data.yaml")) or any(root.rglob("labels/*.txt")):
        return import_yolo(root, source)
    raise ValueError(
        f"unsupported Roboflow export under {root}; set options.export_format to coco or yolo"
    )
