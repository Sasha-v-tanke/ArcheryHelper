from __future__ import annotations

from pathlib import Path
from typing import Callable

from neural_network.archery_ml.contracts import DatasetSample
from neural_network.archery_ml.data.importers.coco import import_coco
from neural_network.archery_ml.data.importers.legacy import import_legacy
from neural_network.archery_ml.data.importers.roboflow import import_roboflow
from neural_network.archery_ml.data.importers.yolo import import_yolo
from neural_network.archery_ml.data.registry import DatasetSource


Importer = Callable[[Path, DatasetSource], list[DatasetSample]]
IMPORTERS: dict[str, Importer] = {
    "legacy": import_legacy,
    "yolo": import_yolo,
    "coco": import_coco,
    "roboflow": import_roboflow,
}


def import_dataset(root: str | Path, source: DatasetSource) -> list[DatasetSample]:
    return IMPORTERS[source.importer](Path(root), source)
