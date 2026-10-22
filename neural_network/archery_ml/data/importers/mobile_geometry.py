"""Import independently annotated mobile originals with session isolation."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from neural_network.archery_ml.contracts import DatasetSample
from neural_network.archery_ml.data.registry import DatasetSource
from neural_network.archery_ml.data.importers.common import portable_path


def import_mobile_geometry(root: Path, source: DatasetSource) -> list[DatasetSample]:
    path = root / str(source.options.get("annotation_file", "geometry.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = []
    partitions = {}
    ids = set()
    for record in payload["samples"]:
        sample = DatasetSample.from_dict(record)
        meta = sample.capture_metadata or {}
        for key in ("capture_session_id", "original_id", "device_model", "original_sha256",
                    "gt_revision", "annotator", "reviewer", "orientation_reference", "suitability"):
            if not meta.get(key):
                raise ValueError(f"missing mobile provenance: {key}")
        if meta["annotator"] == meta["reviewer"]:
            raise ValueError("geometry GT requires independent review")
        if meta["suitability"] not in {"usable", "unusable"}:
            raise ValueError("suitability must be usable or unusable")
        if sample.id in ids or sample.split not in {"mobile_dev", "mobile_real_test"}:
            raise ValueError("duplicate mobile image ID or invalid split")
        ids.add(sample.id)
        for key in ("capture_session_id", "original_id", "original_sha256"):
            identity = (key, meta[key])
            previous = partitions.setdefault(identity, sample.split)
            if previous != sample.split:
                raise ValueError(f"mobile cross-split leakage: {key}")
        image_path = (root / sample.image_path).resolve()
        if not image_path.is_relative_to(root.resolve()):
            raise ValueError("mobile image outside source directory")
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != meta["original_sha256"]:
            raise ValueError("mobile image checksum mismatch")
        with Image.open(image_path) as image:
            if list(image.size) != meta.get("image_size"):
                raise ValueError("mobile image dimensions mismatch")
        if sample.target_metadata is None:
            raise ValueError("mobile target configuration required")
        if meta["orientation_reference"] not in {"physical_marker", "board", "manual_calibration", "unknown"}:
            raise ValueError("unsupported orientation reference")
        count = 1 if sample.target_metadata.format == "SINGLE" else 3
        for annotation in sample.geometry_annotations:
            if annotation.face_index is None or annotation.face_index not in range(count):
                raise ValueError("mobile GT requires explicit valid face_index")
        if meta["suitability"] == "usable":
            if meta["orientation_reference"] == "unknown":
                raise ValueError("usable 2D GT requires independently established orientation")
            for index in range(count):
                annotations = [a for a in sample.geometry_annotations if a.face_index == index]
                centers = [a for a in annotations if a.kind == "target_center"]
                points = [p for a in annotations for p in a.canonical_points]
                image_points = [p for a in annotations if a.canonical_points for p in a.points]
                if len(centers) != 1 or len(points) < 4:
                    raise ValueError("each usable face requires one physical center and four landmarks")
                if any(a.annotation_quality != "reviewed" or a.uncertainty_norm is None
                       for a in annotations if a.canonical_points or a.kind == "target_center"):
                    raise ValueError("geometry landmarks require reviewed quality and uncertainty")
                for values in (points, image_points):
                    array = np.asarray(values)
                    if np.linalg.matrix_rank(array - array.mean(axis=0)) < 2:
                        raise ValueError("geometry landmarks must be non-collinear")
        samples.append(replace(sample, image_path=portable_path(image_path), annotation_path=portable_path(path),
                               source_id=source.id, group_id=f"{source.id}:{meta['capture_session_id']}"))
    return samples
