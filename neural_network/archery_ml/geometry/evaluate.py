from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from neural_network.archery_ml.contracts import ImageGeometryAnnotation
from neural_network.archery_ml.data.view import DatasetView
from neural_network.archery_ml.geometry.detector import FaceGeometry, TargetGeometryDetector
from neural_network.archery_ml.geometry.quality import ImageQualityGate
from neural_network.archery_ml.geometry.target_templates import (
    CANONICAL_CENTER_PX,
    CANONICAL_RADIUS_PX,
)
from neural_network.archery_ml.targets import TargetConfig, TargetFormat, TargetTemplate


_RING_SCORE_RE = re.compile(r"ring[-_ ]?(10|[1-9])", re.IGNORECASE)


@dataclass(frozen=True)
class GeometryErrorMetrics:
    mean: float | None
    median: float | None
    p95: float | None
    maximum: float | None
    count: int

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "median": self.median,
            "p95": self.p95,
            "max": self.maximum,
            "count": self.count,
        }


@dataclass(frozen=True)
class GeometryEvaluationReport:
    samples: int
    evaluated_samples: int
    ready_samples: int
    status_counts: dict[str, int]
    center_error: GeometryErrorMetrics
    radial_error: GeometryErrorMetrics
    target_config: dict[str, object]

    def to_dict(self) -> dict:
        return {
            "samples": self.samples,
            "evaluated_samples": self.evaluated_samples,
            "ready_samples": self.ready_samples,
            "status_counts": self.status_counts,
            "center_error": self.center_error.to_dict(),
            "radial_error": self.radial_error.to_dict(),
            "target_config": self.target_config,
        }


class GeometryEvaluator:
    def __init__(
        self,
        detector: TargetGeometryDetector | None = None,
        quality_gate: ImageQualityGate | None = None,
    ):
        self.detector = detector or TargetGeometryDetector()
        self.quality_gate = quality_gate or ImageQualityGate()

    def evaluate(
        self,
        view: DatasetView,
        target_config: TargetConfig,
    ) -> GeometryEvaluationReport:
        if target_config.format is not TargetFormat.SINGLE:
            raise ValueError(
                "real geometry evaluation currently requires SINGLE because "
                "ImageGeometryAnnotation has no face_index"
            )
        samples = tuple(sample for sample in view if sample.geometry_annotations)
        if not samples:
            raise ValueError("no geometry-annotated samples selected")

        status_counts: Counter[str] = Counter()
        center_errors: list[float] = []
        radial_errors: list[float] = []
        evaluated_samples = 0
        ready_samples = 0
        template = TargetTemplate.from_config(target_config)

        for sample in samples:
            image = _load_rgb(sample.image_path)
            geometry = self.detector.detect(image, target_config)
            quality = self.quality_gate.evaluate(image, geometry)
            status_counts[quality.status.value] += 1
            if quality.status.value == "READY":
                ready_samples += 1
            if len(geometry.faces) != 1:
                continue

            face = geometry.faces[0]
            sample_has_metric = False
            ground_truth_center = _ground_truth_center(
                sample.geometry_annotations,
                template,
            )
            if ground_truth_center is not None:
                center_norm = _to_canonical_norm(
                    face,
                    ground_truth_center,
                    geometry.image_width,
                    geometry.image_height,
                )
                center_errors.append(float(np.linalg.norm(center_norm)))
                sample_has_metric = True

            for annotation in sample.geometry_annotations:
                expected_radius = _expected_radius(annotation, template)
                if expected_radius is None:
                    continue
                for point in annotation.points:
                    normalized = _to_canonical_norm(
                        face,
                        point,
                        geometry.image_width,
                        geometry.image_height,
                    )
                    radial_errors.append(abs(float(np.linalg.norm(normalized)) - expected_radius))
                    sample_has_metric = True

            if sample_has_metric:
                evaluated_samples += 1

        return GeometryEvaluationReport(
            samples=len(samples),
            evaluated_samples=evaluated_samples,
            ready_samples=ready_samples,
            status_counts=dict(sorted(status_counts.items())),
            center_error=_metrics(center_errors),
            radial_error=_metrics(radial_errors),
            target_config={
                "format": target_config.format.value,
                "minimum_scoring_zone": target_config.minimum_scoring_zone,
                "ten_ring_mode": target_config.ten_ring_mode.value,
                "face_diameter_mm": target_config.face_diameter_mm,
                "triple_layout": (
                    None
                    if target_config.triple_layout is None
                    else target_config.triple_layout.value
                ),
            },
        )


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _ground_truth_center(
    annotations: tuple[ImageGeometryAnnotation, ...],
    template: TargetTemplate,
) -> tuple[float, float] | None:
    center_points = [
        annotation.points[0]
        for annotation in annotations
        if annotation.kind == "target_center" and annotation.points
    ]
    if center_points:
        values = np.asarray(center_points, dtype=np.float64)
        return float(values[:, 0].mean()), float(values[:, 1].mean())

    candidates = []
    for annotation in annotations:
        expected_radius = _expected_radius(annotation, template)
        if expected_radius is None or not annotation.points:
            continue
        candidates.append((expected_radius, len(annotation.points), annotation))
    if not candidates:
        return None

    _, _, annotation = max(candidates, key=lambda item: (item[0], item[1]))
    points = np.asarray(annotation.points, dtype=np.float32)
    if len(points) >= 5:
        ellipse = cv2.fitEllipse(points.reshape(-1, 1, 2))
        return float(ellipse[0][0]), float(ellipse[0][1])
    center = points.mean(axis=0)
    return float(center[0]), float(center[1])


def _expected_radius(
    annotation: ImageGeometryAnnotation,
    template: TargetTemplate,
) -> float | None:
    if annotation.kind == "target_face":
        return float(template.visible_radius_norm)
    if annotation.kind != "ring" or annotation.source_label is None:
        return None
    match = _RING_SCORE_RE.search(annotation.source_label)
    if match is None:
        return None
    score = int(match.group(1))
    for boundary in template.ring_boundaries:
        if boundary.score == score:
            return float(boundary.radius_norm)
    return None


def _to_canonical_norm(
    face: FaceGeometry,
    point: tuple[float, float],
    image_width: int,
    image_height: int,
) -> np.ndarray:
    x_fraction, y_fraction = point
    source = np.array(
        [[[x_fraction * image_width, y_fraction * image_height]]],
        dtype=np.float32,
    )
    transform = np.asarray(face.homography, dtype=np.float64)
    canonical = cv2.perspectiveTransform(source, transform).reshape(2)
    return np.asarray(
        [
            (canonical[0] - CANONICAL_CENTER_PX) / CANONICAL_RADIUS_PX,
            (canonical[1] - CANONICAL_CENTER_PX) / CANONICAL_RADIUS_PX,
        ],
        dtype=np.float64,
    )


def _metrics(values: list[float]) -> GeometryErrorMetrics:
    if not values:
        return GeometryErrorMetrics(
            mean=None,
            median=None,
            p95=None,
            maximum=None,
            count=0,
        )
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return GeometryErrorMetrics(
            mean=None,
            median=None,
            p95=None,
            maximum=None,
            count=0,
        )
    return GeometryErrorMetrics(
        mean=float(finite.mean()),
        median=float(np.median(finite)),
        p95=float(np.percentile(finite, 95)),
        maximum=float(finite.max()),
        count=int(len(finite)),
    )
