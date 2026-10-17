from __future__ import annotations

import re
from collections import Counter, defaultdict
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
class GeometrySampleResult:
    sample_id: str
    image_path: str
    status: str
    detected_face_count: int
    geometry_confidence: float
    fit_error: float
    center_error: float | None
    radial_error: GeometryErrorMetrics
    ring_error: dict[str, GeometryErrorMetrics]

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "status": self.status,
            "detected_face_count": self.detected_face_count,
            "geometry_confidence": self.geometry_confidence,
            "fit_error": self.fit_error,
            "center_error": self.center_error,
            "radial_error": self.radial_error.to_dict(),
            "ring_error": {
                key: value.to_dict()
                for key, value in sorted(self.ring_error.items(), key=lambda item: int(item[0]))
            },
        }


@dataclass(frozen=True)
class GeometryEvaluationReport:
    samples: int
    evaluated_samples: int
    ready_samples: int
    status_counts: dict[str, int]
    center_error: GeometryErrorMetrics
    radial_error: GeometryErrorMetrics
    ring_error: dict[str, GeometryErrorMetrics]
    sample_results: tuple[GeometrySampleResult, ...]
    target_config: dict[str, object]

    def to_dict(self) -> dict:
        return {
            "samples": self.samples,
            "evaluated_samples": self.evaluated_samples,
            "ready_samples": self.ready_samples,
            "status_counts": self.status_counts,
            "center_error": self.center_error.to_dict(),
            "radial_error": self.radial_error.to_dict(),
            "ring_error": {
                key: value.to_dict()
                for key, value in sorted(self.ring_error.items(), key=lambda item: int(item[0]))
            },
            "sample_results": [result.to_dict() for result in self.sample_results],
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
        preview_dir: Path | None = None,
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
        ring_errors: dict[str, list[float]] = defaultdict(list)
        sample_results: list[GeometrySampleResult] = []
        evaluated_samples = 0
        ready_samples = 0
        template = TargetTemplate.from_config(target_config)
        if preview_dir is not None:
            preview_dir.mkdir(parents=True, exist_ok=True)

        for index, sample in enumerate(samples):
            image = _load_rgb(sample.image_path)
            geometry = self.detector.detect(image, target_config)
            quality = self.quality_gate.evaluate(image, geometry)
            status = quality.status.value
            status_counts[status] += 1
            if status == "READY":
                ready_samples += 1

            sample_center_error: float | None = None
            sample_radial_errors: list[float] = []
            sample_ring_errors: dict[str, list[float]] = defaultdict(list)

            if len(geometry.faces) == 1:
                face = geometry.faces[0]
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
                    sample_center_error = float(np.linalg.norm(center_norm))
                    center_errors.append(sample_center_error)

                for annotation in sample.geometry_annotations:
                    expected_radius = _expected_radius(annotation, template)
                    if expected_radius is None:
                        continue
                    ring_key = _ring_key(annotation)
                    for point in annotation.points:
                        normalized = _to_canonical_norm(
                            face,
                            point,
                            geometry.image_width,
                            geometry.image_height,
                        )
                        error = abs(float(np.linalg.norm(normalized)) - expected_radius)
                        sample_radial_errors.append(error)
                        radial_errors.append(error)
                        if ring_key is not None:
                            sample_ring_errors[ring_key].append(error)
                            ring_errors[ring_key].append(error)

                if sample_center_error is not None or sample_radial_errors:
                    evaluated_samples += 1

            sample_results.append(
                GeometrySampleResult(
                    sample_id=sample.id,
                    image_path=str(sample.image_path),
                    status=status,
                    detected_face_count=len(geometry.faces),
                    geometry_confidence=geometry.geometry_confidence,
                    fit_error=geometry.fit_error,
                    center_error=sample_center_error,
                    radial_error=_metrics(sample_radial_errors),
                    ring_error={
                        key: _metrics(values)
                        for key, values in sample_ring_errors.items()
                    },
                )
            )
            if preview_dir is not None:
                _render_preview(
                    image,
                    geometry.faces,
                    status,
                    sample.geometry_annotations,
                    template,
                    preview_dir / f"{index:03d}-{sample.image_path.stem}.jpg",
                )

        return GeometryEvaluationReport(
            samples=len(samples),
            evaluated_samples=evaluated_samples,
            ready_samples=ready_samples,
            status_counts=dict(sorted(status_counts.items())),
            center_error=_metrics(center_errors),
            radial_error=_metrics(radial_errors),
            ring_error={
                key: _metrics(values)
                for key, values in ring_errors.items()
            },
            sample_results=tuple(sample_results),
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


def _ring_score(annotation: ImageGeometryAnnotation) -> int | None:
    if annotation.kind != "ring" or annotation.source_label is None:
        return None
    match = _RING_SCORE_RE.search(annotation.source_label)
    if match is None:
        return None
    return int(match.group(1))


def _ring_key(annotation: ImageGeometryAnnotation) -> str | None:
    score = _ring_score(annotation)
    return None if score is None else str(score)


def _expected_radius(
    annotation: ImageGeometryAnnotation,
    template: TargetTemplate,
) -> float | None:
    if annotation.kind == "target_face":
        return float(template.visible_radius_norm)
    score = _ring_score(annotation)
    if score is None:
        return None
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


def _render_preview(
    image: np.ndarray,
    faces: tuple[FaceGeometry, ...],
    status: str,
    annotations: tuple[ImageGeometryAnnotation, ...],
    template: TargetTemplate,
    output_path: Path,
) -> None:
    canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    for annotation in annotations:
        if _ring_score(annotation) not in {1, 5, 10}:
            continue
        points = np.asarray(annotation.points, dtype=np.float64)
        if len(points) < 3:
            continue
        pixel_points = np.rint(points * (width, height)).astype(np.int32)
        cv2.polylines(canvas, [pixel_points], True, (80, 210, 40), 2)

    ground_truth_center = _ground_truth_center(annotations, template)
    if ground_truth_center is not None:
        point = (
            int(round(ground_truth_center[0] * width)),
            int(round(ground_truth_center[1] * height)),
        )
        cv2.drawMarker(canvas, point, (255, 255, 255), cv2.MARKER_CROSS, 18, 2)

    angles = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
    radii = {ring.score: ring.radius_norm for ring in template.ring_boundaries}
    for face in faces:
        point = tuple(int(round(value)) for value in face.center)
        cv2.drawMarker(canvas, point, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
        inverse = np.asarray(face.inverse_homography, dtype=np.float64)
        for score in (1, 5, 10):
            radius = radii.get(score)
            if radius is None:
                continue
            canonical_points = np.column_stack((
                CANONICAL_CENTER_PX + CANONICAL_RADIUS_PX * radius * np.cos(angles),
                CANONICAL_CENTER_PX + CANONICAL_RADIUS_PX * radius * np.sin(angles),
            )).astype(np.float32).reshape(-1, 1, 2)
            projected = cv2.perspectiveTransform(canonical_points, inverse).reshape(-1, 2)
            if not np.isfinite(projected).all() or np.max(np.abs(projected)) > 1e7:
                continue
            cv2.polylines(
                canvas,
                [np.rint(projected).astype(np.int32)],
                True,
                (0, 165, 255),
                2,
            )

    cv2.rectangle(canvas, (0, 0), (min(width, 520), 32), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        f"{status} faces={len(faces)} | green=GT orange=predicted",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.49,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"could not write geometry preview: {output_path}")


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
