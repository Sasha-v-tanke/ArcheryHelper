from __future__ import annotations

import re
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from neural_network.archery_ml.contracts import ImageGeometryAnnotation
from neural_network.archery_ml.data.view import DatasetView
from neural_network.archery_ml.data.geometry_provenance import verify_mobile_gt
from neural_network.archery_ml.geometry.lineage import original_family
from neural_network.archery_ml.geometry.detector import FaceGeometry, GeometryResult, TargetGeometryDetector
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
    split: str
    group_id: str
    family: str
    annotation_issues: tuple[str, ...]
    point_error: GeometryErrorMetrics = field(default_factory=lambda: _metrics([]))
    face_error: dict[str, dict] = field(default_factory=dict)
    capture_metadata: dict = field(default_factory=dict)
    point_values: tuple[float, ...] = ()
    radial_values: tuple[float, ...] = ()
    gt_provenance: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    selection: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "group_id": self.group_id,
            "family": self.family,
            "annotation_issues": list(self.annotation_issues),
            "image_path": self.image_path,
            "status": self.status,
            "detected_face_count": self.detected_face_count,
            "geometry_confidence": self.geometry_confidence,
            "fit_error": self.fit_error,
            "center_error": self.center_error,
            "radial_error": self.radial_error.to_dict(),
            "point_error_2d": self.point_error.to_dict(),
            "point_error_ring_widths": _scaled_metrics(self.point_error),
            "face_error": self.face_error,
            "capture_metadata": self.capture_metadata,
            "gt_provenance": self.gt_provenance,
            "calibration": self.calibration,
            "selection": self.selection,
            "point_values_2d": list(self.point_values),
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
    breakdowns: dict[str, dict]
    point_errors: tuple[float, ...] = ()
    face_point_values: tuple[tuple[float, ...], ...] = ()

    def to_dict(self) -> dict:
        return {
            "samples": self.samples,
            "evaluated_samples": self.evaluated_samples,
            "ready_samples": self.ready_samples,
            "status_counts": self.status_counts,
            "center_error": self.center_error.to_dict(),
            "radial_error": self.radial_error.to_dict(),
            "point_error_2d": _metrics(list(self.point_errors)).to_dict(),
            "point_error_ring_widths": _scaled_metrics(_metrics(list(self.point_errors))),
            "equal_image_mean_2d": _metrics([
                result.point_error.mean for result in self.sample_results
                if result.point_error.mean is not None
            ]).to_dict(),
            "equal_face_mean_2d": _metrics([
                face["point_error_2d"]["mean"] for result in self.sample_results
                for face in result.face_error.values() if face["point_error_2d"]["mean"] is not None
            ]).to_dict(),
            "equal_image_mean_radial": _metrics([
                result.radial_error.mean for result in self.sample_results
                if result.radial_error.mean is not None
            ]).to_dict(),
            "equal_image_weight_2d": _equal_group_metrics([s.point_values for s in self.sample_results]),
            "equal_face_weight_2d": _equal_group_metrics(self.face_point_values),
            "equal_image_weight_radial": _equal_group_metrics([s.radial_values for s in self.sample_results]),
            "ring_error": {
                key: value.to_dict()
                for key, value in sorted(self.ring_error.items(), key=lambda item: int(item[0]))
            },
            "sample_results": [result.to_dict() for result in self.sample_results],
            "target_config": self.target_config,
            "breakdowns": self.breakdowns,
            "annotation_issue_samples": sum(bool(sample.annotation_issues) for sample in self.sample_results),
            "coverage": _coverage(self.sample_results),
            "selection_coverage": dict(Counter(s.selection.get("status","FULL_FRAME") for s in self.sample_results)),
            "metrics_version": "stage3-2d-v3-equal-image",
        }


class GeometryEvaluator:
    def __init__(
        self,
        detector: TargetGeometryDetector | None = None,
        quality_gate: ImageQualityGate | None = None,
        orientation_calibrator=None,
        selected_targets: dict | None = None,
    ):
        self.detector = detector or TargetGeometryDetector()
        self.quality_gate = quality_gate or ImageQualityGate()
        self.orientation_calibrator = orientation_calibrator
        self.selected_targets = selected_targets

    def _selected_geometry(self, image, sample, config, preview_dir):
        # Candidate/group selection reads image evidence and a separately supplied
        # user decision. No evaluation annotations enter this pipeline.
        from neural_network.archery_ml.geometry.selection import TargetCandidateDetector, TargetGroupBuilder, TargetGroupSelector
        from neural_network.archery_ml.geometry.selected_geometry import SelectedTargetGeometryPipeline
        image_sha = hashlib.sha256(sample.image_path.read_bytes()).hexdigest()
        original_sha = (sample.capture_metadata or {}).get("original_sha256",image_sha)
        candidates = TargetCandidateDetector().detect(image)
        groups = TargetGroupBuilder().build(candidates,image)
        empty = GeometryResult((),0.,1.,None,1 if config.format is TargetFormat.SINGLE else 3,False,image.shape[1],image.shape[0])
        record = {"original_sha256":original_sha,"image_sha256":image_sha,"candidates":candidates,"groups":groups}
        try:
            selection = TargetGroupSelector().select(candidates,groups,self.selected_targets.get(original_sha))
            record.update(selection)
            record.update(original_sha256=original_sha,image_sha256=image_sha,candidates=candidates,groups=groups)
            record["ownership_confirmed"] = selection["status"] == "USER_CONFIRMED" and selection.get("source") == "USER"
            if not record["ownership_confirmed"]:
                return empty,"GROUP_SELECTION_REQUIRED",record,{"orientation_status":"UNANCHORED"}
            layout = None if config.triple_layout is None else config.triple_layout.value
            if (selection["format"] != config.format.value or selection["layout"] != layout
                    or selection.get("minimum_scoring_zone",config.minimum_scoring_zone) != config.minimum_scoring_zone):
                raise ValueError("selected configuration differs from evaluation configuration")
            pipeline = SelectedTargetGeometryPipeline(self.orientation_calibrator)
            pipeline.detector, pipeline.gate = self.detector, self.quality_gate
            result,geometry = pipeline.evaluate(image,candidates,selection,config,
                None if preview_dir is None else preview_dir/(image_sha+"-selected"))
            record["pipeline_result"] = result
            return geometry or empty,result["quality_status"],record,result.get("calibration",{"orientation_status":"UNANCHORED"})
        except ValueError as error:
            record.update(status="INVALID_SELECTION",ownership_confirmed=False,reason=str(error))
            return empty,"GROUP_SELECTION_REQUIRED",record,{"orientation_status":"UNANCHORED"}

    def evaluate(
        self,
        view: DatasetView,
        target_config: TargetConfig,
        preview_dir: Path | None = None,
    ) -> GeometryEvaluationReport:
        samples = tuple(sample for sample in view if sample.geometry_annotations or sample.capture_metadata)
        if not samples:
            raise ValueError("no geometry-annotated samples selected")

        status_counts: Counter[str] = Counter()
        center_errors: list[float] = []
        radial_errors: list[float] = []
        ring_errors: dict[str, list[float]] = defaultdict(list)
        sample_results: list[GeometrySampleResult] = []
        evaluated_samples = 0
        ready_samples = 0
        point_errors: list[float] = []
        face_point_values: list[tuple[float, ...]] = []
        template = TargetTemplate.from_config(target_config)
        buckets: dict[str, dict[str, dict]] = defaultdict(dict)
        if preview_dir is not None:
            preview_dir.mkdir(parents=True, exist_ok=True)

        for index, sample in enumerate(samples):
            gt_provenance = (verify_mobile_gt(sample)
                             if (sample.capture_metadata or {}).get("gt_annotations_sha256") else {})
            expected_count = 1 if target_config.format is TargetFormat.SINGLE else 3
            for annotation in sample.geometry_annotations:
                if expected_count > 1 and annotation.face_index is None:
                    raise ValueError("triple evaluation requires explicit face_index")
                if (annotation.face_index or 0) >= expected_count:
                    raise ValueError("geometry face_index outside configured face count")
            for face_index in range(expected_count):
                if sum(a.kind == "target_center" and (a.face_index or 0) == face_index
                       for a in sample.geometry_annotations) > 1:
                    raise ValueError("duplicate physical center for face")
            image = _load_rgb(sample.image_path)
            if sample.target_metadata is not None:
                metadata = sample.target_metadata
                layout = None if target_config.triple_layout is None else target_config.triple_layout.value
                if (metadata.format, metadata.minimum_scoring_zone, metadata.ten_ring_mode, metadata.triple_layout) != (
                    target_config.format.value, target_config.minimum_scoring_zone, target_config.ten_ring_mode.value, layout
                ):
                    raise ValueError("sample target configuration differs from evaluation configuration")
                if metadata.face_diameter_mm is not None and metadata.face_diameter_mm != target_config.face_diameter_mm:
                    raise ValueError("sample physical face diameter differs from evaluation configuration")
            selection_record = {}
            calibration_record = {"orientation_status": "UNANCHORED"}
            if self.selected_targets is not None:
                geometry,status,selection_record,calibration_record = self._selected_geometry(image,sample,target_config,preview_dir)
            else:
                geometry = self.detector.detect(image, target_config)
                status = self.quality_gate.evaluate(image, geometry).status.value
            if self.selected_targets is None and status == "READY" and self.orientation_calibrator is not None:
                try:
                    calibrated = self.orientation_calibrator.calibrate(
                        image, geometry, self.orientation_calibrator.observe(image))
                    geometry, calibration_record = calibrated.geometry, calibrated.record
                except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
                    # Calibration failure never hides a geometry READY or creates a false anchor.
                    calibration_record = {"orientation_status": "UNANCHORED", "reason": str(error)}
            status_counts[status] += 1
            if status == "READY":
                ready_samples += 1

            sample_center_error: float | None = None
            sample_radial_errors: list[float] = []
            sample_ring_errors: dict[str, list[float]] = defaultdict(list)
            sample_point_errors: list[float] = []
            per_face: dict[str, dict] = {}
            face_centers: list[float] = []

            for face in geometry.faces:
                annotations = tuple(a for a in sample.geometry_annotations if (a.face_index or 0) == face.face_index)
                face_points: list[float] = []
                face_radial: list[float] = []
                face_ring_points: dict[str, list[float]] = defaultdict(list)
                matched_image_points = set()
                matched_canonical_points = set()
                measured_image_points = set()
                measured_canonical_points = set()
                predicted_landmarks, expected_landmarks = [], []
                physical_center_error = None
                ground_truth_center = _ground_truth_center(
                    annotations,
                    template,
                )
                if ground_truth_center is not None:
                    center_norm = _to_canonical_norm(
                        face,
                        ground_truth_center,
                        geometry.image_width,
                        geometry.image_height,
                    )
                    face_centers.append(float(np.linalg.norm(center_norm)))
                    center_errors.append(face_centers[-1])
                    if any(a.kind == "target_center" for a in annotations):
                        physical_center_error = face_centers[-1]

                for annotation in annotations:
                    for point, expected in zip(annotation.points, annotation.canonical_points):
                        if tuple(point) in measured_image_points or tuple(expected) in measured_canonical_points:
                            raise ValueError("duplicate evaluation landmark coordinates")
                        measured_image_points.add(tuple(point))
                        measured_canonical_points.add(tuple(expected))
                        predicted = _to_canonical_norm(face, point, geometry.image_width, geometry.image_height)
                        error = float(np.linalg.norm(predicted - expected))
                        if not np.isfinite(error):
                            raise ValueError("non-finite landmark projection")
                        face_points.append(error)
                        predicted_landmarks.append(predicted)
                        expected_landmarks.append(expected)
                        if annotation.annotation_quality == "reviewed" and annotation.uncertainty_norm is not None:
                            matched_image_points.add(tuple(point))
                            matched_canonical_points.add(tuple(expected))
                        ring_key = _ring_key(annotation)
                        if ring_key is not None:
                            face_ring_points[ring_key].append(error)
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
                        face_radial.append(error)
                        radial_errors.append(error)
                        if ring_key is not None:
                            sample_ring_errors[ring_key].append(error)
                            ring_errors[ring_key].append(error)

                sample_point_errors.extend(face_points)
                face_point_values.append(tuple(face_points))
                per_face[str(face.face_index)] = {
                    "point_error_2d": _metrics(face_points).to_dict(),
                    "radial_error": _metrics(face_radial).to_dict(),
                    "ring_point_error_2d": {key: _metrics(values).to_dict() for key, values in face_ring_points.items()},
                    "center_reference": "physical_gt" if any(a.kind == "target_center" for a in annotations) else "ellipse_proxy",
                    "physical_center_error_norm": physical_center_error,
                    "landmark_decomposition": _landmark_decomposition(predicted_landmarks, expected_landmarks),
                    "perspective_axis_ratio": face.perspective_ratio,
                    "visible_coverage": face.coverage,
                    "orientation_reference": geometry.orientation_reference,
                    "independent_landmarks": _independent_landmarks(matched_image_points, matched_canonical_points),
                    "gt_uncertainty_max": max((a.uncertainty_norm for a in annotations
                        if a.uncertainty_norm is not None and (a.canonical_points or a.kind == "target_center")), default=None),
                }

            if face_centers:
                sample_center_error = float(np.mean(face_centers))
            if sample_center_error is not None or sample_radial_errors or sample_point_errors:
                evaluated_samples += 1
            point_errors.extend(sample_point_errors)

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
                    split=sample.split,
                    group_id=sample.group_id or sample.id,
                    family=f"{sample.source_id}:{original_family(sample.image_path)}",
                    annotation_issues=_annotation_issues(sample.geometry_annotations),
                    point_error=_metrics(sample_point_errors),
                    face_error=per_face,
                    capture_metadata=sample.capture_metadata or {},
                    point_values=tuple(sample_point_errors),
                    radial_values=tuple(sample_radial_errors),
                    gt_provenance=gt_provenance,
                    calibration=calibration_record,
                    selection=selection_record,
                )
            )
            for dimension, key in (
                ("acceptance", "ready" if status == "READY" else "rejected"),
                ("status", status),
                ("configuration", f"{target_config.format.value}:{target_config.triple_layout.value if target_config.triple_layout else 'SINGLE'}:{target_config.minimum_scoring_zone}:{target_config.ten_ring_mode.value}"),
                ("split", sample.split),
                ("family", sample_results[-1].family),
                ("group", sample.group_id or sample.id),
                *((dimension, str((sample.capture_metadata or {}).get(dimension, "unknown")))
                  for dimension in ("capture_session_id", "perspective", "scale", "lighting", "device_model")),
            ):
                bucket = buckets[dimension].setdefault(key, {
                    "samples": 0, "evaluated_samples": 0, "ready_samples": 0,
                    "center": [], "radial": [], "points": [], "rings": defaultdict(list),
                })
                bucket["samples"] += 1
                bucket["evaluated_samples"] += int(sample_center_error is not None or bool(sample_radial_errors))
                bucket["ready_samples"] += int(status == "READY")
                if sample_center_error is not None:
                    bucket["center"].append(sample_center_error)
                bucket["radial"].extend(sample_radial_errors)
                bucket["points"].extend(sample_point_errors)
                for ring_key, errors in sample_ring_errors.items():
                    bucket["rings"][ring_key].extend(errors)
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
            breakdowns={
                dimension: {key: _bucket_metrics(bucket) for key, bucket in sorted(values.items())}
                for dimension, values in sorted(buckets.items())
            },
            point_errors=tuple(point_errors),
            face_point_values=tuple(face_point_values),
        )


def _landmark_decomposition(predicted, expected) -> dict:
    """Diagnostics only: fit GT -> prediction; never alter detector coordinates.

    The affine residual can expose remaining projective/nonlinear error. It is
    not an independent measurement of camera perspective or lens distortion.
    """
    if len(expected) < 4:
        return {"status": "INSUFFICIENT_LANDMARKS"}
    expected, predicted = np.asarray(expected), np.asarray(predicted)
    design = np.column_stack((expected, np.ones(len(expected))))
    if np.linalg.matrix_rank(design) < 3:
        return {"status": "DEGENERATE_LANDMARKS"}
    coefficients, _, _, _ = np.linalg.lstsq(design, predicted, rcond=None)
    linear = coefficients[:2].T
    u, singular, vt = np.linalg.svd(linear)
    determinant = float(np.linalg.det(linear))
    if determinant <= 0 or singular[-1] <= 1e-10:
        return {"status": "MIRRORED_OR_DEGENERATE"}
    rotation = u @ vt
    residual = np.linalg.norm(design @ coefficients - predicted, axis=1)
    return {"status": "MEASURED", "scale_bias_fraction": float(np.sqrt(determinant)-1),
        "orientation_error_deg": float(np.rad2deg(np.arctan2(rotation[1,0], rotation[0,0]))),
        "affine_center_offset_norm": float(np.linalg.norm(coefficients[2])),
        "affine_axis_ratio": float(singular[0]/singular[-1]),
        "affine_residual_p95_norm": float(np.percentile(residual,95)),
        "reference": "evaluation landmarks; diagnostic affine fit only, predictions unchanged"}


def _independent_landmarks(image_points: set, canonical_points: set) -> int:
    for points in (image_points, canonical_points):
        array = np.asarray(list(points))
        if len(points) < 4 or np.linalg.matrix_rank(array - array.mean(axis=0)) < 2:
            return 0
    return min(len(image_points), len(canonical_points))


def _bucket_metrics(bucket: dict) -> dict:
    return {
        "samples": bucket["samples"],
        "evaluated_samples": bucket["evaluated_samples"],
        "ready_samples": bucket["ready_samples"],
        "center_error": _metrics(bucket["center"]).to_dict(),
        "radial_error": _metrics(bucket["radial"]).to_dict(),
        "point_error_2d": _metrics(bucket["points"]).to_dict(),
        "ring_error": {key: _metrics(values).to_dict() for key, values in sorted(bucket["rings"].items())},
    }


def _annotation_issues(annotations: tuple[ImageGeometryAnnotation, ...]) -> tuple[str, ...]:
    """Flag contradictory cumulative zones without repairing or excluding GT."""
    indices = {a.face_index for a in annotations}
    if len(indices) > 1:
        return tuple(f"face {index}: {issue}" for index in sorted(indices, key=lambda i: -1 if i is None else i)
                     for issue in _annotation_issues(tuple(a for a in annotations if a.face_index == index)))
    areas: dict[int, float] = defaultdict(float)
    for annotation in annotations:
        score = _ring_score(annotation)
        if score is not None and len(annotation.points) >= 3:
            areas[score] += abs(cv2.contourArea(np.asarray(annotation.points, dtype=np.float32)))
    issues = []
    ordered = sorted(areas)
    for outer, inner in zip(ordered, ordered[1:]):
        if areas[outer] <= areas[inner]:
            issues.append(f"cumulative ring {outer} is not larger than ring {inner}")
    return tuple(issues)


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
    homogeneous = transform @ np.array([x_fraction * image_width, y_fraction * image_height, 1.0])
    if not np.isfinite(homogeneous).all() or abs(homogeneous[2]) < 1e-12 * max(1.0, np.linalg.norm(homogeneous)):
        raise ValueError("landmark lies on invalid homography horizon")
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

    for index in sorted({a.face_index or 0 for a in annotations}):
        face_annotations = tuple(a for a in annotations if (a.face_index or 0) == index)
        ground_truth_center = _ground_truth_center(face_annotations, template)
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


def _scaled_metrics(metrics: GeometryErrorMetrics) -> dict:
    return {key: value if key == "count" or value is None else value * 10
            for key, value in metrics.to_dict().items()}


def _coverage(samples: tuple[GeometrySampleResult, ...]) -> dict:
    usable = [s for s in samples if s.capture_metadata.get("suitability") == "usable"]
    unusable = [s for s in samples if s.capture_metadata.get("suitability") == "unusable"]
    return {
        "usable_samples": len(usable), "unusable_samples": len(unusable),
        "usable_ready_fraction": None if not usable else sum(s.status == "READY" for s in usable) / len(usable),
        "false_reject_fraction": None if not usable else sum(s.status != "READY" for s in usable) / len(usable),
        "false_accept_fraction": None if not unusable else sum(s.status == "READY" for s in unusable) / len(unusable),
        "unknown_suitability_samples": len(samples) - len(usable) - len(unusable),
    }


def _equal_group_metrics(groups) -> dict:
    """Equal mass per group and equal mass per point within it; inverse-CDF quantiles."""
    groups = [values for values in groups if values]
    if not groups:
        return {**_metrics([]).to_dict(), "groups": 0}
    values = np.concatenate([np.asarray(group) for group in groups])
    weights = np.concatenate([np.full(len(group), 1 / len(group)) for group in groups])
    distinct, inverse = np.unique(values, return_inverse=True)
    masses = np.bincount(inverse, weights=weights)
    cumulative = np.cumsum(masses) / masses.sum()
    def quantile(q):
        # Coalesce ties and tolerate probability rounding at exact CDF boundaries.
        return float(distinct[min(np.searchsorted(cumulative, q - 1e-12), len(distinct) - 1)])
    return {"mean": float(np.average(values, weights=weights)), "median": quantile(0.5),
            "p95": quantile(0.95), "max": float(values.max()), "count": len(values), "groups": len(groups)}
