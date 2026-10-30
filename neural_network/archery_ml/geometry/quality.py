import math
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from neural_network.archery_ml.geometry.detector import GeometryResult, _checked_inverse
from neural_network.archery_ml.geometry.projective_solver import SolverPolicy


class QualityStatus(str, Enum):
    READY = "READY"
    TARGET_NOT_FOUND = "TARGET_NOT_FOUND"
    WRONG_FACE_COUNT = "WRONG_FACE_COUNT"
    GEOMETRY_UNCERTAIN = "GEOMETRY_UNCERTAIN"
    TARGET_CLIPPED = "TARGET_CLIPPED"
    EXCESSIVE_PERSPECTIVE = "EXCESSIVE_PERSPECTIVE"
    TOO_BLURRY = "TOO_BLURRY"
    BAD_EXPOSURE = "BAD_EXPOSURE"
    TARGET_TOO_SMALL = "TARGET_TOO_SMALL"


@dataclass(frozen=True)
class QualityThresholds:
    minimum_geometry_confidence: float = 0.2
    maximum_ring_fit_error: float = 0.03
    minimum_ring_count: int = 2
    minimum_coverage: float = 0.97
    minimum_perspective_ratio: float = 0.55
    minimum_target_diameter_fraction: float = 0.14
    minimum_laplacian_variance: float = 45.0
    minimum_mean_luma: float = 35.0
    maximum_mean_luma: float = 220.0
    maximum_clipped_fraction: float = 0.45


@dataclass(frozen=True)
class QualityResult:
    status: QualityStatus
    geometry_confidence: float
    blur_score: float
    mean_luma: float
    clipped_fraction: float
    minimum_coverage: float
    minimum_perspective_ratio: float
    minimum_target_diameter_fraction: float


class ImageQualityGate:
    def __init__(self, thresholds: QualityThresholds | None = None):
        self.thresholds = thresholds or QualityThresholds()

    def evaluate(
        self, image: np.ndarray, geometry_result: GeometryResult
    ) -> QualityResult:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape H x W x 3")
        for face in geometry_result.faces:
            try:
                inverse = _checked_inverse(np.asarray(face.homography))
                supplied = np.asarray(face.inverse_homography, dtype=np.float64)
                _checked_inverse(supplied)
                inverse /= np.max(np.abs(inverse))
                supplied = supplied / np.max(np.abs(supplied))
                values = (
                    *face.center,
                    *face.visible_bounds,
                    face.coverage,
                    face.geometry_confidence,
                    face.ellipse.center_x,
                    face.ellipse.center_y,
                    face.ellipse.diameter_x,
                    face.ellipse.diameter_y,
                    face.ellipse.angle_deg,
                )
                if (
                    not np.isfinite(values).all()
                    or not math.isfinite(geometry_result.geometry_confidence)
                    or face.ellipse.diameter_x <= 0
                    or face.ellipse.diameter_y <= 0
                    or not 0 <= face.coverage <= 1
                    or not 0 <= face.geometry_confidence <= 1
                    or not (
                        np.allclose(inverse, supplied, atol=1e-7)
                        or np.allclose(inverse, -supplied, atol=1e-7)
                    )
                ):
                    raise np.linalg.LinAlgError("invalid face geometry")
            except np.linalg.LinAlgError:
                return self._result(
                    QualityStatus.GEOMETRY_UNCERTAIN,
                    geometry_result,
                    {
                        "blur_score": 0.0,
                        "mean_luma": 0.0,
                        "clipped_fraction": 0.0,
                        "minimum_coverage": 0.0,
                        "minimum_perspective_ratio": 0.0,
                        "minimum_target_diameter_fraction": 0.0,
                    },
                )
        metrics = self._metrics(image, geometry_result)
        if not geometry_result.faces:
            return self._result(
                QualityStatus.TARGET_NOT_FOUND, geometry_result, metrics
            )
        if len(geometry_result.faces) != geometry_result.expected_face_count:
            return self._result(
                QualityStatus.WRONG_FACE_COUNT, geometry_result, metrics
            )
        if not geometry_result.layout_valid:
            return self._result(
                QualityStatus.TARGET_NOT_FOUND, geometry_result, metrics
            )
        for face in geometry_result.faces:
            if face.diagnostics_version:
                policy = SolverPolicy()
                values = (
                    face.center_uncertainty_norm,
                    face.center_stability_norm,
                    face.angular_coverage,
                    face.cross_ring_consistency,
                    face.hypothesis_disagreement,
                    face.numerical_conditioning,
                    face.held_sector_error,
                )
                if (
                    face.diagnostics_version != 1
                    or face.solver_status != "AUTO_READY"
                    or face.rejection_reasons
                    or any(v is None for v in values)
                    or not np.isfinite(values).all()
                    or min(values) < 0
                ):
                    return self._result(
                        QualityStatus.GEOMETRY_UNCERTAIN, geometry_result, metrics
                    )
                center = np.asarray(face.homography) @ np.array([*face.center, 1.0])
                if (
                    face.center_uncertainty_norm > policy.maximum_center_spread
                    or face.center_stability_norm > policy.maximum_center_spread
                    or face.angular_coverage < policy.minimum_angular_fraction
                    or face.observed_ring_count < 2
                    or face.held_sector_error > policy.maximum_validation_error
                    or face.numerical_conditioning > 1e5
                    or abs(center[2]) < 1e-10
                    or np.linalg.norm(center[:2] / center[2] - 255.5) > 0.01
                ):
                    return self._result(
                        QualityStatus.GEOMETRY_UNCERTAIN, geometry_result, metrics
                    )
        if (
            geometry_result.geometry_confidence
            < self.thresholds.minimum_geometry_confidence
        ):
            return self._result(
                QualityStatus.TARGET_NOT_FOUND, geometry_result, metrics
            )
        if any(
            not math.isfinite(face.fit_error)
            or face.fit_error > self.thresholds.maximum_ring_fit_error
            or face.ring_count < self.thresholds.minimum_ring_count
            for face in geometry_result.faces
        ):
            return self._result(
                QualityStatus.GEOMETRY_UNCERTAIN, geometry_result, metrics
            )
        if metrics["minimum_coverage"] < self.thresholds.minimum_coverage:
            return self._result(QualityStatus.TARGET_CLIPPED, geometry_result, metrics)
        if (
            metrics["minimum_perspective_ratio"]
            < self.thresholds.minimum_perspective_ratio
        ):
            return self._result(
                QualityStatus.EXCESSIVE_PERSPECTIVE, geometry_result, metrics
            )
        if (
            metrics["minimum_target_diameter_fraction"]
            < self.thresholds.minimum_target_diameter_fraction
        ):
            return self._result(
                QualityStatus.TARGET_TOO_SMALL, geometry_result, metrics
            )
        if metrics["blur_score"] < self.thresholds.minimum_laplacian_variance:
            return self._result(QualityStatus.TOO_BLURRY, geometry_result, metrics)
        if (
            metrics["mean_luma"] < self.thresholds.minimum_mean_luma
            or metrics["mean_luma"] > self.thresholds.maximum_mean_luma
            or metrics["clipped_fraction"] > self.thresholds.maximum_clipped_fraction
        ):
            return self._result(QualityStatus.BAD_EXPOSURE, geometry_result, metrics)
        return self._result(QualityStatus.READY, geometry_result, metrics)

    def _metrics(
        self, image: np.ndarray, geometry_result: GeometryResult
    ) -> dict[str, float]:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        roi = np.zeros(gray.shape, dtype=np.uint8)
        for face in geometry_result.faces:
            ellipse = face.ellipse
            cv2.ellipse(
                roi,
                (
                    (ellipse.center_x, ellipse.center_y),
                    (ellipse.diameter_x, ellipse.diameter_y),
                    ellipse.angle_deg,
                ),
                255,
                -1,
            )
        selected = roi != 0 if np.any(roi) else np.ones(gray.shape, dtype=bool)
        # The surrounding wall must not determine target focus or exposure.
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F)[selected].var())
        target_luma = gray[selected]
        mean_luma = float(target_luma.mean())
        clipped_fraction = float(np.mean((target_luma <= 5) | (target_luma >= 250)))
        if geometry_result.faces:
            minimum_coverage = min(face.coverage for face in geometry_result.faces)
            minimum_perspective_ratio = min(
                face.perspective_ratio for face in geometry_result.faces
            )
            frame_size = max(
                1.0, min(geometry_result.image_width, geometry_result.image_height)
            )
            minimum_target_diameter_fraction = min(
                face.visible_diameter_px / frame_size for face in geometry_result.faces
            )
        else:
            minimum_coverage = 0.0
            minimum_perspective_ratio = 0.0
            minimum_target_diameter_fraction = 0.0
        return {
            "blur_score": blur_score,
            "mean_luma": mean_luma,
            "clipped_fraction": clipped_fraction,
            "minimum_coverage": float(minimum_coverage),
            "minimum_perspective_ratio": float(minimum_perspective_ratio),
            "minimum_target_diameter_fraction": float(minimum_target_diameter_fraction),
        }

    def _result(
        self,
        status: QualityStatus,
        geometry_result: GeometryResult,
        metrics: dict[str, float],
    ) -> QualityResult:
        return QualityResult(
            status=status,
            geometry_confidence=geometry_result.geometry_confidence,
            blur_score=metrics["blur_score"],
            mean_luma=metrics["mean_luma"],
            clipped_fraction=metrics["clipped_fraction"],
            minimum_coverage=metrics["minimum_coverage"],
            minimum_perspective_ratio=metrics["minimum_perspective_ratio"],
            minimum_target_diameter_fraction=metrics[
                "minimum_target_diameter_fraction"
            ],
        )
