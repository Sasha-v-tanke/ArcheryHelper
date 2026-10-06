from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from neural_network.archery_ml.geometry.detector import GeometryResult


class QualityStatus(str, Enum):
    READY = "READY"
    TARGET_NOT_FOUND = "TARGET_NOT_FOUND"
    WRONG_FACE_COUNT = "WRONG_FACE_COUNT"
    TARGET_CLIPPED = "TARGET_CLIPPED"
    EXCESSIVE_PERSPECTIVE = "EXCESSIVE_PERSPECTIVE"
    TOO_BLURRY = "TOO_BLURRY"
    BAD_EXPOSURE = "BAD_EXPOSURE"
    TARGET_TOO_SMALL = "TARGET_TOO_SMALL"


@dataclass(frozen=True)
class QualityThresholds:
    minimum_geometry_confidence: float = 0.2
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

    def evaluate(self, image: np.ndarray, geometry_result: GeometryResult) -> QualityResult:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape H x W x 3")
        metrics = self._metrics(image, geometry_result)
        if not geometry_result.faces:
            return self._result(QualityStatus.TARGET_NOT_FOUND, geometry_result, metrics)
        if len(geometry_result.faces) != geometry_result.expected_face_count:
            return self._result(QualityStatus.WRONG_FACE_COUNT, geometry_result, metrics)
        if not geometry_result.layout_valid:
            return self._result(QualityStatus.TARGET_NOT_FOUND, geometry_result, metrics)
        if geometry_result.geometry_confidence < self.thresholds.minimum_geometry_confidence:
            return self._result(QualityStatus.TARGET_NOT_FOUND, geometry_result, metrics)
        if metrics["minimum_coverage"] < self.thresholds.minimum_coverage:
            return self._result(QualityStatus.TARGET_CLIPPED, geometry_result, metrics)
        if metrics["minimum_perspective_ratio"] < self.thresholds.minimum_perspective_ratio:
            return self._result(QualityStatus.EXCESSIVE_PERSPECTIVE, geometry_result, metrics)
        if (
            metrics["minimum_target_diameter_fraction"]
            < self.thresholds.minimum_target_diameter_fraction
        ):
            return self._result(QualityStatus.TARGET_TOO_SMALL, geometry_result, metrics)
        if metrics["blur_score"] < self.thresholds.minimum_laplacian_variance:
            return self._result(QualityStatus.TOO_BLURRY, geometry_result, metrics)
        if (
            metrics["mean_luma"] < self.thresholds.minimum_mean_luma
            or metrics["mean_luma"] > self.thresholds.maximum_mean_luma
            or metrics["clipped_fraction"] > self.thresholds.maximum_clipped_fraction
        ):
            return self._result(QualityStatus.BAD_EXPOSURE, geometry_result, metrics)
        return self._result(QualityStatus.READY, geometry_result, metrics)

    def _metrics(self, image: np.ndarray, geometry_result: GeometryResult) -> dict[str, float]:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_luma = float(gray.mean())
        clipped_fraction = float(np.mean((gray <= 5) | (gray >= 250)))
        if geometry_result.faces:
            minimum_coverage = min(face.coverage for face in geometry_result.faces)
            minimum_perspective_ratio = min(
                face.perspective_ratio for face in geometry_result.faces
            )
            frame_size = max(1.0, min(geometry_result.image_width, geometry_result.image_height))
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
            minimum_target_diameter_fraction=metrics["minimum_target_diameter_fraction"],
        )
