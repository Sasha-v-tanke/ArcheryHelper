from dataclasses import dataclass

import cv2
import numpy as np

from neural_network.archery_ml.geometry.detector import GeometryResult
from neural_network.archery_ml.geometry.quality import ImageQualityGate, QualityStatus
from neural_network.archery_ml.geometry.target_templates import CANONICAL_SIZE


@dataclass(frozen=True)
class CanonicalFace:
    face_index: int
    image: np.ndarray
    original_to_canonical: tuple[tuple[float, float, float], ...]
    canonical_to_original: tuple[tuple[float, float, float], ...]
    geometry_confidence: float
    orientation_reference: str = "unanchored_image_or_layout_axes"


class TargetNormalizer:
    def __init__(self, quality_gate: ImageQualityGate | None = None):
        self.quality_gate = quality_gate or ImageQualityGate()

    def normalize(self, image: np.ndarray, geometry_result: GeometryResult) -> list[CanonicalFace]:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape H x W x 3")
        quality = self.quality_gate.evaluate(image, geometry_result)
        if quality.status is not QualityStatus.READY:
            raise ValueError(f"target normalization requires READY: {quality.status.value}")
        faces = []
        for face in geometry_result.faces:
            transform = np.asarray(face.homography, dtype=np.float64)
            normalized = cv2.warpPerspective(
                image,
                transform,
                (CANONICAL_SIZE, CANONICAL_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            faces.append(
                CanonicalFace(
                    face_index=face.face_index,
                    image=normalized,
                    original_to_canonical=face.homography,
                    canonical_to_original=face.inverse_homography,
                    geometry_confidence=face.geometry_confidence,
                    orientation_reference=geometry_result.orientation_reference,
                )
            )
        return faces
