"""Explicit external axis calibration; concentric rings never imply physical axes."""

from dataclasses import dataclass, replace
import hashlib
import math

import numpy as np


@dataclass(frozen=True)
class OrientationCalibration:
    angle_radians: float
    image_rgb_sha256: str
    reference: str
    revision: str
    author: str

    def __post_init__(self):
        if (
            not math.isfinite(self.angle_radians)
            or len(self.image_rgb_sha256) != 64
            or self.reference not in {"physical_marker", "board", "manual_calibration"}
            or not self.revision
            or not self.author
        ):
            raise ValueError(
                "external orientation requires finite angle and explicit provenance"
            )


def apply_orientation(image, geometry, calibration):
    """Rotate only canonical axes, preserving physical center, radii and face IDs."""
    if (
        hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
        != calibration.image_rgb_sha256
    ):
        raise ValueError("orientation calibration belongs to another image")
    angle = calibration.angle_radians
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    correction = np.eye(3)
    correction[:2, :2] = rotation
    correction[:2, 2] = 255.5 - rotation @ np.full(2, 255.5)
    faces = []
    for face in geometry.faces:
        h = correction @ np.asarray(face.homography)
        inverse = np.linalg.inv(h)
        faces.append(
            replace(
                face,
                homography=tuple(map(tuple, h)),
                inverse_homography=tuple(map(tuple, inverse)),
            )
        )
    return replace(
        geometry,
        faces=tuple(faces),
        orientation_reference=calibration.reference,
        orientation_provenance={
            "image_rgb_sha256": calibration.image_rgb_sha256,
            "angle_radians": angle,
            "reference": calibration.reference,
            "revision": calibration.revision,
            "author": calibration.author,
        },
    )
