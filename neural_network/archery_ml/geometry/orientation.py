"""Versioned external-axis calibration bound to RGB pixels and geometry."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace

import numpy as np

from neural_network.archery_ml.geometry.benchmark import rgb_pixel_sha256

ORIENTATION_CALIBRATION_VERSION = 2


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def geometry_fingerprint(geometry):
    """Fingerprint the uncalibrated geometry to which an angle belongs."""
    payload = asdict(geometry)
    # Orientation is derived state, not an input to the calibration identity.
    payload.pop("orientation_reference", None)
    payload.pop("orientation_provenance", None)
    return hashlib.sha256(
        json.dumps(
            payload, default=_json_value, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class OrientationCalibration:
    angle_radians: float
    image_rgb_sha256: str
    reference: str
    revision: str
    author: str
    geometry_sha256: str = ""
    contract_version: int = ORIENTATION_CALIBRATION_VERSION

    def __post_init__(self):
        if (
            not math.isfinite(self.angle_radians)
            or self.contract_version != ORIENTATION_CALIBRATION_VERSION
            or len(self.image_rgb_sha256) != 64
            or len(self.geometry_sha256) != 64
            or self.reference not in {"physical_marker", "board", "manual_calibration"}
            or not self.revision
            or not self.author
        ):
            raise ValueError(
                "external orientation requires finite angle and explicit provenance"
            )


def apply_orientation(image, geometry, calibration):
    """Rotate only canonical axes, preserving physical center, radii and face IDs."""
    if not isinstance(calibration, OrientationCalibration):
        raise TypeError("unsupported orientation calibration contract")
    if rgb_pixel_sha256(image) != calibration.image_rgb_sha256:
        raise ValueError("orientation calibration belongs to another image")
    if geometry.orientation_provenance is not None:
        raise ValueError("orientation calibration already applied")
    if geometry_fingerprint(geometry) != calibration.geometry_sha256:
        raise ValueError("orientation calibration belongs to another geometry revision")
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
            "geometry_sha256": calibration.geometry_sha256,
            "contract_version": calibration.contract_version,
            "angle_radians": angle,
            "reference": calibration.reference,
            "revision": calibration.revision,
            "author": calibration.author,
        },
    )
