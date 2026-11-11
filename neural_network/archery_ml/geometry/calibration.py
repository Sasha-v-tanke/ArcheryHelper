"""External board orientation. Never fits target geometry from board/holdout GT."""
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from neural_network.archery_ml.geometry.detector import GeometryResult, _checked_inverse, _matrix_tuple
from neural_network.archery_ml.geometry.quality import ImageQualityGate, QualityStatus
from neural_network.archery_ml.geometry.target_templates import CANONICAL_CENTER_PX, CANONICAL_RADIUS_PX


def content_sha(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def image_sha(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def project(points, matrix) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ np.asarray(matrix).T
    if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) < 1e-10):
        raise ValueError("invalid calibration projection")
    return homogeneous[:, :2] / homogeneous[:, 2:]


@dataclass(frozen=True)
class BoardTemplate:
    payload: dict

    def __post_init__(self):
        p = self.payload
        if p.get("schema_version") != 1 or not p.get("board_template_id") or not p.get("revision"):
            raise ValueError("invalid board template identity/version")
        if p.get("dictionary") != "DICT_4X4_50":
            raise ValueError("unsupported marker dictionary")
        if not np.isfinite([p["width_mm"],p["height_mm"]]).all() or min(p["width_mm"],p["height_mm"]) <= 0:
            raise ValueError("invalid board dimensions")
        count = 1 if p["format"] == "SINGLE" else 3
        if (p["format"] not in {"SINGLE", "TRIPLE"} or len(p["faces"]) != count
                or [f["face_index"] for f in p["faces"]] != list(range(count))):
            raise ValueError("invalid physical face identities")
        if p.get("triple_layout") not in ({None} if count == 1 else {"VERTICAL", "TRIANGULAR"}):
            raise ValueError("invalid board layout")
        ids = [m["id"] for m in p["markers"]]
        if len(set(ids)) != len(ids) or len(ids) < 3 or any(type(i) is not int or not 0 <= i < 50 for i in ids):
            raise ValueError("invalid marker identities")
        if p["quality"]["minimum_markers"] < 3 or any(
                not np.isfinite(v) or v <= 0 for v in p["quality"].values()):
            raise ValueError("invalid calibration quality thresholds")
        for marker in p["markers"]:
            corners = np.asarray(marker["corners_mm"], dtype=float)
            if (corners.shape != (4, 2) or not np.isfinite(corners).all()
                    or cv2.contourArea(corners.astype(np.float32), oriented=True) <= 0
                    or not cv2.isContourConvex(corners.astype(np.float32))):
                raise ValueError("marker corners must follow top-left clockwise order")
        for face in p["faces"]:
            if len(face["center_mm"]) != 2 or not np.isfinite([*face["center_mm"], face["radius_mm"]]).all() or face["radius_mm"] <= 0:
                raise ValueError("invalid physical face size")

    @classmethod
    def load(cls, path: Path):
        return cls(json.loads(path.read_text()))

    @property
    def sha256(self):
        return content_sha(self.payload)


@dataclass(frozen=True)
class BoardCalibration:
    board_template_sha256: str
    image_rgb_sha256: str
    marker_correspondences: tuple
    calibration_revision: str = "aruco-board-orientation-v1"

    def to_dict(self):
        return {"board_template_sha256": self.board_template_sha256, "image_rgb_sha256": self.image_rgb_sha256,
                "marker_correspondences": list(self.marker_correspondences), "calibration_revision": self.calibration_revision}

    @classmethod
    def from_dict(cls, data):
        return cls(data["board_template_sha256"], data["image_rgb_sha256"], tuple(data["marker_correspondences"]),
                   data["calibration_revision"])


@dataclass(frozen=True)
class CalibratedGeometryResult:
    geometry: GeometryResult
    record: dict


class TargetOrientationCalibrator:
    def __init__(self, template: BoardTemplate):
        self.template = template

    def observe(self, image: np.ndarray) -> BoardCalibration:
        detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        observations = () if ids is None else tuple({"marker_id": int(i), "image_corners_px": c.reshape(4, 2).tolist()}
                                                   for i, c in zip(ids.flatten(), corners))
        return BoardCalibration(self.template.sha256, image_sha(image), observations)

    def calibrate(self, image: np.ndarray, geometry: GeometryResult, calibration: BoardCalibration) -> CalibratedGeometryResult:
        if ImageQualityGate().evaluate(image, geometry).status is not QualityStatus.READY:
            raise ValueError("orientation calibration requires geometry READY")
        if calibration.board_template_sha256 != self.template.sha256 or calibration.image_rgb_sha256 != image_sha(image):
            raise ValueError("calibration template/image checksum mismatch")
        if calibration.calibration_revision != "aruco-board-orientation-v1":
            raise ValueError("unsupported calibration revision")
        # Re-observe image evidence: a serialized claim alone cannot confer ANCHORED.
        observed = self.observe(image)
        if content_sha({"observations": observed.to_dict()["marker_correspondences"]}) != content_sha({"observations": calibration.to_dict()["marker_correspondences"]}):
            raise ValueError("calibration correspondences do not match image markers")
        p = self.template.payload
        markers = {m["id"]: m for m in p["markers"]}
        observations = calibration.marker_correspondences
        ids = [o["marker_id"] for o in observations]
        if (len(set(ids)) != len(ids) or any(i not in markers for i in ids)
                or len(ids) < p["quality"]["minimum_markers"]):
            raise ValueError("unknown, duplicate or insufficient markers")
        source = np.concatenate([np.asarray(o["image_corners_px"], dtype=float) for o in observations])
        physical = np.concatenate([np.asarray(markers[o["marker_id"]]["corners_mm"], dtype=float) for o in observations])
        board_to_image, _ = cv2.findHomography(physical, source, method=0)
        if board_to_image is None:
            raise ValueError("degenerate marker correspondences")
        try:
            image_to_board = _checked_inverse(board_to_image)
        except np.linalg.LinAlgError as error:
            raise ValueError("singular board calibration") from error
        error = float(np.linalg.norm(project(physical, board_to_image) - source, axis=1).max())
        if not np.isfinite(error) or error > p["quality"]["maximum_reprojection_px"]:
            raise ValueError("inconsistent marker correspondences")
        if len(geometry.faces) != len(p["faces"]) or {f.face_index for f in geometry.faces} != set(range(len(p["faces"]))):
            raise ValueError("invalid detector face_index/count")
        center = CANONICAL_CENTER_PX
        radius = CANONICAL_RADIUS_PX
        norm = np.array([[1/radius, 0, -center/radius], [0, 1/radius, -center/radius], [0, 0, 1]])
        pixels = _checked_inverse(norm)
        assigned = set()
        faces, records = [], []
        for detected in geometry.faces:
            physical_center = project([detected.center], image_to_board)[0]
            expected = min(p["faces"], key=lambda f: np.linalg.norm(physical_center - f["center_mm"]))
            index = expected["face_index"]
            if index in assigned or np.linalg.norm(physical_center - expected["center_mm"]) > expected["radius_mm"] * p["quality"]["maximum_center_distance_radius"]:
                raise ValueError("ambiguous physical face assignment")
            assigned.add(index)
            # Use the local physical basis only to find a rotation. Do not translate,
            # rescale or replace the detector's projective rectification with board GT.
            x, y = expected["center_mm"]
            step = expected["radius_mm"] * .01
            probes = project([[x,y], [x+step,y], [x,y+step]], board_to_image)
            normalized = project(probes, norm @ np.asarray(detected.homography))
            basis = (normalized[1:] - normalized[0]).T
            if not np.isfinite(basis).all() or np.linalg.det(basis) <= 0:
                raise ValueError("mirrored or degenerate physical axes")
            u, singular, vt = np.linalg.svd(basis)
            if singular[-1] <= 1e-10 or singular[0] / singular[-1] > p["quality"]["maximum_axis_anisotropy"]:
                raise ValueError("detector/board orientation disagreement")
            rotation = np.eye(3)
            rotation[:2, :2] = vt.T @ u.T
            transform = pixels @ rotation @ norm @ np.asarray(detected.homography)
            inverse = _checked_inverse(transform)
            faces.append(replace(detected, face_index=index, homography=_matrix_tuple(transform), inverse_homography=_matrix_tuple(inverse)))
            records.append({"face_index": index, "detector_face_index": detected.face_index,
                            "axis_anisotropy": float(singular[0] / singular[-1]),
                            "center_distance_radius": float(np.linalg.norm(physical_center - expected["center_mm"]) / expected["radius_mm"]),
                            "rotation": rotation.tolist(), "image_to_physical_canonical": transform.tolist(),
                            "physical_canonical_to_image": inverse.tolist()})
        result = replace(geometry, faces=tuple(sorted(faces, key=lambda f: f.face_index)), orientation_reference="physical_marker")
        record = {**calibration.to_dict(), "board_template_id": p["board_template_id"], "board_revision": p["revision"],
                  "orientation_status": "ANCHORED", "maximum_marker_reprojection_px": error,
                  "reference_type": "physical_marker", "faces": records,
                  "orientation_only": True,
                  "quality_status": "development-proposal-not-real-validated"}
        return CalibratedGeometryResult(result, record)
