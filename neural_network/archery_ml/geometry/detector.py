from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from neural_network.archery_ml.geometry.target_templates import (
    CANONICAL_CENTER_PX,
    CANONICAL_RADIUS_PX,
    GeometryTemplate,
    geometry_template,
)
from neural_network.archery_ml.targets import TargetConfig, TripleLayout


@dataclass(frozen=True)
class EllipseFit:
    center_x: float
    center_y: float
    diameter_x: float
    diameter_y: float
    angle_deg: float

    @property
    def major_axis_px(self) -> float:
        return max(self.diameter_x, self.diameter_y)

    @property
    def minor_axis_px(self) -> float:
        return min(self.diameter_x, self.diameter_y)

    @property
    def axis_ratio(self) -> float:
        if self.major_axis_px <= 0.0:
            return 0.0
        return self.minor_axis_px / self.major_axis_px

    def scaled(self, factor: float) -> "EllipseFit":
        return EllipseFit(
            center_x=self.center_x,
            center_y=self.center_y,
            diameter_x=self.diameter_x * factor,
            diameter_y=self.diameter_y * factor,
            angle_deg=self.angle_deg,
        )


@dataclass(frozen=True)
class FaceGeometry:
    face_index: int
    center: tuple[float, float]
    ellipse: EllipseFit
    homography: tuple[tuple[float, float, float], ...]
    inverse_homography: tuple[tuple[float, float, float], ...]
    fit_error: float
    geometry_confidence: float
    visible_bounds: tuple[float, float, float, float]
    coverage: float
    ring_count: int

    @property
    def perspective_ratio(self) -> float:
        return self.ellipse.axis_ratio

    @property
    def visible_diameter_px(self) -> float:
        return self.ellipse.major_axis_px


@dataclass(frozen=True)
class GeometryResult:
    faces: tuple[FaceGeometry, ...]
    geometry_confidence: float
    fit_error: float
    visible_bounds: tuple[float, float, float, float] | None
    expected_face_count: int
    layout_valid: bool
    image_width: int
    image_height: int

    @property
    def face_centers(self) -> tuple[tuple[float, float], ...]:
        return tuple(face.center for face in self.faces)


@dataclass(frozen=True)
class _Candidate:
    contour: np.ndarray
    ellipse: EllipseFit
    area: float


class TargetGeometryDetector:
    def detect(self, image: np.ndarray, target_config: TargetConfig) -> GeometryResult:
        rgb = _validate_image(image)
        height, width = rgb.shape[:2]
        template = geometry_template(target_config)
        masks = _target_color_masks(rgb)
        candidates = _face_candidates(masks, width, height)
        ordered_candidates, layout_valid = _order_candidates(candidates, template)
        orientation_axes = _layout_axes(ordered_candidates, template)
        faces = tuple(
            self._fit_face(index, candidate, masks, template, width, height, orientation_axes)
            for index, candidate in enumerate(ordered_candidates)
        )
        bounds = _aggregate_bounds(faces)
        fit_error = float(np.mean([face.fit_error for face in faces])) if faces else 1.0
        confidence = float(np.mean([face.geometry_confidence for face in faces])) if faces else 0.0
        if len(faces) != template.expected_face_count:
            confidence *= 0.4
        if not layout_valid:
            confidence *= 0.4
        return GeometryResult(
            faces=faces,
            geometry_confidence=confidence,
            fit_error=fit_error,
            visible_bounds=bounds,
            expected_face_count=template.expected_face_count,
            layout_valid=layout_valid,
            image_width=width,
            image_height=height,
        )

    def _fit_face(
        self,
        face_index: int,
        candidate: _Candidate,
        masks: dict[str, np.ndarray],
        template: GeometryTemplate,
        image_width: int,
        image_height: int,
        orientation_axes: tuple[np.ndarray, np.ndarray],
    ) -> FaceGeometry:
        rings = _associated_rings(candidate, masks, template)
        image_to_norm = _initial_image_to_norm(
            candidate,
            template.colored_outer_radius_norm,
            orientation_axes,
        )
        if rings:
            refined_transform = _fit_projective_transform(image_to_norm, rings)
            image_to_norm = _stable_transform(image_to_norm, refined_transform, rings)
        if _ring_fit_error(image_to_norm, rings) > 0.06:
            fallback_transform = _reliable_inner_ring_transform(candidate, masks, template)
            if fallback_transform is not None:
                image_to_norm = fallback_transform
        fit_error = _ring_fit_error(image_to_norm, rings)
        canonical_transform = np.array(
            [
                [CANONICAL_RADIUS_PX, 0.0, CANONICAL_CENTER_PX],
                [0.0, CANONICAL_RADIUS_PX, CANONICAL_CENTER_PX],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ) @ image_to_norm
        inverse_transform = np.linalg.inv(canonical_transform)
        angles = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
        boundary = template.visible_radius_norm * np.column_stack((np.cos(angles), np.sin(angles)))
        norm_to_image = np.linalg.inv(image_to_norm)
        projected = _perspective_points(boundary, norm_to_image)
        visible_ellipse = _ellipse_from_cv(cv2.fitEllipse(projected.astype(np.float32)))
        visible_bounds = (
            float(projected[:, 0].min()), float(projected[:, 1].min()),
            float(projected[:, 0].max()), float(projected[:, 1].max()),
        )
        center = _perspective_points(np.zeros((1, 2)), norm_to_image)[0]
        coverage = _bounds_coverage(visible_bounds, image_width, image_height)
        scale_fraction = visible_ellipse.major_axis_px / max(1.0, min(image_width, image_height))
        fit_score = math.exp(-fit_error / 0.03)
        ring_score = min(1.0, len(rings) / 3.0)
        perspective_score = min(1.0, visible_ellipse.axis_ratio / 0.7)
        coverage_score = min(1.0, coverage / 0.98)
        scale_score = min(1.0, scale_fraction / 0.3)
        confidence = (
            0.35 * fit_score
            + 0.2 * ring_score
            + 0.15 * perspective_score
            + 0.15 * coverage_score
            + 0.15 * scale_score
        )
        return FaceGeometry(
            face_index=face_index,
            center=(float(center[0]), float(center[1])),
            ellipse=visible_ellipse,
            homography=_matrix_tuple(canonical_transform),
            inverse_homography=_matrix_tuple(inverse_transform),
            fit_error=fit_error,
            geometry_confidence=float(max(0.0, min(1.0, confidence))),
            visible_bounds=visible_bounds,
            coverage=coverage,
            ring_count=len(rings),
        )


def _validate_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape H x W x 3")
    if image.shape[0] < 16 or image.shape[1] < 16:
        raise ValueError("image is too small for target geometry")
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def _target_color_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    saturation = hsv[:, :, 1]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]

    yellow_hsv = cv2.inRange(hsv, np.array([14, 55, 35]), np.array([42, 255, 255]))
    red_hsv = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 55, 30]), np.array([13, 255, 255])),
        cv2.inRange(hsv, np.array([165, 55, 30]), np.array([179, 255, 255])),
    )
    blue_hsv = cv2.inRange(hsv, np.array([85, 45, 25]), np.array([140, 255, 255]))

    yellow_lab = np.where(
        (b_channel >= 155) & (a_channel <= 150) & (saturation >= 35),
        255,
        0,
    ).astype(np.uint8)
    red_lab = np.where(
        (a_channel >= 155) & (b_channel >= 140) & (saturation >= 35),
        255,
        0,
    ).astype(np.uint8)
    blue_lab = np.where(
        (b_channel <= 120) & (a_channel <= 170) & (saturation >= 30),
        255,
        0,
    ).astype(np.uint8)

    masks = {
        "yellow": cv2.bitwise_or(yellow_hsv, yellow_lab),
        "red": cv2.bitwise_or(red_hsv, red_lab),
        "blue": cv2.bitwise_or(blue_hsv, blue_lab),
    }
    # Hue is unstable near neutral: a cool black ring can have the same HSV
    # hue as blue. Require actual chroma before interpreting it as radius 0.6.
    chroma = np.hypot(
        a_channel.astype(np.float32) - 128.0,
        b_channel.astype(np.float32) - 128.0,
    )
    blue_dominant = image[:, :, 2].astype(np.int16) > image[:, :, 0].astype(np.int16)
    masks["blue"][(chroma < 25.0) | ~blue_dominant] = 0
    masks["red"][chroma < 25.0] = 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    for name, mask in tuple(masks.items()):
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        masks[name] = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return masks


def _face_candidates(
    masks: dict[str, np.ndarray],
    image_width: int,
    image_height: int,
) -> list[_Candidate]:
    primary = _candidates_from_mask(masks["blue"], image_width, image_height)
    # A split annulus has displaced component centers. Associate its pieces
    # through a common red interior rather than treating them as extra faces.
    inner_candidates = _candidates_from_mask(masks["red"], image_width, image_height)
    grouped: dict[int, list[_Candidate]] = {}
    unmatched = []
    for candidate in primary:
        matches = [
            (math.hypot(candidate.ellipse.center_x - inner.ellipse.center_x,
                        candidate.ellipse.center_y - inner.ellipse.center_y), index)
            for index, inner in enumerate(inner_candidates)
            if 0.5 <= candidate.ellipse.major_axis_px / inner.ellipse.major_axis_px <= 2.0
        ]
        distance, index = min(matches, default=(float("inf"), -1))
        if index >= 0 and distance < 0.75 * inner_candidates[index].ellipse.major_axis_px:
            grouped.setdefault(index, []).append(candidate)
        else:
            unmatched.append(candidate)
    primary = unmatched
    for fragments in grouped.values():
        if len(fragments) == 1:
            primary.extend(fragments)
        else:
            contour = cv2.convexHull(np.concatenate([part.contour for part in fragments]).astype(np.float32))
            primary.append(_Candidate(contour, _ellipse_from_cv(cv2.fitEllipse(contour)),
                                      sum(part.area for part in fragments)))
    if not primary:
        union = cv2.bitwise_or(masks["yellow"], cv2.bitwise_or(masks["red"], masks["blue"]))
        primary = _candidates_from_mask(union, image_width, image_height)
    if not primary:
        return []
    primary.sort(key=lambda candidate: candidate.area, reverse=True)
    area_floor = primary[0].area * 0.2
    selected: list[_Candidate] = []
    for candidate in primary:
        if candidate.area < area_floor:
            continue
        overlapping_index = None
        for index, existing in enumerate(selected):
            distance = math.hypot(
                candidate.ellipse.center_x - existing.ellipse.center_x,
                candidate.ellipse.center_y - existing.ellipse.center_y,
            )
            if distance < 0.55 * max(candidate.ellipse.major_axis_px, existing.ellipse.major_axis_px):
                overlapping_index = index
                break
        if overlapping_index is None:
            selected.append(candidate)
            continue

        existing = selected[overlapping_index]
        if candidate.area >= 0.45 * existing.area:
            contour = cv2.convexHull(np.concatenate((existing.contour, candidate.contour), axis=0).astype(np.float32))
            selected[overlapping_index] = _Candidate(
                contour=contour,
                ellipse=_ellipse_from_cv(cv2.fitEllipse(contour)),
                area=existing.area + candidate.area,
            )
    return selected


def _candidates_from_mask(
    mask: np.ndarray,
    image_width: int,
    image_height: int,
) -> list[_Candidate]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_area = max(80.0, image_width * image_height * 0.00035)
    result = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or len(contour) < 5:
            continue
        contour = _ellipse_inliers(contour)
        ellipse_raw = cv2.fitEllipse(contour)
        ellipse = _ellipse_from_cv(ellipse_raw)
        if ellipse.minor_axis_px < 12.0 or ellipse.axis_ratio < 0.2:
            continue
        result.append(_Candidate(contour=contour, ellipse=ellipse, area=area))
    return result


def _associated_rings(
    candidate: _Candidate,
    masks: dict[str, np.ndarray],
    template: GeometryTemplate,
) -> list[tuple[np.ndarray, float]]:
    rings = []
    for color, radius in template.ring_outer_radii:
        if color == "blue":
            rings.append((_sample_contour(candidate.contour, 96), radius))
            continue
        contours, _ = cv2.findContours(masks[color], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        points = np.concatenate(contours).reshape(-1, 2).astype(np.float32)
        ellipse = candidate.ellipse
        angle = math.radians(ellipse.angle_deg)
        delta = points - (ellipse.center_x, ellipse.center_y)
        x = delta[:, 0] * math.cos(angle) + delta[:, 1] * math.sin(angle)
        y = -delta[:, 0] * math.sin(angle) + delta[:, 1] * math.cos(angle)
        relative_radius = np.hypot(2 * x / ellipse.diameter_x, 2 * y / ellipse.diameter_y)
        relative_radius *= template.colored_outer_radius_norm / radius
        # Gather all outer arcs, including disconnected pieces, but exclude
        # the inner boundary and shafts/fletchings outside this color zone.
        points = points[(relative_radius >= 0.75) & (relative_radius <= 1.25)]
        if len(points) < 12:
            continue
        inliers = _ellipse_inliers(points.reshape(-1, 1, 2))
        rings.append((_sample_contour(inliers, 96), radius))
    return rings


def _ellipse_inliers(contour: np.ndarray) -> np.ndarray:
    """Discard arrow edges and the inner edge of an interrupted color annulus.

    The outer arc supplies the largest ellipse consensus. A fixed seed keeps
    detection reproducible; points remain measured pixels, not ellipse samples.
    """
    points = _sample_contour(contour, 384)
    if len(points) < 12:
        return contour
    reference = _ellipse_from_cv(cv2.fitEllipse(points))
    rng = np.random.default_rng(42)
    best = np.ones(len(points), dtype=bool)
    best_count = 0
    for _ in range(96):
        ellipse = _ellipse_from_cv(cv2.fitEllipse(points[rng.choice(len(points), 5, replace=False)]))
        if ellipse.minor_axis_px <= 0 or ellipse.axis_ratio < 0.2:
            continue
        if not 0.65 <= ellipse.major_axis_px / max(reference.major_axis_px, 1.0) <= 1.5:
            continue
        if math.hypot(ellipse.center_x - reference.center_x,
                      ellipse.center_y - reference.center_y) > 0.3 * reference.major_axis_px:
            continue
        angle = math.radians(ellipse.angle_deg)
        delta = points - (ellipse.center_x, ellipse.center_y)
        x = delta[:, 0] * math.cos(angle) + delta[:, 1] * math.sin(angle)
        y = -delta[:, 0] * math.sin(angle) + delta[:, 1] * math.cos(angle)
        residual = np.abs(np.hypot(2 * x / ellipse.diameter_x, 2 * y / ellipse.diameter_y) - 1)
        inliers = residual < 0.035
        count = int(inliers.sum())
        if count > best_count:
            best, best_count = inliers, count
    if best_count < max(12, len(points) * 0.45):
        return contour
    return points[best].reshape(-1, 1, 2)


def _initial_image_to_norm(
    candidate: _Candidate,
    radius_norm: float,
    orientation_axes: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    center = np.array(
        [candidate.ellipse.center_x, candidate.ellipse.center_y],
        dtype=np.float64,
    )
    contour_points = candidate.contour.reshape(-1, 2).astype(np.float64) - center
    linear_rows = []
    for axis in orientation_axes:
        unit = np.asarray(axis, dtype=np.float64)
        unit /= max(float(np.linalg.norm(unit)), 1e-6)
        projection = contour_points @ unit
        diameter = max(float(projection.max() - projection.min()), 1e-6)
        linear_rows.append(unit * (2.0 * radius_norm / diameter))
    linear = np.vstack(linear_rows)
    translation = -linear @ center
    return np.array(
        [
            [linear[0, 0], linear[0, 1], translation[0]],
            [linear[1, 0], linear[1, 1], translation[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _fit_projective_transform(
    initial_transform: np.ndarray,
    rings: list[tuple[np.ndarray, float]],
) -> np.ndarray:
    transform = initial_transform
    for _ in range(4):
        source_points = []
        target_points = []
        for contour, radius in rings:
            mapped = _perspective_points(contour, transform)
            angles = np.arctan2(mapped[:, 1], mapped[:, 0])
            targets = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
            source_points.append(contour)
            target_points.append(targets.astype(np.float32))
        if not source_points:
            break
        source = np.concatenate(source_points, axis=0).astype(np.float32)
        target = np.concatenate(target_points, axis=0).astype(np.float32)
        fitted, _ = cv2.findHomography(source, target, method=0)
        if fitted is None or not np.isfinite(fitted).all():
            break
        transform = fitted.astype(np.float64)
    return transform


def _stable_transform(
    initial: np.ndarray,
    refined: np.ndarray,
    rings: list[tuple[np.ndarray, float]],
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    boundary = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
    source = _perspective_points(boundary, np.linalg.inv(initial))
    inverse = np.linalg.inv(refined)
    # Low residuals on inner rings do not prove a safe outer extrapolation.
    # Reject a projective horizon through the full canonical face.
    if abs(inverse[2, 2]) <= float(np.linalg.norm(inverse[2, :2])):
        return initial
    refined_boundary = _perspective_points(source, refined)
    deviation = np.linalg.norm(refined_boundary - boundary, axis=1)
    if not np.isfinite(deviation).all() or float(np.max(deviation)) > 0.12:
        return initial
    return refined


def _reliable_inner_ring_transform(
    candidate: _Candidate,
    masks: dict[str, np.ndarray],
    template: GeometryTemplate,
) -> np.ndarray | None:
    reference_radius_px = (
        candidate.ellipse.major_axis_px / (2.0 * template.colored_outer_radius_norm)
    )
    center = np.asarray(
        (candidate.ellipse.center_x, candidate.ellipse.center_y),
        dtype=np.float64,
    )
    height, width = masks["red"].shape
    yy, xx = np.ogrid[:height, :width]

    for color, radius_norm, minimum_fill in (
        ("red", 0.4, 0.92),
        ("yellow", 0.2, 0.88),
    ):
        search_radius = reference_radius_px * radius_norm * 1.2 + 6.0
        within_search = (
            (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        ) <= search_radius ** 2
        cropped_mask = np.where(within_search, masks[color], 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            cropped_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            if len(contour) < 5:
                continue
            area = float(cv2.contourArea(contour))
            if area < 80.0:
                continue
            ellipse = _ellipse_from_cv(cv2.fitEllipse(contour))
            if math.hypot(
                ellipse.center_x - center[0],
                ellipse.center_y - center[1],
            ) > 0.25 * candidate.ellipse.major_axis_px:
                continue
            ellipse_radius_px = (
                (ellipse.diameter_x + ellipse.diameter_y) / (4.0 * radius_norm)
            )
            if not 0.85 <= ellipse_radius_px / reference_radius_px <= 1.15:
                continue
            ellipse_area = math.pi * ellipse.diameter_x * ellipse.diameter_y / 4.0
            if area / max(ellipse_area, 1.0) < minimum_fill:
                continue

            angle = math.radians(ellipse.angle_deg)
            rotation = np.array(
                [
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)],
                ],
                dtype=np.float64,
            )
            scaling = np.diag(
                [
                    2.0 * radius_norm / ellipse.diameter_x,
                    2.0 * radius_norm / ellipse.diameter_y,
                ]
            )
            linear = rotation @ scaling @ rotation.T
            translation = -linear @ np.array(
                [ellipse.center_x, ellipse.center_y],
                dtype=np.float64,
            )
            return np.array(
                [
                    [linear[0, 0], linear[0, 1], translation[0]],
                    [linear[1, 0], linear[1, 1], translation[1]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
    return None


def _ring_fit_error(
    transform: np.ndarray,
    rings: list[tuple[np.ndarray, float]],
) -> float:
    residuals = []
    for contour, radius in rings:
        mapped = _perspective_points(contour, transform)
        distances = np.linalg.norm(mapped, axis=1)
        residuals.extend((distances - radius).tolist())
    if not residuals:
        return 0.25
    values = np.asarray(residuals, dtype=np.float64)
    return float(math.sqrt(float(np.mean(values * values))))


def _perspective_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    reshaped = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(reshaped, transform.astype(np.float64))
    return mapped.reshape(-1, 2).astype(np.float64)


def _sample_contour(contour: np.ndarray, limit: int) -> np.ndarray:
    points = contour.reshape(-1, 2)
    if len(points) <= limit:
        return points.astype(np.float32)
    indices = np.linspace(0, len(points) - 1, limit, dtype=np.int32)
    return points[indices].astype(np.float32)


def _order_candidates(
    candidates: list[_Candidate],
    template: GeometryTemplate,
) -> tuple[list[_Candidate], bool]:
    if template.expected_face_count == 1:
        ordered = sorted(candidates, key=lambda item: item.area, reverse=True)
        return ordered, len(ordered) == 1
    if len(candidates) != 3:
        return sorted(candidates, key=lambda item: (item.ellipse.center_y, item.ellipse.center_x)), False
    if template.triple_layout is TripleLayout.VERTICAL:
        return _order_vertical(candidates)
    if template.triple_layout is TripleLayout.TRIANGULAR:
        return _order_triangular(candidates)
    return candidates, False


def _order_vertical(candidates: list[_Candidate]) -> tuple[list[_Candidate], bool]:
    centers = np.asarray(
        [[candidate.ellipse.center_x, candidate.ellipse.center_y] for candidate in candidates],
        dtype=np.float64,
    )
    centered = centers - centers.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if abs(axis[1]) >= abs(axis[0]):
        if axis[1] < 0.0:
            axis = -axis
    elif axis[0] < 0.0:
        axis = -axis
    orthogonal = np.array([-axis[1], axis[0]])
    projection = centered @ axis
    cross = centered @ orthogonal
    order = np.argsort(projection)
    ordered = [candidates[int(index)] for index in order]
    sorted_projection = projection[order]
    gaps = np.diff(sorted_projection)
    median_diameter = float(np.median([candidate.ellipse.major_axis_px for candidate in candidates]))
    gap_ratio = float(np.max(gaps) / max(np.min(gaps), 1e-6))
    cross_span = float(np.max(cross) - np.min(cross))
    valid = bool(np.min(gaps) > 0.0 and gap_ratio <= 1.8 and cross_span <= 0.5 * median_diameter)
    return ordered, valid


def _order_triangular(candidates: list[_Candidate]) -> tuple[list[_Candidate], bool]:
    centers = np.asarray(
        [[candidate.ellipse.center_x, candidate.ellipse.center_y] for candidate in candidates],
        dtype=np.float64,
    )
    distances = []
    for first in range(3):
        for second in range(first + 1, 3):
            distances.append(float(np.linalg.norm(centers[first] - centers[second])))
    min_distance = min(distances)
    max_distance = max(distances)
    valid = min_distance > 0.0 and max_distance / min_distance <= 1.7
    top_index = int(np.argmin(centers[:, 1]))
    remaining = [index for index in range(3) if index != top_index]
    remaining.sort(key=lambda index: centers[index, 0])
    ordered = [candidates[top_index], candidates[remaining[0]], candidates[remaining[1]]]
    return ordered, bool(valid)


def _layout_axes(
    ordered_candidates: list[_Candidate],
    template: GeometryTemplate,
) -> tuple[np.ndarray, np.ndarray]:
    if template.expected_face_count != 3 or len(ordered_candidates) != 3:
        return (
            np.array([1.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
        )
    centers = np.asarray(
        [
            [candidate.ellipse.center_x, candidate.ellipse.center_y]
            for candidate in ordered_candidates
        ],
        dtype=np.float64,
    )
    if template.triple_layout is TripleLayout.VERTICAL:
        y_axis = centers[2] - centers[0]
    elif template.triple_layout is TripleLayout.TRIANGULAR:
        y_axis = 0.5 * (centers[1] + centers[2]) - centers[0]
    else:
        y_axis = np.array([0.0, 1.0], dtype=np.float64)
    y_axis /= max(float(np.linalg.norm(y_axis)), 1e-6)
    x_axis = np.array([y_axis[1], -y_axis[0]], dtype=np.float64)
    return x_axis, y_axis


def _ellipse_from_cv(ellipse: tuple) -> EllipseFit:
    (center_x, center_y), (diameter_x, diameter_y), angle = ellipse
    return EllipseFit(
        center_x=float(center_x),
        center_y=float(center_y),
        diameter_x=float(diameter_x),
        diameter_y=float(diameter_y),
        angle_deg=float(angle),
    )


def _ellipse_bounds(ellipse: EllipseFit) -> tuple[float, float, float, float]:
    angle = math.radians(ellipse.angle_deg)
    radius_x = ellipse.diameter_x / 2.0
    radius_y = ellipse.diameter_y / 2.0
    half_width = math.sqrt((radius_x * math.cos(angle)) ** 2 + (radius_y * math.sin(angle)) ** 2)
    half_height = math.sqrt((radius_x * math.sin(angle)) ** 2 + (radius_y * math.cos(angle)) ** 2)
    return (
        ellipse.center_x - half_width,
        ellipse.center_y - half_height,
        ellipse.center_x + half_width,
        ellipse.center_y + half_height,
    )


def _bounds_coverage(
    bounds: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> float:
    left, top, right, bottom = bounds
    width = max(0.0, right - left)
    height = max(0.0, bottom - top)
    area = width * height
    if area <= 0.0:
        return 0.0
    clipped_left = max(0.0, left)
    clipped_top = max(0.0, top)
    clipped_right = min(float(image_width - 1), right)
    clipped_bottom = min(float(image_height - 1), bottom)
    intersection = max(0.0, clipped_right - clipped_left) * max(0.0, clipped_bottom - clipped_top)
    return float(max(0.0, min(1.0, intersection / area)))


def _aggregate_bounds(faces: tuple[FaceGeometry, ...]) -> tuple[float, float, float, float] | None:
    if not faces:
        return None
    return (
        min(face.visible_bounds[0] for face in faces),
        min(face.visible_bounds[1] for face in faces),
        max(face.visible_bounds[2] for face in faces),
        max(face.visible_bounds[3] for face in faces),
    )


def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)
