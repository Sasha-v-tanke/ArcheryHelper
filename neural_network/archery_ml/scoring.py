from __future__ import annotations

import math
from dataclasses import dataclass

from neural_network.archery_ml.targets import TargetTemplate


@dataclass(frozen=True)
class ImpactPoint:
    x_norm: float
    y_norm: float
    face_index: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.x_norm) or not math.isfinite(self.y_norm):
            raise ValueError("impact coordinates must be finite")
        if self.face_index < 0:
            raise ValueError("face_index must be non-negative")

    @property
    def radius_norm(self) -> float:
        return math.hypot(self.x_norm, self.y_norm)


@dataclass(frozen=True)
class ScoreResult:
    final_candidate: int
    is_x: bool
    line_call: bool


def score(
    target_template: TargetTemplate,
    impact: ImpactPoint,
    arrow_diameter_mm: float | None = None,
    localization_error: float = 0.0,
) -> ScoreResult:
    if not math.isfinite(localization_error) or localization_error < 0:
        raise ValueError("localization_error must be a finite non-negative normalized radius")

    diameter = arrow_diameter_mm
    if diameter is None:
        diameter = target_template.config.arrow_diameter_mm
    if diameter is not None and (not math.isfinite(diameter) or diameter <= 0):
        raise ValueError("arrow_diameter_mm must be positive")

    shaft_radius_norm = 0.0 if diameter is None else diameter / target_template.config.face_diameter_mm
    center_radius = impact.radius_norm
    scoring_radius = max(0.0, center_radius - shaft_radius_norm)

    final_candidate = 0
    for boundary in target_template.ring_boundaries:
        if scoring_radius <= boundary.radius_norm:
            final_candidate = boundary.score
            break

    is_x = final_candidate == 10 and scoring_radius <= target_template.x_ring_radius_norm
    line_call = _is_line_call(
        target_template=target_template,
        center_radius=center_radius,
        shaft_radius_norm=shaft_radius_norm,
        localization_error=localization_error,
    )
    return ScoreResult(final_candidate, is_x, line_call)


def _is_line_call(
    target_template: TargetTemplate,
    center_radius: float,
    shaft_radius_norm: float,
    localization_error: float,
) -> bool:
    if localization_error == 0.0:
        return False

    lower = max(0.0, center_radius - localization_error - shaft_radius_norm)
    upper = max(0.0, center_radius + localization_error - shaft_radius_norm)
    boundaries = {boundary.radius_norm for boundary in target_template.ring_boundaries}
    boundaries.add(target_template.x_ring_radius_norm)
    return any(lower <= boundary <= upper for boundary in boundaries)
