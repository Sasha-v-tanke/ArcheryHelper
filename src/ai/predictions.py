from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class DetectedShot:
    confidence: float
    radius_norm: float
    angle_deg: float


def decode_predictions(
    values: Sequence[float],
    max_shots: int,
    confidence_threshold: float,
) -> list[DetectedShot]:
    expected_size = max_shots * 3
    if len(values) < expected_size:
        raise ValueError(f"expected at least {expected_size} output values, got {len(values)}")

    shots = []
    for shot_index in range(max_shots):
        offset = shot_index * 3
        confidence = float(values[offset])
        radius = float(values[offset + 1])
        angle = float(values[offset + 2])
        if confidence >= confidence_threshold and 0.0 <= radius <= 1.0:
            shots.append(DetectedShot(confidence, radius, angle))
    return shots
