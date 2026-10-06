from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TargetFormat(str, Enum):
    SINGLE = "SINGLE"
    TRIPLE = "TRIPLE"


class TripleLayout(str, Enum):
    VERTICAL = "VERTICAL"
    TRIANGULAR = "TRIANGULAR"


class TenRingMode(str, Enum):
    RECURVE = "RECURVE"
    COMPOUND = "COMPOUND"


@dataclass(frozen=True)
class TargetConfig:
    format: TargetFormat
    minimum_scoring_zone: int
    ten_ring_mode: TenRingMode
    face_diameter_mm: int
    triple_layout: TripleLayout | None = None
    arrow_diameter_mm: float | None = None
    expected_arrows_per_series: int | None = None

    def __post_init__(self) -> None:
        if self.minimum_scoring_zone not in (1, 5, 6):
            raise ValueError("minimum_scoring_zone must be 1, 5, or 6")
        if self.format is TargetFormat.SINGLE and self.triple_layout is not None:
            raise ValueError("single target must not define triple_layout")
        if self.format is TargetFormat.TRIPLE and self.triple_layout is None:
            raise ValueError("triple target requires triple_layout")
        if self.face_diameter_mm <= 0:
            raise ValueError("face_diameter_mm must be positive")
        if self.arrow_diameter_mm is not None and self.arrow_diameter_mm <= 0:
            raise ValueError("arrow_diameter_mm must be positive")
        if self.expected_arrows_per_series is not None and self.expected_arrows_per_series <= 0:
            raise ValueError("expected_arrows_per_series must be positive")


@dataclass(frozen=True)
class RingBoundary:
    score: int
    radius_norm: float


@dataclass(frozen=True)
class TargetTemplate:
    config: TargetConfig
    ring_boundaries: tuple[RingBoundary, ...]
    x_ring_radius_norm: float

    @property
    def visible_radius_norm(self) -> float:
        return self.ring_boundaries[-1].radius_norm

    @staticmethod
    def from_config(config: TargetConfig) -> "TargetTemplate":
        boundaries = []
        for score in range(10, config.minimum_scoring_zone - 1, -1):
            if score == 10 and config.ten_ring_mode is TenRingMode.COMPOUND:
                radius = 0.05
            else:
                radius = (11 - score) * 0.1
            boundaries.append(RingBoundary(score, radius))
        return TargetTemplate(
            config=config,
            ring_boundaries=tuple(boundaries),
            x_ring_radius_norm=0.05,
        )
