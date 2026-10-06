from dataclasses import dataclass

from neural_network.archery_ml.targets import TargetConfig, TargetFormat, TripleLayout


CANONICAL_SIZE = 512
CANONICAL_CENTER_PX = (CANONICAL_SIZE - 1) / 2.0
CANONICAL_RADIUS_PX = CANONICAL_SIZE / 2.0


@dataclass(frozen=True)
class GeometryTemplate:
    expected_face_count: int
    visible_radius_norm: float
    colored_outer_radius_norm: float
    ring_outer_radii: tuple[tuple[str, float], ...]
    triple_layout: TripleLayout | None


def geometry_template(config: TargetConfig) -> GeometryTemplate:
    visible_radius = (11 - config.minimum_scoring_zone) * 0.1
    colored_radius = min(0.6, visible_radius)
    return GeometryTemplate(
        expected_face_count=1 if config.format is TargetFormat.SINGLE else 3,
        visible_radius_norm=visible_radius,
        colored_outer_radius_norm=colored_radius,
        ring_outer_radii=(
            ("yellow", 0.2),
            ("red", 0.4),
            ("blue", colored_radius),
        ),
        triple_layout=config.triple_layout,
    )
