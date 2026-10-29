"""Deterministic rendered targets with independent projective ground truth."""

from dataclasses import dataclass

import cv2
import numpy as np

from neural_network.archery_ml.targets import (
    TargetConfig,
    TargetFormat,
    TenRingMode,
    TripleLayout,
)
from neural_network.archery_ml.geometry.geometry_math import project


@dataclass(frozen=True)
class SyntheticScene:
    image: np.ndarray
    config: TargetConfig
    norm_to_image: tuple
    scenario: str
    seed: int
    expected_reject: bool = False


def render_scene(
    scenario="clean",
    zone=5,
    layout=None,
    seed=42,
    size=800,
    occlusion=0.0,
    perspective=0.0,
    rotation_deg=0.0,
    scale_factor=1.0,
):
    """Truth is the generating homography; no detector is used to derive labels."""
    config = TargetConfig(
        TargetFormat.SINGLE if layout is None else TargetFormat.TRIPLE,
        zone,
        TenRingMode.RECURVE,
        40.0,
        layout,
    )
    centers = (
        [(400, 400)]
        if layout is None
        else (
            [(400, 160), (400, 400), (400, 640)]
            if layout is TripleLayout.VERTICAL
            else [(400, 180), (260, 520), (540, 520)]
        )
    )
    nominal = 290 if layout is None else 160
    nominal *= scale_factor
    scale = size / 800
    yy, xx = np.mgrid[:size, :size]
    image = np.full((size, size, 3), 118, np.uint8)
    matrices = []
    camera = np.array(
        [
            [1.0, 0.04, -12.0],
            [0.02, 1.0, -12.0],
            [perspective / 800, -perspective / 1200, 1.0],
        ]
    )
    if rotation_deg:
        rotation = np.vstack(
            (cv2.getRotationMatrix2D((400, 400), rotation_deg, 1.0), [0.0, 0.0, 1.0])
        )
        camera = camera @ rotation
    for index, (cx, cy) in enumerate(centers):
        if scenario == "clipped":
            cx = 70
        matrix = (
            np.diag([scale, scale, 1.0])
            @ camera
            @ np.array([[nominal, 0.0, cx], [0.0, nominal, cy], [0.0, 0.0, 1.0]])
        )
        matrices.append(matrix)
        inverse = np.linalg.inv(matrix)
        denominator = inverse[2, 0] * xx + inverse[2, 1] * yy + inverse[2, 2]
        x = (inverse[0, 0] * xx + inverse[0, 1] * yy + inverse[0, 2]) / denominator
        y = (inverse[1, 0] * xx + inverse[1, 1] * yy + inverse[1, 2]) / denominator
        radius = np.hypot(x, y)
        visible = (11 - zone) * 0.1
        for r, color in (
            (1.0, (240, 240, 240)),
            (0.8, (24, 24, 24)),
            (0.6, (35, 135, 220)),
            (0.4, (220, 35, 45)),
            (0.2, (245, 215, 30)),
        ):
            image[radius <= min(r, visible)] = color
        if scenario in ("center_destroyed", "arrows", "damaged_ring"):
            image[radius < 0.16] = (85, 75, 65)
        if scenario == "missing_yellow":
            image[radius < 0.2] = (118, 118, 118)
        if scenario == "damaged_ring":
            image[(radius > 0.34) & (radius < 0.43) & (x > 0)] = (105, 100, 95)
        if occlusion:
            angle = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)
            image[(radius <= visible) & (angle < occlusion * 2 * np.pi)] = (95, 90, 85)
        if scenario == "arrows":
            for start, end in (
                ((-0.65, -0.15), (0.6, 0.12)),
                ((-0.2, -0.65), (0.15, 0.6)),
            ):
                points = project([start, end], matrix).round().astype(int)
                cv2.line(
                    image,
                    tuple(points[0]),
                    tuple(points[1]),
                    (25, 28, 25),
                    max(2, round(7 * scale)),
                )
    if scenario == "lighting":
        gain = np.linspace(0.55, 1.05, size)[None, :, None]
        image = np.clip(image * gain * np.array([1.05, 0.94, 0.9]), 0, 255).astype(
            np.uint8
        )
    if scenario == "sensor_noise":
        noise = np.random.default_rng(seed).normal(0.0, 2.0, image.shape)
        image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
    if scenario == "blur_jpeg":
        image = cv2.GaussianBlur(image, (7, 7), 1.5)
        _, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 65],
        )
        image = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    reject = scenario in ("blank", "false_colors", "insufficient")
    if scenario == "blank":
        image[:] = 118
    elif scenario == "false_colors":
        image[:] = 118
        for x, color in (
            (100, (35, 135, 220)),
            (350, (220, 35, 45)),
            (600, (245, 215, 30)),
        ):
            cv2.rectangle(
                image,
                (round(x * scale), round(250 * scale)),
                (round((x + 80) * scale), round(500 * scale)),
                color,
                -1,
            )
    elif scenario == "insufficient":
        image[: round(size * 0.55)] = 118
        image[:, : round(size * 0.55)] = 118
    return SyntheticScene(image, config, tuple(matrices), scenario, seed, reject)


def scenarios():
    for zone in (1, 5, 6):
        for layout in (None, TripleLayout.VERTICAL, TripleLayout.TRIANGULAR):
            yield render_scene(zone=zone, layout=layout)
    for name in (
        "center_destroyed",
        "arrows",
        "damaged_ring",
        "missing_yellow",
        "lighting",
        "blur_jpeg",
        "clipped",
        "blank",
        "false_colors",
        "insufficient",
    ):
        yield render_scene(name, perspective=0.15)
    for fraction in (0.1, 0.25, 0.4, 0.6):
        yield render_scene(
            "occlusion",
            occlusion=fraction,
            perspective=0.15,
            seed=round(fraction * 100),
        )
    for perspective in (-0.25, -0.1, 0.1, 0.25):
        yield render_scene(
            "perspective", perspective=perspective, seed=round((perspective + 1) * 100)
        )
