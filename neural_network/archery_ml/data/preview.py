from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

from neural_network.archery_ml.contracts import DatasetSample


def render_preview(
    samples: Sequence[DatasetSample],
    output_dir: str | Path,
    limit: int = 20,
) -> tuple[Path, ...]:
    if limit < 0:
        raise ValueError("limit must be non-negative")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    index: list[dict] = []

    for position, sample in enumerate(samples[:limit]):
        with Image.open(sample.image_path) as source_image:
            image = source_image.convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        radius = max(3, round(min(width, height) * 0.008))

        for geometry in sample.geometry_annotations:
            points = [
                (round(x_fraction * width), round(y_fraction * height))
                for x_fraction, y_fraction in geometry.points
            ]
            if geometry.kind == "target_center":
                for x, y in points:
                    draw.line((x - radius * 2, y, x + radius * 2, y), fill="cyan", width=2)
                    draw.line((x, y - radius * 2, x, y + radius * 2), fill="cyan", width=2)
            elif len(points) >= 2:
                draw.line(points + [points[0]], fill="yellow", width=2)

        for annotation in sample.raw_annotations:
            x = round(annotation.x_fraction * width)
            y = round(annotation.y_fraction * height)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline="red",
                width=2,
            )

        for annotation in sample.annotations:
            x = round((annotation.x_norm + 1.0) * 0.5 * width)
            y = round((annotation.y_norm + 1.0) * 0.5 * height)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline="lime",
                width=2,
            )

        digest = hashlib.sha256(sample.id.encode("utf-8")).hexdigest()[:12]
        rendered_path = output / f"{position:04d}-{digest}.jpg"
        image.save(rendered_path, quality=95)
        rendered.append(rendered_path)
        index.append(
            {
                "sample_id": sample.id,
                "source_id": sample.source_id,
                "split": sample.split,
                "image_path": str(sample.image_path),
                "preview_path": str(rendered_path),
                "canonical_impacts": len(sample.annotations),
                "raw_impacts": len(sample.raw_annotations),
                "geometry_annotations": len(sample.geometry_annotations),
            }
        )

    (output / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return tuple(rendered)
