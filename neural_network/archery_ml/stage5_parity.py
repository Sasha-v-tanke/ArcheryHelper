"""Golden input validation and deterministic cross-runtime comparison for Stage 5."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from neural_network.archery_ml.contracts import ModelMetadata
from neural_network.archery_ml.postprocess import ImpactPrediction, decode_impacts

# Frozen release gates. Changing these values requires code review, not a CLI flag.
LOGIT_MAX_ABS = 0.01
OFFSET_MAX_ABS = 0.01
POINT_NORM_DISTANCE = 0.005
POINT_CONFIDENCE_ABS = 0.01
MIN_GOLDEN = 20
MAX_GOLDEN = 50
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class GoldenSample:
    sample_id: str
    image: Path
    crop_sha256: str
    original_sha256: str
    face_index: int
    split: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_tensor(image: Image.Image, metadata: ModelMetadata) -> torch.Tensor:
    """Match Stage 4's RGB/255, per-channel normalization without resizing."""
    if image.mode != "RGB" or image.size != (metadata.input.width, metadata.input.height):
        raise ValueError("golden image must be an exact RGB canonical crop")
    rgb = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    mean = np.asarray(metadata.input.mean, dtype=np.float32)
    std = np.asarray(metadata.input.std, dtype=np.float32)
    if (mean.shape != (3,) or std.shape != (3,)
            or not np.isfinite(mean).all() or not np.isfinite(std).all()
            or np.any(std <= 0)):
        raise ValueError("invalid RGB normalization")
    return torch.from_numpy(((rgb - mean) / std).transpose(2, 0, 1).copy()).unsqueeze(0)


def read_golden_manifest(manifest: Path, metadata: ModelMetadata) -> list[GoldenSample]:
    document = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError("unsupported golden manifest schema")
    records = document.get("samples")
    if not isinstance(records, list) or not MIN_GOLDEN <= len(records) <= MAX_GOLDEN:
        raise ValueError("golden set requires 20 to 50 canonical face samples")

    root = manifest.resolve().parent
    samples: list[GoldenSample] = []
    ids: set[str] = set()
    crops: set[str] = set()
    faces: set[tuple[str, int]] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("golden sample must be an object")
        sample_id = record.get("sample_id")
        relative = record.get("image_path")
        crop_hash = record.get("crop_sha256")
        original_hash = record.get("original_sha256")
        face_index = record.get("face_index")
        split = record.get("split")
        if not isinstance(sample_id, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]{1,128}", sample_id):
            raise ValueError("invalid sample_id")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("golden image path must stay inside manifest directory")
        if not isinstance(crop_hash, str) or not SHA256_PATTERN.fullmatch(crop_hash):
            raise ValueError("invalid canonical crop SHA256")
        if not isinstance(original_hash, str) or not SHA256_PATTERN.fullmatch(original_hash):
            raise ValueError("invalid original photo SHA256")
        if type(face_index) is not int or not 0 <= face_index <= 2:
            raise ValueError("invalid canonical face index")
        if split not in ("val", "test"):
            raise ValueError("golden set must use reviewed validation/test crops, not train or mobile holdout")
        image = (root / relative).resolve(strict=True)
        if not image.is_relative_to(root) or not image.is_file() or image.suffix.lower() != ".png":
            raise ValueError("golden crop must be a local PNG file inside manifest directory")
        if sample_id in ids or crop_hash in crops or (original_hash, face_index) in faces:
            raise ValueError("duplicate golden sample ID, crop bytes, or original face identity")
        if sha256_file(image) != crop_hash:
            raise ValueError(f"golden crop checksum mismatch: {sample_id}")
        with Image.open(image) as pixels:
            if pixels.format != "PNG" or pixels.mode != "RGB" or pixels.size != (
                metadata.input.width, metadata.input.height
            ):
                raise ValueError(f"golden crop is not canonical RGB PNG: {sample_id}")
            pixels.verify()
        ids.add(sample_id)
        crops.add(crop_hash)
        faces.add((original_hash, face_index))
        samples.append(GoldenSample(sample_id, image, crop_hash, original_hash, face_index, split))
    return sorted(samples, key=lambda sample: sample.sample_id)


def _match_points(reference: list[ImpactPrediction], actual: list[ImpactPrediction]) -> tuple[float, float]:
    if len(reference) != len(actual):
        raise ValueError(f"postprocess parity count mismatch: eager={len(reference)} executorch={len(actual)}")

    # Bipartite matching tolerates changes of confidence ordering across runtimes.
    candidates: list[list[int]] = []
    for old in reference:
        possibilities = []
        for index, new in enumerate(actual):
            distance = math.hypot(old.x_norm - new.x_norm, old.y_norm - new.y_norm)
            if (old.face_index == new.face_index and distance <= POINT_NORM_DISTANCE
                    and abs(old.confidence - new.confidence) <= POINT_CONFIDENCE_ABS):
                possibilities.append(index)
        candidates.append(possibilities)
    matched: dict[int, int] = {}

    def assign(left: int, seen: set[int]) -> bool:
        for right in candidates[left]:
            if right in seen:
                continue
            seen.add(right)
            if right not in matched or assign(matched[right], seen):
                matched[right] = left
                return True
        return False

    for left in range(len(reference)):
        if not assign(left, set()):
            raise ValueError("postprocessed points exceed coordinate/confidence tolerance")
    xy = max((math.hypot(reference[left].x_norm - actual[right].x_norm,
                         reference[left].y_norm - actual[right].y_norm)
              for right, left in matched.items()), default=0.0)
    confidence = max((abs(reference[left].confidence - actual[right].confidence)
                      for right, left in matched.items()), default=0.0)
    return xy, confidence


def compare_outputs(
    eager: tuple[torch.Tensor, torch.Tensor],
    deployed: tuple[torch.Tensor, torch.Tensor],
    metadata: ModelMetadata,
    face_index: int,
) -> dict:
    expected_height = metadata.input.height // metadata.output.stride
    expected_width = metadata.input.width // metadata.output.stride
    shapes = ((1, 1, expected_height, expected_width), (1, 2, expected_height, expected_width))
    errors = []
    for name, a, b, shape, limit in zip(
        ("heatmap_logits", "offsets"), eager, deployed, shapes,
        (LOGIT_MAX_ABS, OFFSET_MAX_ABS), strict=True,
    ):
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise ValueError(f"{name} is not a tensor")
        if tuple(a.shape) != shape or tuple(b.shape) != shape:
            raise ValueError(f"{name} shape mismatch; expected {shape}")
        if a.dtype != torch.float32 or b.dtype != torch.float32:
            raise ValueError(f"{name} must be float32 in both runtimes")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f"{name} has NaN or infinity")
        error = float((a.cpu() - b.cpu()).abs().max())
        if error > limit:
            raise ValueError(f"{name} parity error {error:.8g} exceeds {limit}")
        errors.append(error)

    kwargs = dict(confidence_threshold=metadata.postprocess.confidence_threshold,
                  nms_radius=metadata.postprocess.nms_radius, face_index=face_index)
    eager_points = decode_impacts(eager[0][0].cpu(), eager[1][0].cpu(), **kwargs)
    deployed_points = decode_impacts(deployed[0][0].cpu(), deployed[1][0].cpu(), **kwargs)
    xy_error, confidence_error = _match_points(eager_points, deployed_points)
    return {"logits_max_abs": errors[0], "offsets_max_abs": errors[1],
            "points_xy_max_norm": xy_error, "points_confidence_max_abs": confidence_error,
            "point_count": len(eager_points),
            "expected_points": [vars(point) for point in eager_points]}
