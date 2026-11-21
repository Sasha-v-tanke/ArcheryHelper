"""Deterministic variable-length decoding shared by validation and inference."""

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from neural_network.archery_ml.heatmap import grid_to_canonical


@dataclass(frozen=True)
class ImpactPrediction:
    x_norm: float
    y_norm: float
    confidence: float
    face_index: int = 0


def decode_impacts(
    heatmap_logits: torch.Tensor,
    offsets: torch.Tensor,
    *,
    confidence_threshold: float,
    nms_radius: int,
    face_index: int = 0,
) -> list[ImpactPrediction]:
    if heatmap_logits.ndim != 3 or heatmap_logits.shape[0] != 1:
        raise ValueError("heatmap must be [1,H,W]")
    if offsets.shape != (2, *heatmap_logits.shape[-2:]):
        raise ValueError("offset map must be [2,H,W]")
    if not 0 < confidence_threshold <= 1 or not math.isfinite(confidence_threshold):
        raise ValueError("invalid confidence threshold")
    if type(nms_radius) is not int or nms_radius < 0:
        raise ValueError("nms_radius must be a nonnegative integer")
    if face_index < 0:
        raise ValueError("face_index must be nonnegative")
    confidence = heatmap_logits.detach().float().sigmoid()[0]
    height, width = confidence.shape
    peak = F.max_pool2d(confidence[None, None], 2 * nms_radius + 1, stride=1,
                        padding=nms_radius)[0, 0]
    mask = (confidence >= confidence_threshold) & (confidence == peak)
    # Prevent a flat local maximum from producing a point in every pixel.
    if nms_radius:
        mask[:, 1:] &= confidence[:, 1:] != confidence[:, :-1]
        mask[1:, :] &= confidence[1:, :] != confidence[:-1, :]
    locations = torch.nonzero(mask, as_tuple=False).tolist()
    locations.sort(key=lambda row: (-float(confidence[row[0], row[1]]), row[0], row[1]))
    selected = []
    result = []
    for y, x in locations:
        if any((x - sx) ** 2 + (y - sy) ** 2 <= nms_radius**2 for sy, sx in selected):
            continue
        dx = float(offsets[0, y, x])
        dy = float(offsets[1, y, x])
        if not all(math.isfinite(v) and abs(v) <= 0.5 for v in (dx, dy)):
            continue
        cx, cy = grid_to_canonical(x + dx, y + dy, height, width)
        if not (-1 <= cx <= 1 and -1 <= cy <= 1):
            continue
        result.append(ImpactPrediction(cx, cy, float(confidence[y, x]), face_index))
        selected.append((y, x))
    return result
