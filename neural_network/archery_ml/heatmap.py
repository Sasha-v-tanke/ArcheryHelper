"""Shared canonical coordinate/grid transform and CenterNet-style training targets."""

import math
from collections.abc import Sequence

import torch
from torch.nn import functional as F


def canonical_to_grid(x_norm: float, y_norm: float, height: int, width: int) -> tuple[float, float]:
    if height < 2 or width < 2:
        raise ValueError("grid dimensions must be at least two")
    if not all(math.isfinite(v) and -1 <= v <= 1 for v in (x_norm, y_norm)):
        raise ValueError("canonical point must be finite and within [-1,1]")
    return (x_norm + 1) * (width - 1) / 2, (y_norm + 1) * (height - 1) / 2


def grid_to_canonical(x: float, y: float, height: int, width: int) -> tuple[float, float]:
    return 2 * x / (width - 1) - 1, 2 * y / (height - 1) - 1


def encode_impacts(
    points: Sequence[tuple[float, float]],
    height: int,
    width: int,
    *,
    sigma: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Target heatmap, signed fractional offsets and center-only mask.

    Two impacts whose nearest grid centers coincide cannot be supervised by
    a single-vector offset head; explicitly reject rather than silently drop.
    """
    if sigma <= 0 or not math.isfinite(sigma):
        raise ValueError("sigma must be finite and positive")
    heatmap = torch.zeros((1, height, width), dtype=torch.float32)
    offsets = torch.zeros((2, height, width), dtype=torch.float32)
    positive = torch.zeros((1, height, width), dtype=torch.bool)
    yy = torch.arange(height, dtype=torch.float32).view(height, 1)
    xx = torch.arange(width, dtype=torch.float32).view(1, width)
    for x_norm, y_norm in points:
        x, y = canonical_to_grid(x_norm, y_norm, height, width)
        ix, iy = min(width - 1, int(math.floor(x + 0.5))), min(height - 1, int(math.floor(y + 0.5)))
        if positive[0, iy, ix]:
            raise ValueError("distinct impacts occupy the same output cell")
        positive[0, iy, ix] = True
        offsets[:, iy, ix] = torch.tensor((x - ix, y - iy))
        heatmap[0] = torch.maximum(
            heatmap[0], torch.exp(-((xx - ix) ** 2 + (yy - iy) ** 2) / (2 * sigma**2))
        )
    return heatmap, offsets, positive


def impact_loss(
    prediction: tuple[torch.Tensor, torch.Tensor],
    target: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    offset_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits, predicted_offsets = prediction
    truth_heatmap, truth_offsets, positive = target
    if logits.shape != truth_heatmap.shape or predicted_offsets.shape != truth_offsets.shape:
        raise ValueError("prediction and target shapes differ")
    if positive.shape != truth_heatmap.shape:
        raise ValueError("positive mask shape differs")
    probs = logits.sigmoid().clamp(1e-6, 1 - 1e-6)
    pos = positive
    neg = ~pos
    positive_loss = -torch.log(probs) * (1 - probs) ** 2 * pos
    negative_loss = -torch.log(1 - probs) * probs**2 * (1 - truth_heatmap) ** 4 * neg
    count = pos.sum().clamp(min=1)
    heatmap_loss = (positive_loss.sum() + negative_loss.sum()) / count
    if pos.any():
        offset_loss = F.smooth_l1_loss(predicted_offsets[pos.expand_as(predicted_offsets)],
                                       truth_offsets[pos.expand_as(truth_offsets)], reduction="sum") / (pos.sum() * 2)
    else:
        offset_loss = predicted_offsets.sum() * 0.0
    total = heatmap_loss + offset_weight * offset_loss
    return total, {"heatmap": heatmap_loss.detach(), "offset": offset_loss.detach()}
