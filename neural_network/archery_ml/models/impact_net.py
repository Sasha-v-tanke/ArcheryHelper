"""Unordered impact detector: per-cell heatmap logits and continuous x/y offsets."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class ImpactPointNet(nn.Module):
    """MobileNetV3-Large + lightweight top-down FPN, output stride two.

    Returns (heatmap_logits[B,1,H/2,W/2], offsets[B,2,H/2,W/2]); offsets
    are measured in output grid cells, not pixels or normalized target units.
    The feature indices follow torchvision MobileNetV3-Large 0.23.x.
    """

    def __init__(self, *, fpn_channels: int = 48, pretrained: bool = False) -> None:
        super().__init__()
        if fpn_channels <= 0:
            raise ValueError("fpn_channels must be positive")
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = mobilenet_v3_large(weights=weights).features
        self.lateral = nn.ModuleList(
            nn.Conv2d(channels, fpn_channels, kernel_size=1)
            for channels in (16, 24, 40, 112, 960)
        )
        self.refine = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, groups=fpn_channels, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.Hardswish(),
                nn.Conv2d(fpn_channels, fpn_channels, 1),
            )
            for _ in range(4)
        )
        self.heatmap_head = nn.Conv2d(fpn_channels, 1, kernel_size=1)
        self.offset_head = nn.Conv2d(fpn_channels, 2, kernel_size=1)
        nn.init.constant_(self.heatmap_head.bias, -2.19)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("expected [B,3,H,W]")
        if image.shape[-2] % 32 or image.shape[-1] % 32:
            raise ValueError("image dimensions must be multiples of 32")
        features = []
        x = image
        for index, layer in enumerate(self.backbone):
            x = layer(x)
            if index in (1, 3, 6, 12, 16):
                features.append(x)
        pyramid = self.lateral[-1](features[-1])
        for index in range(3, -1, -1):
            pyramid = self.refine[index](
                F.interpolate(pyramid, size=features[index].shape[-2:], mode="nearest")
                + self.lateral[index](features[index])
            )
        return self.heatmap_head(pyramid), self.offset_head(pyramid)
