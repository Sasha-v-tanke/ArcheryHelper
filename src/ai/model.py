import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from src.ai.config import OUTPUT_DIM


class ArcheryResNet(nn.Module):
    def __init__(self, output_dim=OUTPUT_DIM):
        super().__init__()
        self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, output_dim)

    def forward(self, x):
        return self.backbone(x)
