import torch
import torch.nn as nn

from src.ai.config import SHOT


class ArrowCriterion(nn.Module):
    def __init__(self, alpha=1.0, beta=10.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.reg = nn.SmoothL1Loss(reduction='mean')
        self.alpha = alpha
        self.beta = beta

    def forward(self, preds, targets):
        B, D = preds.shape
        N = D // 3

        preds = preds.view(B, N, 3)
        targets = targets.view(B, N, 3)

        p_pred = preds[..., 0]
        xy_pred = preds[..., 1:3]

        p_true = targets[..., 0]
        xy_true = targets[..., 1:3]

        loss_cls = self.bce(p_pred, p_true)

        mask = (p_true > 0.5 * SHOT)
        if mask.sum() > 0:
            loss_reg = self.reg(xy_pred[mask], xy_true[mask])
        else:
            loss_reg = torch.tensor(0.0, device=preds.device)

        return self.alpha * loss_cls + self.beta * loss_reg
