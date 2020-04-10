import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import weighted_loss
import numpy as np

@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


def cos_loss(pred, target, weight=None, reduction='none', avg_factor=1.):
    target = target * 2 * np.pi
    pred = pred * 2 * np.pi
    loss = torch.tensor(1.)-torch.cos(target - pred)
    if weight is not None:
        loss = loss * weight
    loss = loss.sum() / avg_factor
    return loss


@LOSSES.register_module
class SmoothL1Loss_360(nn.Module):

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0,
                 angle_loss_type='mse', angle_loss_weight=1.0):
        super(SmoothL1Loss_360, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.angle_loss_type = angle_loss_type
        self.angle_loss_weight = angle_loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred[:, :4],
            target[:, :4],
            weight[:, :4],
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        if self.angle_loss_type == 'mse':
            loss_angle = self.angle_loss_weight * mse_loss(
                pred[:, 4],
                target[:, 4],
                weight[:, 4],
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            loss_angle = self.angle_loss_weight * cos_loss(
                pred[:, 4],
                target[:, 4],
                weight[:, 4],
                reduction=reduction,
                avg_factor=avg_factor
            )
        return loss_bbox + loss_angle
