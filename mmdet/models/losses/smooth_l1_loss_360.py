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
def smooth_l1_loss_for_angle(pred, target):
    # bad
    assert pred.size() == target.size() and target.numel() > 0
    diff1 = torch.abs(pred - target) - torch.tensor(np.pi)
    diff = torch.abs(diff1)
    delta = np.pi / 2
    loss = torch.where(diff < delta, 0.5 * diff * diff,
                       delta * (diff - 0.5 * delta))
    loss = torch.tensor(3*(np.pi**2)/8) - loss
    return loss


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def cosine_loss(pred, target):
    diff1 = torch.abs(pred - target)
    diff1 = torch.where(diff1 >= np.pi, 2 * np.pi - diff1, diff1)
    loss = torch.tensor(1.0) - torch.cos(diff1)
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
        elif self.angle_loss_type == 'smooth_l1_loss_for_angle':
            loss_angle = self.angle_loss_weight * smooth_l1_loss_for_angle(
                pred[:, 4],
                target[:, 4],
                weight[:, 4],
                reduction=reduction,
                avg_factor=avg_factor
            )
        elif self.angle_loss_type == 'cosine':
            loss_angle = self.angle_loss_weight * cosine_loss(
                pred[:, 4],
                target[:, 4],
                weight[:, 4],
                reduction=reduction,
                avg_factor=avg_factor
            )
        else:
            raise ValueError('Error angle loss type')

        return loss_bbox + loss_angle
