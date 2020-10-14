#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/4/18 12:57
# @Author  : Zisha Zhong
# @Email   : zhongzisha@outlook.com
# @File    : rotated_iou_loss.py

import torch
import torch.nn as nn

from ..registry import LOSSES
from .utils import weighted_loss

import numpy as np


@weighted_loss
def rotated_iou_loss(pred, target, angle_loss_type='mse', angle_loss_weight=1.0, eps=1e-6):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (Tensor): Predicted bboxes of format (xc, yc, w, h, angle),
            shape (n, 5).
        target (Tensor): Corresponding gt bboxes, shape (xc, yc, w, h, angle).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    # d1 -> top, d2->right, d3->bottom, d4->left
    area_gt = (target[...,0] + target[...,2]) * (target[...,1] + target[...,3])
    area_pred = (pred[...,0] + pred[...,2]) * (pred[...,1] + pred[...,3])
    w_union = torch.min(target[...,1], pred[...,1]) + torch.min(target[...,3], pred[...,3])
    h_union = torch.min(target[...,0], pred[...,0]) + torch.min(target[...,2], pred[...,2])
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    ious = (area_intersect + 1.0) / (area_union + 1.0)
    ious = ious.clamp(min=eps)
    L_AABB = 1 - ious

    if angle_loss_type == 'v1':
        theta_pred = pred[..., 4].sigmoid()
        theta_gt = target[..., 4]
        beta = np.pi / 2
        diff = torch.abs(theta_pred - theta_gt) * 2 * np.pi - torch.tensor(
            np.pi)  # [-2pi ~ 2pi] --> [0, 2pi] --> [-pi, pi]
        L_theta = torch.where(diff < beta, 0.5 * diff * diff / beta,
                              diff - 0.5 * beta)
        L_theta = torch.tensor(3 * (np.pi ** 2) / 8) - L_theta
    elif angle_loss_type == 'v2':
        theta_pred = pred[...,4].sigmoid()  # 0~1
        theta_gt = target[...,4]    # 0~1
        L_theta = torch.tensor(1.) - torch.cosine_similarity(theta_pred, theta_gt, dim=1)
    elif angle_loss_type == 'v3':
        theta_pred = pred[..., 4].sigmoid()
        theta_gt = target[..., 4]
        diff = (theta_pred - theta_gt) * 2 * np.pi
        L_theta = torch.abs(torch.sin(diff / 2.))
    elif angle_loss_type == 'mse':
        L_theta = torch.pow(pred[...,4].sigmoid() - target[...,4], 2)

    loss = L_AABB + angle_loss_weight * L_theta
    return loss


@LOSSES.register_module
class RotatedIoULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0,
                 angle_loss_type='mse',
                 angle_loss_weight=1.0):
        super(RotatedIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.angle_loss_weight = angle_loss_weight
        self.angle_loss_type = angle_loss_type

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * rotated_iou_loss(
            pred,
            target,
            weight,
            angle_loss_type=self.angle_loss_type,
            angle_loss_weight=self.angle_loss_weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
