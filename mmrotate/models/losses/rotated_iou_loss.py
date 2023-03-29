# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
from mmdet.models.losses.utils import weighted_loss

from ..builder import ROTATED_LOSSES

try:
    from mmcv.ops import diff_iou_rotated_2d
    from mmcv.ops.diff_iou_rotated import box2corners, oriented_box_intersection_2d
except:  # noqa: E722
    diff_iou_rotated_2d = None

from .min_enclosing_box import smallest_bounding_box


@weighted_loss
def rotated_iou_loss(pred, target, linear=False, mode='log', eps=1e-6):
    """Rotated IoU loss.

    Computing the IoU loss between a set of predicted rbboxes and target
     rbboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x, y, h, w, angle),
            shape (n, 5).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 5).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
        eps (float): Eps to avoid log(0).
    Return:
        torch.Tensor: Loss tensor.
    """
    assert mode in ['linear', 'square', 'log']
    if linear:
        mode = 'linear'
        warnings.warn(
            'DeprecationWarning: Setting "linear=True" in '
            'poly_iou_loss is deprecated, please use "mode=`linear`" '
            'instead.')

    if diff_iou_rotated_2d is None:
        raise ImportError('Please install mmcv-full >= 1.5.0.')

    ious = diff_iou_rotated_2d(pred.unsqueeze(0), target.unsqueeze(0))
    ious = ious.squeeze(0).clamp(min=eps)

    if mode == 'linear':
        loss = 1 - ious
    elif mode == 'square':
        loss = 1 - ious**2
    elif mode == 'log':
        loss = -ious.log()
    else:
        raise NotImplementedError
    return loss


def diff_giou_rotated_2d(box1, box2):
    """Calculate differentiable iou of rotated 2d boxes.

    Args:
        box1 (Tensor): (B, N, 5) First box.
        box2 (Tensor): (B, N, 5) Second box.

    Returns:
        Tensor: (B, N) IoU.
    """
    corners1 = box2corners(box1)
    corners2 = box2corners(box2)
    intersection, _ = oriented_box_intersection_2d(corners1,
                                                   corners2)  # (B, N)
    area1 = box1[:, :, 2] * box1[:, :, 3]
    area2 = box2[:, :, 2] * box2[:, :, 3]
    union = area1 + area2 - intersection
    iou = intersection / union

    w, h = smallest_bounding_box(torch.cat([corners1, corners2], dim=-2))
    enclose = w*h
    giou = iou - (enclose - union) / enclose
    return giou


@weighted_loss
def rotated_giou_loss(pred, target, linear=False, mode='log', eps=1e-6):
    """Rotated IoU loss.

    Computing the IoU loss between a set of predicted rbboxes and target
     rbboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x, y, h, w, angle),
            shape (n, 5).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 5).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
        eps (float): Eps to avoid log(0).
    Return:
        torch.Tensor: Loss tensor.
    """
    assert mode in ['linear', 'square', 'log']
    if linear:
        mode = 'linear'
        warnings.warn(
            'DeprecationWarning: Setting "linear=True" in '
            'poly_iou_loss is deprecated, please use "mode=`linear`" '
            'instead.')

    if diff_giou_rotated_2d is None:
        raise ImportError('Please install mmcv-full >= 1.5.0.')

    gious = diff_giou_rotated_2d(pred.unsqueeze(0), target.unsqueeze(0))
    gious = gious.squeeze(0).clamp(min=eps)

    if mode == 'linear':
        loss = 1 - gious
    elif mode == 'square':
        loss = 1 - gious**2
    elif mode == 'log':
        loss = -gious.log()
    else:
        raise NotImplementedError
    return loss


@ROTATED_LOSSES.register_module()
class RotatedIoULoss(nn.Module):
    """RotatedIoULoss.

    Computing the IoU loss between a set of predicted rbboxes and
    target rbboxes.
    Args:
        linear (bool): If True, use linear scale of loss else determined
            by mode. Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(self,
                 linear=False,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0,
                 mode='log'):
        super(RotatedIoULoss, self).__init__()
        assert mode in ['linear', 'square', 'log']
        if linear:
            mode = 'linear'
            warnings.warn('DeprecationWarning: Setting "linear=True" in '
                          'IOULoss is deprecated, please use "mode=`linear`" '
                          'instead.')
        self.mode = mode
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 5) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * rotated_iou_loss(
            pred,
            target,
            weight,
            mode=self.mode,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@ROTATED_LOSSES.register_module()
class RotatedGIoULoss(nn.Module):
    """RotatedIoULoss.

    Computing the IoU loss between a set of predicted rbboxes and
    target rbboxes.
    Args:
        linear (bool): If True, use linear scale of loss else determined
            by mode. Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(self,
                 linear=False,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0,
                 mode='linear'):
        super(RotatedGIoULoss, self).__init__()
        assert mode in ['linear', 'square', 'log']
        if linear:
            mode = 'linear'
            warnings.warn('DeprecationWarning: Setting "linear=True" in '
                          'IOULoss is deprecated, please use "mode=`linear`" '
                          'instead.')
        self.mode = mode
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 5) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * rotated_giou_loss(
            pred,
            target,
            weight,
            mode=self.mode,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
