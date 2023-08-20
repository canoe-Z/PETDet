# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import ROTATED_LOSSES
from mmdet.models.losses.utils import weight_reduce_loss


def softmax_focal_loss(pred,
                       label,
                       alpha=0.25,
                       gamma=2.0,
                       weight=None,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None,
                       ignore_index=-100,
                       avg_non_ignore=False):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.

    Returns:
        torch.Tensor: The calculated loss
    """
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index
    # element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)
    bg_class_ind = pred.size(1) - 1
    bg_mask = label == bg_class_ind
    fg_mask = (label >= 0) & (label < bg_class_ind)
    pt = torch.exp(-loss)

    # average loss over non-ignored elements
    # pytorch's official cross_entropy average loss over non-ignored elements
    # refer to https://github.com/pytorch/pytorch/blob/56b43f4fec1f76953f15a627694d4bba34588969/torch/nn/functional.py#L2660  # noqa
    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = label.numel() - (label == ignore_index).sum().item()

    # apply weights and do the reduction
    if weight is None:
        weight = torch.ones_like(pred)
    weight[bg_mask] = weight[bg_mask] * (1-alpha)
    weight[fg_mask] = weight[fg_mask] * alpha
    focal_weight = weight * (1 - pt).pow(gamma)
    loss = weight_reduce_loss(
        loss, weight=focal_weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@ROTATED_LOSSES.register_module()
class SoftmaxFocalLoss(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2.0,
                 use_sigmoid=False,
                 reduction='mean',
                 class_weight=None,
                 ignore_index=None,
                 loss_weight=1.0,
                 avg_non_ignore=False):
        """SoftmaxFocalLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            ignore_index (int | None): The label index to be ignored.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
            avg_non_ignore (bool): The flag decides to whether the loss is
                only averaged over non-ignored targets. Default: False.
        """
        super(SoftmaxFocalLoss, self).__init__()
        assert use_sigmoid is False
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore
        if ((ignore_index is not None) and not self.avg_non_ignore
                and self.reduction == 'mean'):
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')

        self.cls_criterion = softmax_focal_loss

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss. Options are "none", "mean" and "sum".
            ignore_index (int | None): The label index to be ignored.
                If not None, it will override the default value. Default: None.
        Returns:
            torch.Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if ignore_index is None:
            ignore_index = self.ignore_index

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None

        loss_cls = self.loss_weight * softmax_focal_loss(
            cls_score,
            label,
            alpha=self.alpha,
            gamma=self.gamma,
            weight=weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index,
            avg_non_ignore=self.avg_non_ignore,
            **kwargs)
        return loss_cls
