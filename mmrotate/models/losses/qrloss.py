# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch.nn as nn
import torch.nn.functional as F

from ..builder import ROTATED_LOSSES
from mmdet.models.losses.utils import weighted_loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def qr_focal_loss(pred, target, alpha=0.5, beta=2.0):
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert len(target) == 2, """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    label, iou = target

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()
    focal_weight = (1 - alpha) * pred_sigmoid.pow(beta)
    zerolabel = focal_weight.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none') * focal_weight

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = (label >= 0) & (label < bg_class_ind)
    pos_rcnn_pos = ((iou >= 0.4) & pos).nonzero().squeeze(1)
    pos_rcnn_neg = ((iou < 0.4) & pos).nonzero().squeeze(1)
    pos_rcnn_pos_label = label[pos_rcnn_pos].long()
    pos_rcnn_neg_label = label[pos_rcnn_neg].long()

    # positives are supervised by bbox quality (IoU) score
    # iou >= 0.4
    def func_iou(iou):
        return 1 - (1 - iou).pow(2.0)
    focal_weight = alpha * func_iou(iou[pos_rcnn_pos]) * \
        (1 - pred_sigmoid[pos_rcnn_pos, pos_rcnn_pos_label]).abs().pow(beta)
    onelabel = focal_weight.new_ones(
        pred[pos_rcnn_pos, pos_rcnn_pos_label].shape)
    loss[pos_rcnn_pos, pos_rcnn_pos_label] = F.binary_cross_entropy_with_logits(
        pred[pos_rcnn_pos, pos_rcnn_pos_label], onelabel,
        reduction='none') * focal_weight

    # iou < 0.4
    focal_weight = alpha * iou[pos_rcnn_neg].pow(2.0) * \
        (1 - pred_sigmoid[pos_rcnn_neg, pos_rcnn_neg_label]).abs().pow(beta)
    onelabel = focal_weight.new_ones(
        pred[pos_rcnn_neg, pos_rcnn_neg_label].shape)
    loss[pos_rcnn_neg, pos_rcnn_neg_label] = F.binary_cross_entropy_with_logits(
        pred[pos_rcnn_neg, pos_rcnn_neg_label], onelabel,
        reduction='none') * focal_weight

    loss = loss.sum(dim=1, keepdim=False)
    return loss


@ROTATED_LOSSES.register_module()
class QRFocalLoss(nn.Module):
    r"""Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
        activated (bool, optional): Whether the input is activated.
            If True, it means the input has been activated and can be
            treated as probabilities. Else, it should be treated as logits.
            Defaults to False.
    """

    def __init__(self,
                 use_sigmoid=True,
                 alpha=0.5,
                 beta=2.0,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False):
        super(QRFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * qr_focal_loss(
                pred,
                target,
                weight,
                alpha=self.alpha,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls
