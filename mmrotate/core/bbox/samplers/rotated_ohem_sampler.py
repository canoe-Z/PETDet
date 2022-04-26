# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..transforms import rbbox2roi
from mmdet.core.bbox.samplers.base_sampler import BaseSampler
from mmdet.core.bbox.samplers.sampling_result import SamplingResult

from ..builder import ROTATED_BBOX_SAMPLERS


@ROTATED_BBOX_SAMPLERS.register_module()
class ROHEMSampler(BaseSampler):
    r"""Online Hard Example Mining Sampler described in `Training Region-based
    Object Detectors with Online Hard Example Mining
    <https://arxiv.org/abs/1604.03540>`_.
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 context,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 loss_key='loss_cls',
                 **kwargs):
        super(ROHEMSampler, self).__init__(num, pos_fraction, neg_pos_ub,
                                          add_gt_as_proposals)
        self.context = context
        if not hasattr(self.context, 'num_stages'):
            self.bbox_head = self.context.bbox_head
        else:
            self.bbox_head = self.context.bbox_head[self.context.current_stage]

        self.loss_key = loss_key

    def hard_mining(self, inds, num_expected, bboxes, labels, feats):
        with torch.no_grad():
            rois = rbbox2roi([bboxes])
            if not hasattr(self.context, 'num_stages'):
                bbox_results = self.context._bbox_forward(feats, rois)
            else:
                bbox_results = self.context._bbox_forward(
                    self.context.current_stage, feats, rois)
            cls_score = bbox_results['cls_score']
            loss = self.bbox_head.loss(
                cls_score=cls_score,
                bbox_pred=None,
                rois=rois,
                labels=labels,
                label_weights=cls_score.new_ones(cls_score.size(0)),
                bbox_targets=None,
                bbox_weights=None,
                reduction_override='none')[self.loss_key]
            _, topk_loss_inds = loss.topk(num_expected)
        return inds[topk_loss_inds]

    def _sample_pos(self,
                    assign_result,
                    num_expected,
                    bboxes=None,
                    feats=None,
                    **kwargs):
        """Sample positive boxes.

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            num_expected (int): Number of expected positive samples
            bboxes (torch.Tensor, optional): Boxes. Defaults to None.
            feats (list[torch.Tensor], optional): Multi-level features.
                Defaults to None.

        Returns:
            torch.Tensor: Indices  of positive samples
        """
        # Sample some hard positive samples
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.hard_mining(pos_inds, num_expected, bboxes[pos_inds],
                                    assign_result.labels[pos_inds], feats)

    def _sample_neg(self,
                    assign_result,
                    num_expected,
                    bboxes=None,
                    feats=None,
                    **kwargs):
        """Sample negative boxes.

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            num_expected (int): Number of expected negative samples
            bboxes (torch.Tensor, optional): Boxes. Defaults to None.
            feats (list[torch.Tensor], optional): Multi-level features.
                Defaults to None.

        Returns:
            torch.Tensor: Indices  of negative samples
        """
        # Sample some hard negative samples
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            neg_labels = assign_result.labels.new_empty(
                neg_inds.size(0)).fill_(self.bbox_head.num_classes)
            return self.hard_mining(neg_inds, num_expected, bboxes[neg_inds],
                                    neg_labels, feats)

    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (torch.Tensor): Boxes to be sampled from.
            gt_bboxes (torch.Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.

        Example:
            >>> from mmdet.core.bbox import RandomSampler
            >>> from mmdet.core.bbox import AssignResult
            >>> from mmdet.core.bbox.demodata import ensure_rng, random_boxes
            >>> rng = ensure_rng(None)
            >>> assign_result = AssignResult.random(rng=rng)
            >>> bboxes = random_boxes(assign_result.num_preds, rng=rng)
            >>> gt_bboxes = random_boxes(assign_result.num_gts, rng=rng)
            >>> gt_labels = None
            >>> self = RandomSampler(num=32, pos_fraction=0.5, neg_pos_ub=-1,
            >>>                      add_gt_as_proposals=False)
            >>> self = self.sample(assign_result, bboxes, gt_bboxes, gt_labels)
        """
        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]

        bboxes = bboxes[:, :5]

        gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=torch.uint8)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError(
                    'gt_labels must be given when add_gt_as_proposals is True')
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        neg_inds = neg_inds.unique()

        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result
