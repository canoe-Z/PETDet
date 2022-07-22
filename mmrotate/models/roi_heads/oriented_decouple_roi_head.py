# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ROTATED_HEADS
from .oriented_standard_roi_head import OrientedStandardRoIHead


@ROTATED_HEADS.register_module()
class OrientedDecoupleHeadRoIHead(OrientedStandardRoIHead):
    """Oriented RoI head for Double Head RCNN.
    https://arxiv.org/abs/1904.06493
    """

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing time."""
        bbox_cls_feats, bbox_reg_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_cls_feats = self.shared_head(bbox_cls_feats)
            bbox_reg_feats = self.shared_head(bbox_reg_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_cls_feats, bbox_reg_feats)

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            bbox_feats=bbox_cls_feats)
        return bbox_results
