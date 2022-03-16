from .oriented_standard_roi_head import OrientedStandardRoIHead
from ..builder import ROTATED_HEADS


@ROTATED_HEADS.register_module()
class BatchOrientedRoIHeadFPN(OrientedStandardRoIHead):

    def _bbox_forward(self, x, rois):
        feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        bbox_feats = feats[0]
        fpn_feats = feats[1]
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        BS = x[0].size(0)
        #print(bbox_feats.shape)
        num_rois, channels, roi_h, roi_w = bbox_feats.shape
        bbox_feats = bbox_feats.reshape((BS, -1, channels, roi_h, roi_w))
        cls_score, bbox_pred = self.bbox_head(bbox_feats, fpn_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results
