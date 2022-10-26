# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ROTATED_DETECTORS
from .oriented_rcnn import OrientedRCNN

@ROTATED_DETECTORS.register_module()
class OrientedRCNNLFCP(OrientedRCNN):
    """Introduce low-level features into Mask R-CNN."""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(OrientedRCNNLFCP, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x, y = self.backbone(img)
        if self.with_neck:
            x = self.neck(x, y)
        return x