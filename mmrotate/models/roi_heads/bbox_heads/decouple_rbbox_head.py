# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_linear_layer

from ...builder import ROTATED_HEADS
from .convfc_rbbox_head import RotatedConvFCBBoxHead
from vit_pytorch import ViT


@ROTATED_HEADS.register_module()
class RotatedDecoupleBBoxHead(RotatedConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, use_vit=False, *args, **kwargs):
        super(RotatedDecoupleBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=0,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=4,
            num_reg_fcs=1,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        self.use_vit = use_vit
        if self.use_vit:
            cls_channels = self.num_classes + 1
            self.v = ViT(
                image_size=self.roi_feat_size,
                patch_size=1,
                num_classes=cls_channels,
                dim=256,
                depth=2,
                heads=2,
                channels=256,
                mlp_dim=1024,
                dropout=0.1,
                emb_dropout=0.1)

    def forward(self, x_cls, x_reg):
        """Forward function."""
        if self.use_vit:
            cls_score = self.v(x_cls) if self.with_cls else None
        else:
            for conv in self.cls_convs:
                x_cls = conv(x_cls)
            if x_cls.dim() > 2:
                if self.with_avg_pool:
                    x_cls = self.avg_pool(x_cls)
                x_cls = x_cls.flatten(1)
            for fc in self.cls_fcs:
                x_cls = self.relu(fc(x_cls))

            cls_score = self.fc_cls(x_cls) if self.with_cls else None
        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred
