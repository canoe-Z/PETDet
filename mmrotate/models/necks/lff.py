# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, xavier_init
import torch.nn.functional as F
from mmcv.runner import BaseModule
from ..builder import ROTATED_NECKS


@ROTATED_NECKS.register_module()
class LFF(BaseModule):
    def __init__(self,
                 num_ins=5,
                 feat_channels=256,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None):
        super(LFF, self).__init__(init_cfg)
        self.num_ins = num_ins
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.init_cfg = init_cfg

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = ConvModule(
            feat_channels * 2,
            feat_channels * 2,
            1,
            padding=0,
            stride=1,
            groups=1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)
        self.down_conv = ConvModule(
            feat_channels,
            feat_channels,
            1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)
        upsample_cfg = dict(mode='nearest')
        self.upsample_cfg = upsample_cfg.copy()
        self.gamma = nn.Parameter(torch.ones(
            (1, feat_channels, 1, 1)), requires_grad=True)

    def init_weights(self):
        """Initialize the weights of module."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        """Forward function."""
        inputs = list(inputs)
        
        for i in range(len(inputs[:-1])):
            prev_shape = inputs[i].shape[2:]
            feat = F.interpolate(
                inputs[i + 1], size=prev_shape, **self.upsample_cfg)
            feat = torch.cat([inputs[i], feat], dim=1)

            # Channel Attention
            weight = self.avgpool(feat)
            weight = self.channel_attention(weight)
            feat = feat * weight

            # Hadamard Product
            feat1, feat2 = torch.chunk(feat, 2, dim=1)
            feat = feat1 * feat2
            feat = self.down_conv(feat)

            inputs[i] = inputs[i] + self.gamma * feat

        return inputs[:-1]
