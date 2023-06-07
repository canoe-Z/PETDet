# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, xavier_init, normal_init
import torch.nn.functional as F
from mmcv.runner import BaseModule
from ..builder import ROTATED_NECKS


class ChannelAttention(BaseModule):
    def __init__(self,
                 feat_channels=256,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None):
        super(ChannelAttention, self).__init__(init_cfg)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.init_cfg = init_cfg

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = ConvModule(
            feat_channels,
            feat_channels,
            1,
            padding=0,
            stride=1,
            groups=1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)

    def init_weights(self):
        """Initialize the weights of module."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                normal_init(m, std=0.001)

    def forward(self, x):
        """Forward function."""
        weight = self.avgpool(x)
        weight = self.channel_attention(weight)
        x = x * weight
        return x


class ChannelInteractionModule(BaseModule):
    def __init__(self,
                 feat_channels=256,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None):
        super(ChannelInteractionModule, self).__init__(init_cfg)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.init_cfg = init_cfg

        self.ca1 = ChannelAttention(feat_channels)
        self.ca2 = ChannelAttention(feat_channels)

    def init_weights(self):
        """Initialize the weights of module."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                xavier_init(m, distribution='uniform')
        self.ca1.init_weights()
        self.ca2.init_weights()

    def forward(self, x1, x2):
        """Forward function."""
        x1_1, x1_2 = torch.chunk(x1, 2, dim=1)
        x2_1, x2_2 = torch.chunk(x1, 2, dim=1)
        x1 = torch.cat([x1_1, x2_2], dim=1)
        x2 = torch.cat([x2_1, x1_2], dim=1)
        x1 = self.ca1(x1)
        x2 = self.ca2(x2)

        return x1, x2


@ROTATED_NECKS.register_module()
class BCFNCA(BaseModule):
    def __init__(self,
                 num_ins=5,
                 feat_channels=256,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None):
        super(BCFNCA, self).__init__(init_cfg)
        self.num_ins = num_ins
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.init_cfg = init_cfg

        self.cim = ChannelInteractionModule(feat_channels)
        self.pw_conv = ConvModule(
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
        self.cim.init_weights()

    def forward(self, inputs):
        """Forward function."""
        inputs = list(inputs)

        for i in range(len(inputs[:-1])):
            prev_shape = inputs[i].shape[2:]
            upsampled_feat = F.interpolate(
                inputs[i + 1], size=prev_shape, **self.upsample_cfg)

            feat1, feat2 = self.cim(inputs[i], upsampled_feat)
            feat = feat1 * feat2
            feat = self.pw_conv(feat)

            inputs[i] = inputs[i] + self.gamma * feat

        return inputs[:-1]