# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init, xavier_init
from mmcv.runner import BaseModule

from ..builder import ROTATED_NECKS


class ChannelAttention(nn.Module):
    def __init__(self,
                 feat_channels):
        super(ChannelAttention, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.attention = ConvModule(
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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)

    def forward(self, x):
        """Forward function."""
        weight = self.avgpool(x)
        weight = self.attention(weight)
        return x * weight


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

        self.sca = ChannelAttention(feat_channels * 2)
        self.ds_conv = ConvModule(
            feat_channels,
            feat_channels,
            kernel_size=3,
            padding=1,
            stride=2,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)
        self.down_conv = ConvModule(
            feat_channels * 2,
            feat_channels,
            1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)

    def init_weights(self):
        """Initialize the weights of module."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                xavier_init(m, distribution='uniform')
        self.sca.init_weights()

    def forward(self, inputs):
        inputs = list(inputs)
        """Forward function."""
        # build laterals
        # for i in range(len(inputs)):
        #     inputs[i] = self.layernorm[i](inputs[i])

        # for i in range(len(inputs[:-1])):
        #     prev_shape = inputs[i].shape[2:]
        #     feat = F.interpolate(
        #         inputs[i + 1], size=prev_shape, **self.upsample_cfg_b)
        #     feat = torch.cat([inputs[i], feat], dim=1)
        #     weight = self.avgpool(feat)
        #     weight = self.attention(weight)
        #     feat = feat * weight
        #     inputs[i] = inputs[i] + self.down_conv(feat)
        # print(range(len(inputs[1:]), 1, -1))
        for i in range(len(inputs[1:]), 0, -1):  # reversed
            # prev_shape = inputs[i].shape[2:]
            # feat = F.interpolate(
            #     inputs[i + 1], size=prev_shape, **self.upsample_cfg_b)
            # print(i)
            feat = self.ds_conv(inputs[i - 1])
            feat = torch.cat([inputs[i], feat], dim=1)
            feat = self.sca(feat)
            # weight = self.avgpool(feat)
            # weight = self.attention(weight)
            # feat = feat * weight
            inputs[i] = inputs[i] + self.down_conv(feat)

        return inputs[1:]
