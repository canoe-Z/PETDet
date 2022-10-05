# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_upsample_layer, xavier_init, MaxPool2d
import torch.nn.functional as F
from mmcv.cnn.bricks import build_plugin_layer
from mmcv.ops.carafe import CARAFEPack
from mmcv.runner import BaseModule, ModuleList

from ..builder import ROTATED_NECKS


@ROTATED_NECKS.register_module()
class LFCP(BaseModule):

    """Implementation of `Low-level Feature Concerned Pyramid (LFCP)`
    It constructs the low-level features in the top-down pathway of feature pyramid.
    It can reproduce the performance of TGRS 2022 paper
    LFG-Net: Low-level Feature Guided Network for Precise Ship Instance Segmentation in SAR Images
    Args:
        in_channels (list[int]): Number of channels for each input feature map.
        out_channels (int): Output channels of feature pyramids.
        num_outs (int): Number of output stages.
        start_level (int): Start level of feature pyramids.
            (Default: 0)
        end_level (int): End level of feature pyramids.
            (Default: -1 indicates the last level).
        norm_cfg (dict): Dictionary to construct and config norm layer.
        activate (str): Type of activation function in ConvModule
            (Default: None indicates w/o activation).
        upsample_cfg_b (dict): Dictionary to construct and config upsample layer in top-down path.
        att_cfg (dict): Dictionary to construct the attention layer.
        upsample_cfg (dict): Dictionary to construct and config upsample layer of P2.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg_b=dict(mode='nearest'),
                 att_cfg=dict(type='ContextBlock',
                              in_channels=256, ratio=1. / 4),
                 upsample_cfg=dict(
                     type='carafe',
                     up_kernel=5,
                     up_group=1,
                     encoder_kernel=3,
                     encoder_dilation=1),
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(LFCP, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_bias = norm_cfg is None
        self.upsample_cfg = upsample_cfg.copy()
        self.upsample = self.upsample_cfg.get('type')
        self.relu = nn.ReLU(inplace=False)
        self.upsample_cfg_b = upsample_cfg_b.copy()

        assert self.upsample in [
            'nearest', 'bilinear', 'deconv', 'pixel_shuffle', 'carafe', None
        ]
        if self.upsample in ['deconv', 'pixel_shuffle']:
            assert hasattr(
                self.upsample_cfg,
                'upsample_kernel') and self.upsample_cfg.upsample_kernel > 0
            self.upsample_kernel = self.upsample_cfg.pop('upsample_kernel')

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = ModuleList()
        self.fpn_convs = ModuleList()
        self.upsample_modules = ModuleList()
        upsample_cfg_ = self.upsample_cfg.copy()
        upsample_cfg_.update(channels=out_channels, scale_factor=2)
        upsample_module = build_upsample_layer(upsample_cfg_)
        self.upsample_modules.append(upsample_module)

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)

        for i in range(self.start_level, self.backbone_end_level+1):
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.fpn_convs.append(fpn_conv)

        self.y_conv = ConvModule(
            out_channels//4,
            out_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)
        self.down_conv = ConvModule(
            out_channels*2,
            out_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)
        self.fil_conv = ConvModule(
            out_channels,
            out_channels,
            3,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            inplace=False)
        self.fil_conv_1 = ConvModule(
            out_channels,
            out_channels,
            3,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=None,
            inplace=False)
        self.att_module = build_plugin_layer(att_cfg, '_att_module')[1]
        self.att_module_1 = build_plugin_layer(att_cfg, '_att_module_1')[1]

    def init_weights(self):
        """Initialize the weights of module."""
        super(LFCP, self).init_weights()
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                xavier_init(m, distribution='uniform')
        for m in self.modules():
            if isinstance(m, CARAFEPack):
                m.init_weights()

    def forward(self, inputs, y):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            if i <= self.backbone_end_level - self.start_level:
                input = inputs[min(i + self.start_level, len(inputs) - 1)]
            else:
                input = laterals[-1]
            lateral = lateral_conv(input)
            laterals.append(lateral)

        # build top-down path
        for i in range(len(laterals) - 2, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, **self.upsample_cfg_b)

        y_out = self.y_conv(y)
        upsample_out = self.upsample_modules[0](laterals[0])
        cat_feats = torch.cat((upsample_out, y_out), dim=1)
        down_feats = self.down_conv(cat_feats)
        out_feats = self.fil_conv(self.fil_conv_1(down_feats))
        att_feats = self.att_module(out_feats)
        att_feats = upsample_out+att_feats
        att_feats = self.att_module_1(att_feats)
        lowlevel_out = att_feats

        laterals.append(lowlevel_out)

        laterals_o = []
        for i in range(len(laterals)):
            if i == 0:
                laterals_o.append(laterals[4])
            else:
                laterals_o.append(laterals[i-1])

        # build outputs
        num_conv_outs = len(self.fpn_convs)
        outs = []
        for i in range(num_conv_outs):
            out = self.fpn_convs[i](laterals_o[i])
            outs.append(out)
        return tuple(outs)
