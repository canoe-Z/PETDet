import torch
import torch.nn as nn
import torch.nn.functional as F
from fightingcv_attention.attention.ResidualAttention import ResidualAttention
from mmdet.models.dense_heads.tood_head import TaskDecomposition
from vit_pytorch import SimpleViT
from ...builder import ROTATED_HEADS
from .convfc_rbbox_head import RotatedConvFCBBoxHead
from mmcv.cnn.bricks import build_plugin_layer
from mmcv.runner import BaseModule
from mmcv.cnn import xavier_init,constant_init


class BilinearPooling(BaseModule):
    def __init__(self, in_channels, num_class):
        super().__init__()
        self.fc = nn.Linear(in_channels, num_class)

    def init_weights(self):
        xavier_init(self.fc,0,0)

    def forward(self, x):
        batch_size = x.size(0)
        channel_size = x.size(1)
        feature_size = x.size(2) * x.size(3)
        x = x.view(batch_size, channel_size, feature_size)
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / feature_size

        x = x.view(batch_size, -1)
        x = torch.sqrt(x + 1e-5)

        x = torch.nn.functional.normalize(x)

        x = self.fc(x)

        return x


@ROTATED_HEADS.register_module()
class ExperimentBBoxHead(RotatedConvFCBBoxHead):
    def __init__(self,
                 conv_cfg=None,
                 norm_cfg=None,
                 fc_out_channels=1024,
                 use_vit=True,
                 use_ra=False,
                 use_bilinear_pooling=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 *args,
                 **kwargs):
        super(ExperimentBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=0,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=4,
            num_reg_fcs=1,
            fc_out_channels=fc_out_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            loss_cls=loss_cls,
            *args,
            **kwargs)
        self.use_vit = use_vit
        self.use_ra = use_ra
        self.use_bilinear_pooling = use_bilinear_pooling
        cls_channels = self.num_classes + 1
        if self.use_vit:
            self.v = SimpleViT(
                image_size=self.roi_feat_size,
                patch_size=1,
                num_classes=cls_channels,
                dim=512,
                depth=2,
                heads=2,
                channels=256,
                mlp_dim=1024)
        if self.use_ra:
            self.ra = ResidualAttention(
                channel=256, num_class=cls_channels, la=0.2)

        # num_taps = self.num_shared_convs
        # self.cls_decomp = TaskDecomposition(self.in_channels,
        #                                     num_taps,
        #                                     num_taps * 8,
        #                                     self.conv_cfg, self.norm_cfg)
        # self.reg_decomp = TaskDecomposition(self.in_channels,
        #                                     num_taps,
        #                                     num_taps * 8,
        #                                     self.conv_cfg, self.norm_cfg)

        att_cfg = dict(type='ContextBlock',
                       in_channels=256, ratio=1. / 4)
        self.cls_att = build_plugin_layer(att_cfg, '_att_module')[1]
        self.reg_att = build_plugin_layer(att_cfg, '_att_module_1')[1]
        self.bilinear_pooling = BilinearPooling(
            in_channels=65536, num_class=cls_channels)

    def forward(self, x):
        # extract task interactive features
        # inter_feats = []
        # inter_feats.append(x)
        # for conv in self.shared_convs:
        #     x = conv(x)
        #     inter_feats.append(x)
        # print(inter_feats[0].shape)

        # feat = torch.cat(inter_feats, 1)
        # print(feat.shape)

        # if self.num_shared_fcs > 0:
        #     if self.with_avg_pool:
        #         x = self.avg_pool(x)

        #     x = x.flatten(1)

        #     for fc in self.shared_fcs:
        #         x = self.relu(fc(x))
        # separate branches
        # x_cls = x
        # x_reg = x
        # task decomposition
        # avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        # print(avg_feat.shape)
        x_cls = self.cls_att(x)
        #x_cls = self.bilinear_pooling(x_cls)
        # print(x_cls.shape)
        # assert(1==2)
        x_reg = self.reg_att(x)

        """Forward function."""
        if self.use_vit:
            cls_score = self.v(x_cls) if self.with_cls else None
        elif self.use_ra:
            cls_score = self.ra(x_cls) if self.with_cls else None
        elif self.use_bilinear_pooling:
            cls_score = self.bilinear_pooling(x_cls) if self.with_cls else None
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
