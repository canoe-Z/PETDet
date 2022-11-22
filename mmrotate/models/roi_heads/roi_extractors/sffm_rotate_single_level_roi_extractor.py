# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv import ops
from mmcv.runner import force_fp32

from ...builder import ROTATED_ROI_EXTRACTORS
from ...utils.attention.CBAM import CBAMBlock
from ...utils.attention.CoordAttention import CoordAtt
from ...utils.attention.SEAttention import SEAttention
from .rotate_single_level_roi_extractor import RotatedSingleRoIExtractor


@ROTATED_ROI_EXTRACTORS.register_module()
class SFFMRotatedSingleRoIExtractor(RotatedSingleRoIExtractor):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (List[int]): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56,
                 current_feat='current',
                 fusion_feat='lower',
                 aggregation='sum',
                 post_process=None,
                 init_cfg=None):
        super(SFFMRotatedSingleRoIExtractor,
              self).__init__(roi_layer, out_channels, featmap_strides,
                             init_cfg)
        self.finest_scale = finest_scale
        self.current_feat = current_feat
        self.fusion_feat = fusion_feat
        self.aggregation = aggregation
        self.post_process = post_process
        if self.post_process == 'CA':
            self.attention = CoordAtt(
                self.out_channels, self.out_channels, reduction=32)
        elif self.post_process == 'CBAM':
            self.attention = CBAMBlock(self.out_channels, reduction=16,
                                       kernel_size=7)
        elif self.post_process == 'SE':
            self.attention = SEAttention(self.out_channels, reduction=8)
        self.fp16_enabled = False

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function.

        Args:
            feats (torch.Tensor): Input features.
            rois (torch.Tensor): Input RoIs, shape (k, 5).
            scale_factor (float): Scale factor that RoI will be multiplied by.

        Returns:
            torch.Tensor: Scaled RoI features.
        """
        from mmrotate import digit_version, mmcv_version
        if isinstance(self.roi_layers[0], ops.RiRoIAlignRotated
                      ) or mmcv_version == digit_version('1.4.5'):
            out_size = nn.modules.utils._pair(self.roi_layers[0].out_size)
        else:
            out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        expand_dims = (-1, self.out_channels * out_size[0] * out_size[1])
        if torch.onnx.is_in_onnx_export():
            # Work around to export mask-rcnn to onnx
            roi_feats = rois[:, :1].clone().detach()
            roi_feats = roi_feats.expand(*expand_dims)
            roi_feats = roi_feats.reshape(-1, self.out_channels, *out_size)
            roi_feats = roi_feats * 0
        else:
            roi_feats = feats[0].new_zeros(
                rois.size(0), self.out_channels, *out_size)
        # TODO: remove this when parrots supports
        if torch.__version__ == 'parrots':
            roi_feats.requires_grad = True

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            mask = target_lvls == i
            inds = mask.nonzero(as_tuple=False).squeeze(1)
            if inds.numel() > 0:
                rois_ = rois[inds]
                if self.current_feat == 'current':
                    roi_feats_c = self.roi_layers[i](feats[i], rois_)
                elif self.current_feat == 'lower':
                    roi_feats_c = self.roi_layers[i-1](feats[i-1], rois_)

                if self.fusion_feat is not None:
                    if self.fusion_feat == 'lower':
                        roi_feats_f = self.roi_layers[i-1](feats[i-1], rois_)
                    elif self.fusion_feat == 'lowest':
                        roi_feats_f = self.roi_layers[0](feats[0], rois_)

                    if self.aggregation == 'sum':
                        roi_feats_t = roi_feats_c + roi_feats_f
                    elif self.aggregation == 'concat':
                        roi_feats_t = torch.cat(
                            [roi_feats_c, roi_feats_f], dim=1)
                else:
                    roi_feats_t = self.roi_layers[i](feats[i], rois_)

                if self.post_process is not None:
                    roi_feats_t = self.attention(roi_feats_t)

                roi_feats[inds] = roi_feats_t
            else:
                # Sometimes some pyramid levels will not be used for RoI
                # feature extraction and this will cause an incomplete
                # computation graph in one GPU, which is different from those
                # in other GPUs and will cause a hanging error.
                # Therefore, we add it to ensure each feature pyramid is
                # included in the computation graph to avoid runtime bugs.
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.
        return roi_feats
