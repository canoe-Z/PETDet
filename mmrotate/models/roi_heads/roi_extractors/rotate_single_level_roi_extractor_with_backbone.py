# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv import ops
from mmcv.runner import force_fp32

from ...builder import ROTATED_ROI_EXTRACTORS
from .rotate_single_level_roi_extractor import RotatedSingleRoIExtractor


@ROTATED_ROI_EXTRACTORS.register_module()
class RotatedSingleRoIExtractorWithBackbone(RotatedSingleRoIExtractor):
    def __init__(self,
                 **kwargs):
        super(RotatedSingleRoIExtractorWithBackbone,
              self).__init__(**kwargs)

    @force_fp32(apply_to=('feats', 'backbone_feat',), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None, backbone_feat=None):
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
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats_backbone = self.roi_layers[0](backbone_feat, rois_)
                roi_feats[inds] = torch.cat(
                    [roi_feats_t, roi_feats_backbone], 1)
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
