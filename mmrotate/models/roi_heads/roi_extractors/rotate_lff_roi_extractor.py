# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv import ops
from mmcv.runner import force_fp32

from ...builder import ROTATED_ROI_EXTRACTORS
from .rotate_single_level_roi_extractor import RotatedSingleRoIExtractor


@ROTATED_ROI_EXTRACTORS.register_module()
class RotatedLFFRoIExtractor(RotatedSingleRoIExtractor):
    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 lowlevel_featmap_stride=2,
                 fusion_mode='cat',
                 finest_scale=56,
                 init_cfg=None):
        self.lowlevel_featmap_stride = lowlevel_featmap_stride
        self.fusion_mode = fusion_mode
        super(RotatedLFFRoIExtractor,
              self).__init__(roi_layer,
                             out_channels,
                             featmap_strides,
                             finest_scale,
                             init_cfg)

    def build_roi_layers(self, layer_cfg, featmap_strides):
        """Build RoI operator to extract feature from each level feature map.

        Args:
            layer_cfg (dict): Dictionary to construct and config RoI layer
                operation. Options are modules under ``mmcv/ops`` such as
                ``RoIAlign``.
            featmap_strides (List[int]): The stride of input feature map w.r.t
                to the original image size, which would be used to scale RoI
                coordinate (original image coordinate system) to feature
                coordinate system.

        Returns:
            nn.ModuleList: The RoI extractor modules for each level feature \
                map.
        """

        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')

        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        total_featmap_strides = featmap_strides + \
            [self.lowlevel_featmap_stride]
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in total_featmap_strides])
        return roi_layers

    @force_fp32(apply_to=('feats', 'lowlevelfeat',), out_fp16=True)
    def forward(self, feats, lowlevelfeat, rois, roi_scale_factor=None):
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
                roi_feats_lowlevel = self.roi_layers[-1](lowlevelfeat, rois_)
                if self.fusion_mode == 'cat':
                    roi_feats[inds] = torch.cat(
                        [roi_feats_t, roi_feats_lowlevel], 1)
                elif self.fusion_mode == 'sum':
                    roi_feats[inds] = roi_feats_t + roi_feats_lowlevel
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
