# Copyright (c) OpenMMLab. All rights reserved.
import torch
from abc import ABCMeta
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import build_plugin_layer
from mmcv.runner import BaseModule
from mmrotate.core import (build_assigner, build_sampler, rbbox2result,
                           rbbox2roi)
from ..builder import (ROTATED_HEADS, build_head, build_roi_extractor,
                       build_shared_head)

import torch.nn.functional as F
import torch.nn as nn
from mmcv.cnn import ConvModule, xavier_init
from mmcv.cnn.bricks import build_plugin_layer
from mmcv.runner import BaseModule


@ROTATED_HEADS.register_module()
class LFFDecoupleHeadRoIHead(BaseModule, metaclass=ABCMeta):
    """Simplest base rotated roi head including one bbox head.

    Args:
        bbox_roi_extractor (dict, optional): Config of ``bbox_roi_extractor``.
        bbox_head (dict, optional): Config of ``bbox_head``.
        shared_head (dict, optional): Config of ``shared_head``.
        train_cfg (dict, optional): Config of train.
        test_cfg (dict, optional): Config of test.
        pretrained (str, optional): Path of pretrained weight.
        init_cfg (dict, optional): Config of initialization.
        version (str, optional): Angle representations. Defaults to 'oc'.
    """

    def __init__(self,
                 bbox_roi_extractor=None,
                 lowlevel_bbox_roi_extractor=None,
                 bbox_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 version='oc',
                 att_cfg=dict(type='ContextBlock',
                              in_channels=256, ratio=1. / 4),
                 #  upsample_cfg=dict(
                 #      type='carafe',
                 #      up_kernel=5,
                 #      up_group=1,
                 #      encoder_kernel=3,
                 #      encoder_dilation=1),
                 upsample_cfg=dict(mode='nearest'),
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):

        super(LFFDecoupleHeadRoIHead, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.version = version

        if shared_head is not None:
            shared_head.pretrained = pretrained
            self.shared_head = build_shared_head(shared_head)

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor,
                                lowlevel_bbox_roi_extractor, bbox_head)

        self.init_assigner_sampler()

        self.with_bbox = True if bbox_head is not None else False
        self.with_shared_head = True if shared_head is not None else False

        self.att_cfg = att_cfg
        self.upsample_cfg = upsample_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.init_fusion_module()

    def init_fusion_module(self):
        out_channels = 256
        self.y_conv = ConvModule(
            out_channels//4,
            out_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            act_cfg=self.act_cfg,
            inplace=False)
        self.down_conv = ConvModule(
            out_channels*2,
            out_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            act_cfg=dict(type='ReLU'),
            inplace=False)
        # self.fil_conv = ConvModule(
        #     out_channels,
        #     out_channels,
        #     3,
        #     conv_cfg=self.conv_cfg,
        #     padding=1,
        #     norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        #     act_cfg=dict(type='ReLU'),
        #     inplace=False)
        # self.fil_conv_1 = ConvModule(
        #     out_channels,
        #     out_channels,
        #     3,
        #     conv_cfg=self.conv_cfg,
        #     padding=1,
        #     norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        #     act_cfg=None,
        #     inplace=False)
        # self.att_module = build_plugin_layer(self.att_cfg, '_att_module')[1]
        # self.upsample_modules = ModuleList()
        # upsample_cfg_ = self.upsample_cfg.copy()
        # upsample_cfg_.update(channels=out_channels, scale_factor=2)
        # upsample_module = build_upsample_layer(upsample_cfg_)
        # self.upsample_modules.append(upsample_module)
        # self.att_module_1 = build_plugin_layer(
        #     self.att_cfg, '_att_module_1')[1]

    def init_weights(self):
        """Initialize the weights of module."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                xavier_init(m, distribution='uniform')
        # for m in self.modules():
        #     if isinstance(m, CARAFEPack):
        #         m.init_weights()

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    # def init_bbox_head(self, cls_bbox_roi_extractor, reg_bbox_roi_extractor, bbox_head):
    #     """Initialize ``bbox_head``.

    #     Args:
    #         bbox_roi_extractor (dict): Config of ``bbox_roi_extractor``.
    #         bbox_head (dict): Config of ``bbox_head``.
    #     """
    #     self.cls_bbox_roi_extractor = build_roi_extractor(
    #         cls_bbox_roi_extractor)
    #     self.reg_bbox_roi_extractor = build_roi_extractor(
    #         reg_bbox_roi_extractor)
    #     self.bbox_head = build_head(bbox_head)

    def init_bbox_head(self, bbox_roi_extractor, lowlevel_bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``.

        Args:
            bbox_roi_extractor (dict): Config of ``bbox_roi_extractor``.
            bbox_head (dict): Config of ``bbox_head``.
        """
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        # self.lowlevel_bbox_roi_extractor = build_roi_extractor(
        #     lowlevel_bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def forward_dummy(self, x, y, proposals):
        """Dummy forward function.

        Args:
            x (list[Tensors]): list of multi-level img features.
            proposals (list[Tensors]): list of region proposals.

        Returns:
            list[Tensors]: list of region of interest.
        """
        outs = ()
        rois = rbbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, y, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        return outs

    def forward_train(self,
                      x,
                      y,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task. Always
                set to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        # assign gts and sample proposals
        if self.with_bbox:

            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])

                if gt_bboxes[i].numel() == 0:
                    sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
                        (0, gt_bboxes[0].size(-1))).zero_()
                else:
                    sampling_result.pos_gt_bboxes = gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]

                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, y, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        return losses

    def _bbox_forward(self, x, y, rois):
        """Box head forward function used in both training and testing.

        Args:
            x (list[Tensor]): list of multi-level img features.
            rois (list[Tensors]): list of region of interests.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        y_out = self.y_conv(y)
        #prev_shape = laterals[i - 1].shape[2:]
        upsample_out = F.interpolate(
            x[0], y_out.shape[2:], **self.upsample_cfg)
        # print(upsample_out.shape)
        # assert (1 == 2)
        # upsample_out = self.upsample_modules[0](x[0])

        cat_feats = torch.cat((upsample_out, y_out), dim=1)
        down_feats = self.down_conv(cat_feats)
        # out_feats = self.fil_conv(self.fil_conv_1(down_feats))
        #att_feats = self.att_module(down_feats)
        #att_feats = upsample_out+att_feats
        # att_feats = self.att_module_1(att_feats)
        lowlevel_feat = upsample_out+down_feats

        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], lowlevel_feat, rois)
        # lowlevel_bbox_feats = self.lowlevel_bbox_roi_extractor(
        #     [lowlevel_feat], rois)
        #bbox_feats = torch.cat([bbox_feats, lowlevel_bbox_feats], dim=1)
        if isinstance(bbox_feats, list):
            assert (len(bbox_feats) == 2)
            bbox_cls_feats, bbox_reg_feats = bbox_feats
            if self.with_shared_head:
                bbox_cls_feats = self.shared_head(bbox_cls_feats)
                bbox_reg_feats = self.shared_head(bbox_reg_feats)
            cls_score, bbox_pred = self.bbox_head(
                bbox_cls_feats, bbox_reg_feats)
        else:
            bbox_cls_feats = bbox_feats
            bbox_reg_feats = bbox_feats
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            bbox_feats=bbox_cls_feats)
        return bbox_results

    def _bbox_forward_train(self, x, y, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training.

        Args:
            x (list[Tensor]): list of multi-level img features.
            sampling_results (list[Tensor]): list of sampling results.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, y, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    async def async_simple_test(self,
                                x,
                                y,
                                proposal_list,
                                img_metas,
                                rescale=False):
        """Async test without augmentation.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposal_list (list[Tensors]): list of region proposals.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, y, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = rbbox2result(det_bboxes, det_labels,
                                    self.bbox_head.num_classes)
        return bbox_results

    def simple_test(self, x, y, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposal_list (list[Tensors]): list of region proposals.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, y, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            rbbox2result(det_bboxes[i], det_labels[i],
                         self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        return bbox_results

    def aug_test(self, x, y, proposal_list, img_metas, rescale=False):
        """Test with augmentations."""
        raise NotImplementedError

    def simple_test_bboxes(self,
                           x,
                           y,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains \
                the boxes of the corresponding image in a batch, each \
                tensor has the shape (num_boxes, 5) and last dimension \
                5 represent (cx, cy, w, h, a, score). Each Tensor \
                in the second list is the labels with shape (num_boxes, ). \
                The length of both lists should be equal to batch_size.
        """

        rois = rbbox2roi(proposals)
        bbox_results = self._bbox_forward(x, y, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels
