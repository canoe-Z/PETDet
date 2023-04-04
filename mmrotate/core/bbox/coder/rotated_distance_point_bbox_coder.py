# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
from mmdet.core.bbox.coder.base_bbox_coder import BaseBBoxCoder

from ..builder import ROTATED_BBOX_CODERS
from ..transforms import norm_angle


@ROTATED_BBOX_CODERS.register_module()
class RotatedDistancePointBBoxCoder(BaseBBoxCoder):
    """Distance Point BBox coder.

    This coder encodes gt bboxes (x1, y1, x2, y2) into (top, bottom, left,
    right) and decode it back to the original.

    Args:
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
    """

    def __init__(self, clip_border=True, angle_version='oc'):
        super(BaseBBoxCoder, self).__init__()
        self.clip_border = clip_border
        self.angle_version = angle_version

    def encode(self, points, gt_bboxes, edge_swap=True, max_dis=None, eps=0.1):
        """Encode bounding box to distances.

        Args:
            points (Tensor): Shape (N, 2), The format is [x, y].
            gt_bboxes (Tensor): Shape (N, 5), The format is "xywha"
            max_dis (float): Upper bound of the distance. Default None.
            eps (float): a small value to ensure target < max_dis, instead <=.
                Default 0.1.

        Returns:
            Tensor: Box transformation deltas. The shape is (N, 5).
        """
        assert points.size(0) == gt_bboxes.size(0)
        assert points.size(-1) == 2
        assert gt_bboxes.size(-1) == 5
        return self.rbbox2distance(points, gt_bboxes, edge_swap, max_dis, eps)

    def decode(self, points, pred_bboxes, edge_swap=True, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (B, N, 2) or (N, 2).
            pred_bboxes (Tensor): Distance from the given point to 4
                boundaries and angle (left, top, right, bottom, angle).
                Shape (B, N, 5) or (N, 5)
            max_shape (Sequence[int] or torch.Tensor or Sequence[
                Sequence[int]],optional): Maximum bounds for boxes, specifies
                (H, W, C) or (H, W). If priors shape is (B, N, 4), then
                the max_shape should be a Sequence[Sequence[int]],
                and the length of max_shape should also be B.
                Default None.
        Returns:
            Tensor: Boxes with shape (N, 5) or (B, N, 5)
        """
        assert points.size(0) == pred_bboxes.size(0)
        assert points.size(-1) == 2
        assert pred_bboxes.size(-1) == 5
        if self.clip_border is False:
            max_shape = None
        return self.distance2rbbox(points, pred_bboxes, edge_swap, max_shape)

    def rbbox2distance(self, points, gt, edge_swap=True, max_dis=None, eps=0.1):
        gt_ctr, gw, gh, ga = torch.split(gt, [2, 1, 1, 1], dim=-1)

        if edge_swap:
            dtheta1 = norm_angle(ga, self.angle_version)
            dtheta2 = norm_angle(ga + np.pi / 2, self.angle_version)
            abs_dtheta1 = torch.abs(dtheta1)
            abs_dtheta2 = torch.abs(dtheta2)
            gw_regular = torch.where(abs_dtheta1 < abs_dtheta2, gw, gh)
            gh_regular = torch.where(abs_dtheta1 < abs_dtheta2, gh, gw)
            gw, gh = gw_regular, gh_regular
            ga = torch.where(abs_dtheta1 < abs_dtheta2, dtheta1, dtheta2)
        else:
            ga = norm_angle(ga, self.angle_version)

        cos, sin = torch.cos(ga), torch.sin(ga)
        Matrix = torch.cat([cos, sin, -sin, cos], dim=-
                           1).reshape(points.size(0), 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(Matrix, offset[..., None])
        offset = offset.squeeze(-1)

        W, H = gw[..., 0], gh[..., 0]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = W / 2 + offset_x
        right = W / 2 - offset_x
        top = H / 2 + offset_y
        bottom = H / 2 - offset_y

        if max_dis is not None:
            left = left.clamp(min=0, max=max_dis - eps)
            top = top.clamp(min=0, max=max_dis - eps)
            right = right.clamp(min=0, max=max_dis - eps)
            bottom = bottom.clamp(min=0, max=max_dis - eps)
        bbox_targets = torch.stack([left, top, right, bottom], -1)
        bbox_targets = torch.cat([bbox_targets, ga], dim=-1)
        return bbox_targets

    def distance2rbbox(self, points, distance, edge_swap=True, max_shape=None):
        distance, theta = distance.split([4, 1], dim=-1)

        cos, sin = torch.cos(theta), torch.sin(theta)
        M = torch.cat([cos, -sin, sin, cos], dim=-1).reshape(-1, 2, 2)

        wh = distance[..., :2] + distance[..., 2:]
        gw, gh = wh.split([1, 1], dim=-1)
        offset_t = (distance[..., 2:] - distance[..., :2]) / 2
        offset_t = offset_t.unsqueeze(-1)
        offset = torch.matmul(M, offset_t).squeeze(-1)
        ctr = points + offset
        theta = norm_angle(theta, self.angle_version)

        if edge_swap:
            gw_regular = torch.where(gw > gh, gw, gh)
            gh_regular = torch.where(gw > gh, gh, gw)
            theta_regular = torch.where(gw > gh, theta, theta + np.pi / 2)
            theta_regular = norm_angle(theta_regular, self.angle_version)
            bboxes = torch.cat([ctr, gw_regular, gh_regular, theta_regular],
                               dim=-1)
        else:
            bboxes = torch.cat([ctr, gw, gh, theta], dim=-1)

        if max_shape is not None:
            if bboxes.dim() == 2:
                # speed up
                bboxes[:, 2].clamp_(min=0, max=max_shape[1])
                bboxes[:, 3].clamp_(min=0, max=max_shape[0])

        return bboxes
