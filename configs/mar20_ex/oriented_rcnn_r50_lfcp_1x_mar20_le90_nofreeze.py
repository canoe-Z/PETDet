_base_ = ['./oriented_rcnn_r50_fpn_1x_mar20_le90.py']
model = dict(
    backbone=dict(
        frozen_stages=-1),
)
