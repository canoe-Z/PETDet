_base_ = ['./qorpn_rcnn_nosample_r50_fpn_1x_mar20_le90.py']

model = dict(
    type='OrientedRCNNLFF',
    backbone=dict(
        type='LowlResNet'),
    roi_head=dict(
        type='LFFDecoupleHeadRoIHead',
        fpn_stride=8,
        bbox_roi_extractor=dict(
            type='RotatedLFFRoIExtractor',
            out_channels=512,
            lowlevel_featmap_stride=4
        ),
        bbox_head=dict(
            in_channels=512))
)
