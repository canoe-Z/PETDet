_base_ = ['./exgiou3adamw_qorpn_rcnn_r50_fpn_1x_fair1m_le90.py']

model = dict(
    type='OrientedRCNNLFF',
    backbone=dict(
        type='LowlResNet'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=6),
    rpn_head=dict(
        start_level=1),
    roi_head=dict(
        type='LFFDecoupleHeadRoIHead',
        start_level=1,
        bbox_roi_extractor=dict(
            type='RotatedLFFRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=512,
            featmap_strides=[8, 16, 32, 64],
            lowlevel_featmap_stride=4
        ),
        bbox_head=dict(
            in_channels=512))
)
