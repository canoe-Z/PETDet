_base_ = ['./qopn_rcnn_r50_fpn_3x_shiprs3_le90.py']

model = dict(
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=6),
    rpn_head=dict(
        start_level=1),
    fusion=dict(
        type='BCFN',
        feat_channels=256,
    ),
    roi_head=dict(
        bbox_roi_extractor=dict(
            featmap_strides=[4, 8, 16, 32],
        )
    )
)