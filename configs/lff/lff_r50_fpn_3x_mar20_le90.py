_base_ = ['../oriented_rcnn/oriented_rcnn_r50_fpn_3x_mar20_le90.py']

model = dict(
    type='OrientedRCNNLFF',
    backbone=dict(
        type='LowlResNet'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    roi_head=dict(
        type='LFFDecoupleHeadRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedLFFRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=512,
            featmap_strides=[4, 8, 16, 32],
            lowlevel_featmap_stride=4
        ),
        bbox_head=dict(
            in_channels=512))
)

# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=2000,
#     warmup_ratio=0.0005,
#     step=[24, 33])
# fp16 = dict(loss_scale='dynamic')

# optimizer = dict(lr=0.02)
