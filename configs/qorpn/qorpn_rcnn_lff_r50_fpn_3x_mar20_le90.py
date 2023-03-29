_base_ = ['./qorpn_rcnn_r50_fpn_3x_mar20_le90.py']

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

# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     step=[24, 33])

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    step=[24, 33])

# optimizer = dict(
#     _delete_=True,
#     type='AdamW',
#     lr=0.0001,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))
