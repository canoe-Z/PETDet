_base_ = ['./qorpn_atss_rcnn_r50_fpn_1x_fair1m_le90.py']

model = dict(
    type='OrientedRCNNLFF',
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5),
    rpn_head=dict(
        start_level=1),
    lff_module=dict(
        type='LFF',
        feat_channels=256,
    ),
    roi_head=dict(
        bbox_roi_extractor=dict(
            featmap_strides=[4, 8, 16, 32],
        )
    )
)

# optimizer = dict(
#     paramwise_cfg=dict(
#         custom_keys={'lff_module': dict(lr_mult=2.0, decay_mult=1.0)}))