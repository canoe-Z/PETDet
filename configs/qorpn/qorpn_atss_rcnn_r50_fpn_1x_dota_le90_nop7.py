_base_ = [
    './qorpn_atss_rcnn_r50_fpn_1x_dota_le90.py'
]
model = dict(
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        num_outs=4),
    rpn_head=dict(
        strides=[8, 16, 32, 64],
        prior_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=8,
            scales_per_octave=1,
            center_offset=0.0,
            ratios=[1.0],
            strides=[8, 16, 32, 64])
    ))

# fp16 = dict(loss_scale='dynamic')
evaluation = dict(interval=12, metric='mAP')