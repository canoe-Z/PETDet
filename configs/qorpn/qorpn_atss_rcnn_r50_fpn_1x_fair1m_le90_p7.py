_base_ = [
    './qorpn_atss_rcnn_r50_fpn_1x_fair1m_le90.py'
]
model = dict(
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    rpn_head=dict(
        strides=[8, 16, 32, 64, 128],
        prior_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=8,
            scales_per_octave=1,
            center_offset=0.0,
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128])
    ))
