_base_ = ['./qorpn_rcnn_r50_fpn_1x_mar20_le90.py']
find_unused_parameters=True
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
    roi_head=dict(
        start_level=1)
)
