_base_ = [
    '../fcos/fcos_p2_p3_sample2_oriented_rcnn_r50_fpn_1x_fair1m_le90.py']

model = dict(
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=6),
        dict(type='DyHead', in_channels=256, out_channels=256, num_blocks=6)
    ],
    rpn_head=dict(
        stacked_convs=0)
)
