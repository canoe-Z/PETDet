_base_ = ['./fcos_oriented_rcnn_r50_fpn_1x_shiprs3_le90.py']

angle_version = 'le90'
model = dict(
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=6),
    rpn_head=dict(
        start_level=1,
        strides=[8, 16, 32, 64, 128]),
    roi_head=dict(
        bbox_roi_extractor=dict(
            featmap_strides=[4, 8, 16, 32])))