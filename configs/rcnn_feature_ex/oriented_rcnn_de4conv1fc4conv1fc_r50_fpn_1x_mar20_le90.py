_base_ = [
    '../oriented_rcnn/oriented_rcnn_r50_fpn_1x_mar20_le90.py']

model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='RotatedConvFCBBoxHead',
            num_shared_convs=0,
            num_shared_fcs=0,
            num_cls_convs=4,
            num_cls_fcs=1,
            num_reg_convs=4,
            num_reg_fcs=1
        )
    )
)