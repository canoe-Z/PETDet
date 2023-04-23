_base_ = ['./qorpn_rcnn_r50_fpn_3x_mar20_le90.py']

model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='RotatedConvFCBBoxHead',
            num_shared_convs=0,
            num_shared_fcs=0,
            num_cls_convs=0,
            num_cls_fcs=2,
            num_reg_convs=2,
            num_reg_fcs=1,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
        )
    )
)
