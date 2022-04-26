_base_ = [
    '../fcos/fcos_p2_p3_sample2_oriented_rcnn_r50_fpn_1x_fair1m_le90.py']
angle_version = 'le90'
model = dict(
    rpn_head=dict(
        _delete_=True,
        type='RotatedVFFCOSRPNHead',
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        start_level=1,
        use_vfl=True,
        loss_cls_vfl=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=0.5),
        angle_version=angle_version,
        edge_swap=False,
        strides=[8, 16, 32, 64, 128],
        scale_theta=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.5),
        loss_bbox=dict(type='PolyIoULoss', loss_weight=0.5)
    )
)
