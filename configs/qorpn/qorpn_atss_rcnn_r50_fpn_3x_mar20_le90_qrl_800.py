_base_ = [
    './qorpn_atss_rcnn_r50_fpn_3x_mar20_le90_800.py'
]
model = dict(
    rpn_head=dict(
        loss_cls_metric='QFL',
        loss_cls_qfl=dict(
            type='QRFocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            beta=1.5,
            gamma=2.0,
            loss_weight=0.5)))