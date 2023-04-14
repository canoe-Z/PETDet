_base_ = [
    './qorpn_atss_rcnn_r50_fpn_1x_fair1m_le90_baseline3.py'
]
model = dict(
    rpn_head=dict(
        loss_cls_metric='QFL',
        loss_cls_qfl=dict(
            type='QRFocalLoss',
            use_sigmoid=True,
            alpha=0.5,
            beta=2.0,
            loss_weight=0.5)))
