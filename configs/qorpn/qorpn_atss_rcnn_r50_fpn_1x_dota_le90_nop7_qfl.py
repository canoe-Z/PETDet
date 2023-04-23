_base_ = [
    './qorpn_atss_rcnn_r50_fpn_1x_dota_le90_nop7.py'
]
model = dict(
    rpn_head=dict(
        loss_cls_metric='QFL',
        loss_cls_qfl=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=0.5)))
