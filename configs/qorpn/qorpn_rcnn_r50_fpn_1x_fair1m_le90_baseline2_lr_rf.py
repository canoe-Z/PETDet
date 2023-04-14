_base_ = [
    './qorpn_rcnn_r50_fpn_1x_fair1m_le90_baseline2.py'
]

model = dict(
    rpn_head=dict(
        refine_bbox=False,
        loss_bbox=dict(type='PolyGIoULoss', loss_weight=0.5),
        loss_bbox_refine=dict(type='PolyGIoULoss', loss_weight=0.75)))
