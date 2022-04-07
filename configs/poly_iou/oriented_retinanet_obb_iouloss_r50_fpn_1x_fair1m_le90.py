_base_ = [
    '../oriented_retinanet/oriented_retinanet_obb_r50_fpn_1x_fair1m_le90.py'
]

model = dict(
    bbox_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(
            _delete_=True,
            type='PolyIoULoss',
            loss_weight=1.0)))