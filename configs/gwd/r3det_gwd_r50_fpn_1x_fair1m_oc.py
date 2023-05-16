_base_ = [
    '../kld/r3det_kld_stable_r50_fpn_1x_fair1m_oc.py'
]

model = dict(
    bbox_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(_delete_=True, type='GDLoss', loss_type='gwd', loss_weight=5.0)))