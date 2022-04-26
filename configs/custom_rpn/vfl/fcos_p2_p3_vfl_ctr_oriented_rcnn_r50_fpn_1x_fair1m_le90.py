_base_ = [
    '../fcos/fcos_p2_p3_sample2_oriented_rcnn_r50_fpn_1x_fair1m_le90.py']
angle_version = 'le90'
model = dict(
    rpn_head=dict(
        use_vfl=True,
        loss_cls_vfl=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=0.5)
    )
)
