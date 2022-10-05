_base_ = ['./fcos_p2_p3_4conv1fc_oriented_rcnn_r50_fpn_1x_fair1m_le90.py']

model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='RotatedShared4Conv1FCBBoxHead',
            loss_cls=dict(
                loss_weight=1.5
            ),
            loss_bbox=dict(
                loss_weight=1.5
            )
        )
    )
)
