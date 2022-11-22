_base_ = ['./oriented_rcnn_r50_fpn_1x_fair1m_le90.py']
find_unused_parameters = True
model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='ExperimentBBoxHead',
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
        )
    )
)
