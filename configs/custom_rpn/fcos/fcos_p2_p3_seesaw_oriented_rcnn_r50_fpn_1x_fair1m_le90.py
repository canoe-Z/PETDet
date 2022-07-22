_base_ = ['./fcos_p2_p3_sample2_oriented_rcnn_r50_fpn_1x_fair1m_le90.py']


model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(
                type='SeesawLoss',
                p=0.8,
                q=2.0,
                num_classes=37,
                loss_weight=1.0),
        )
    ),
    train_cfg=dict(
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
        )
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=1000),
        rcnn=dict(
            nms_pre=1000,
            max_per_img=1000)
    )
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2
)
