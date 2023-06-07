_base_ = ['../petdet_r50_fpn_1x_fair1m_le90.py']

model = dict(
    train_cfg=dict(
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=None,
        )
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=None),
        rcnn=dict(
            nms_pre=2000,
            max_per_img=2000)
    )
)
