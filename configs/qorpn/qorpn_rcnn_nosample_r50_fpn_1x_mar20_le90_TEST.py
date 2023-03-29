_base_ = ['./qorpn_rcnn_r50_fpn_1x_mar20_le90.py']

model = dict(
    rpn_head=dict(
        shrink_sigma=[0, 0.15, 0.3, 0.45, 0.6]
    ),
    train_cfg=dict(
        rpn_proposal=dict(
            nms_pre=500,
            max_per_img=500,
        ),
        rcnn=dict(
            sampler=dict(
                _delete_=True,
                type='RPseudoSampler'),
            debug=False),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=500,
            max_per_img=500),
        rcnn=dict(
            nms_pre=500,
            max_per_img=500)
    )
)
