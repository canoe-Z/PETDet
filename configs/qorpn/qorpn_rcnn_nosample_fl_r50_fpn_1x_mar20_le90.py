_base_ = ['./qorpn_rcnn_r50_fpn_1x_mar20_le90.py']

model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(
                type='SoftmaxFocalLoss',
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
        )
    ),
    train_cfg=dict(
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            score_thr=0),
        rcnn=dict(
            sampler=dict(
                _delete_=True,
                type='RPseudoSampler'),
            debug=False),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=1000,
            score_thr=0),
        rcnn=dict(
            nms_pre=2000,
            max_per_img=1000)
    )
)
