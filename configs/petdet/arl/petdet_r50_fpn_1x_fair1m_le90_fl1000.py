_base_ = ['../qopn_rcnn_bcfn_r50_fpn_1x_fair1m_le90.py']

model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(
                type='SoftmaxFocalLoss',
                alpha=0.25,
                gamma=2.0),
        )
    ),
    train_cfg=dict(
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=None,
        ),
        rcnn=dict(
            sampler=dict(
                _delete_=True,
                type='RPseudoSampler',
                with_score=False),
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=None),
        rcnn=dict(
            nms_pre=2000,
            max_per_img=1000)
    )
)
