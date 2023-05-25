_base_ = ['./qorpn_atss_rcnn_lff_r50_fpn_1x_fair1m_le90.py']

model = dict(
    train_cfg=dict(
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=512,
        ),
        rcnn=dict(
            sampler=dict(
                _delete_=True,
                type='RPseudoSampler')),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=512),
        rcnn=dict(
            nms_pre=2000,
            max_per_img=512)
    )
)
