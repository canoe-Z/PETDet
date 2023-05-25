_base_ = ['./qorpn_atss_rcnn_lff_r50_fpn_1x_fair1m_le90.py']

model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='FineGrainedEnhancedHead',
            num_shared_convs=0,
            num_shared_fcs=0,
            num_cls_convs=0,
            num_cls_fcs=2,
            num_reg_convs=4,
            num_reg_fcs=1,
            loss_cls=dict(
                type='SoftmaxFocalLoss',
                use_sigmoid=False,
                gamma=1.0,
                loss_weight=1.0),
            beta=2.0,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
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
                type='RPseudoSampler')),
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
