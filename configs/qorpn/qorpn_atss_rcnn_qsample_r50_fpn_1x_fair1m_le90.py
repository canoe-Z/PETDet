_base_ = ['./qorpn_atss_rcnn_r50_fpn_1x_fair1m_le90_baseline3.py']

model = dict(
    train_cfg=dict(
        rpn_proposal=dict(
            nms_pre=500,
            max_per_img=500,
        ),
        rcnn=dict(
            sampler=dict(
                type='RRandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
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