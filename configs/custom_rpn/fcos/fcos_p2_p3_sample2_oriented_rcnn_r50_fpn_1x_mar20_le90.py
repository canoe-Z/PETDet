_base_ = ['./fcos_p2_p3_oriented_rcnn_r50_fpn_1x_mar20_le90.py']

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=20
        )
    ),
    train_cfg=dict(
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
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
            nms_pre=2000,
            max_per_img=1000),
        rcnn=dict(
            nms_pre=1000,
            max_per_img=1000)
    )
)