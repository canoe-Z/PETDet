_base_ = ['../oriented_rcnn/oriented_rcnn_r50_fpn_1x_fair1m_le90.py']

angle_version = 'le90'
model = dict(
    roi_head=dict(
        _delete_=True,
        type='BatchOrientedRoIHeadFPN',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractorFPN',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCRoIAttentionBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=37,
            attention_hidden_channels=128,
            attention_pool_size=2,
            attention_pool_size_gram=1,
            subsample='naive',
            combination='cas_rram_gram',
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(
        rpn_proposal=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000),
        rcnn=dict(
            nms_pre=1000,
            max_per_img=1000)))