_base_ = ['../oriented_rcnn/orpn_r50_fpn_1x_fair1m_le90.py']

angle_version = 'le90'
model = dict(
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    rpn_head=dict(
        _delete_=True,
        type='RotatedRetinaRPNHead',
        in_channels=256,
        feat_channels=256,
        reg_dim=6,
        assign_by_circumhbbox=angle_version,
        version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_range=angle_version,
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.5),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=0.5)),
    train_cfg=dict(
        rpn=dict(
            _delete_=True,
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='BboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)))