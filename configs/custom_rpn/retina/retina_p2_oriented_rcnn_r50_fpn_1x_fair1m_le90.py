_base_ = ['./retina_oriented_rcnn_r50_fpn_1x_fair1m_le90.py']

angle_version = 'le90'
model = dict(
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        version=angle_version,
        anchor_generator=dict(
            strides=[4, 8, 16, 32, 64])),
    roi_head=dict(
        bbox_roi_extractor=dict(
            featmap_strides=[4, 8, 16, 32])),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                gpu_assign_thr=500))))

fp16=dict(loss_scale='dynamic')
