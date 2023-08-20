_base_ = ['./petdet_r50_fpn_1x_fair1m_le90.py']

angle_version = 'le90'
data_root = './data/split_ss_fair1m2_0/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(
    train=dict(
        ann_file=data_root + 'val_petdet/annfiles/',
        img_prefix=data_root + 'val_petdet/images/',
        pipeline=train_pipeline),
    val=dict(
        ann_file=data_root + 'val/annfiles/',
        img_prefix=data_root + 'val/images/'),
    test=dict(
        ann_file=data_root + 'val/annfiles/',
        img_prefix=data_root + 'val/images/'))

model = dict(
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.5,
        )
    )
)
