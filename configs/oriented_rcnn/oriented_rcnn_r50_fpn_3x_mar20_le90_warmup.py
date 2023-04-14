_base_ = ['./oriented_rcnn_r50_fpn_3x_mar20_le90.py']

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1.0 / 2000,
    step=[24, 33])