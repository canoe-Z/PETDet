_base_ = [
    './qorpn_rcnn_r50_fpn_1x_fair1m_le90_baseline3.py'
]

lr_config = dict(
    warmup_iters=500,
    warmup_ratio=1.0 / 3)