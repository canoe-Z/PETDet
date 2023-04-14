_base_ = [
    './qorpn_atss_rcnn_r50_fpn_1x_fair1m_le90_baseline3.py'
]
model = dict(
    rpn_head=dict(
        use_sca=False))
