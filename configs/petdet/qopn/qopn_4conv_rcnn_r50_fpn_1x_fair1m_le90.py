_base_ = ['../qopn_rcnn_r50_fpn_1x_fair1m_le90.py']

model = dict(
    rpn_head=dict(
        stacked_convs=4,
        enable_dam=False
    ),
)