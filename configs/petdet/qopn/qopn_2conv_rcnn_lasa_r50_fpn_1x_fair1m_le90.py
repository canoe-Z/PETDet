_base_ = ['../qopn_rcnn_r50_fpn_1x_fair1m_le90.py']

model = dict(
    rpn_head=dict(
        stacked_convs=2,
        enable_dam=True,
        use_fpn_feature=False,
        enable_sa=True
    ),
)