_base_ = ['../qopn_rcnn_bcfn_r50_fpn_1x_fair1m_le90.py']

model = dict(
    fusion=dict(
        type='LFF',
        feat_channels=256,
    ),
)