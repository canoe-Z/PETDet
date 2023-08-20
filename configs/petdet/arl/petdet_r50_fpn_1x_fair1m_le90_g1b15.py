_base_ = ['../petdet_r50_fpn_1x_fair1m_le90.py']

model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(
                gamma=1.0,
                beta=1.5)
        )
    )
)
