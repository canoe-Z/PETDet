_base_ = ['./fcos_p2_p3_vfl_oriented_rcnn_r50_fpn_1x_fair1m_le90.py']
angle_version = 'le90'
model = dict(
    rpn_head=dict(
        use_ctr=True
    )
)
