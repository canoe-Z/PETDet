_base_ = [
    '../fcos/fcos_p2_p3_sample2_oriented_rcnn_r50_fpn_1x_fair1m_le90.py']

find_unused_parameters=True
model = dict(train_cfg=dict(rcnn=dict(sampler=dict(type='ROHEMSampler'))))
