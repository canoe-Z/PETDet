_base_ = [
    '../custom_rpn/fcos/fcos_p2_p3_sample2_oriented_rcnn_r50_fpn_1x_fair1m_le90.py']
find_unused_parameters = True
angle_version = 'le90'
model = dict(
    roi_head=dict(
        type='OrientedDecoupleHeadRoIHead',
        bbox_roi_extractor=dict(
            type='DecoupleRotatedSingleRoIExtractor',
            cls_feat='lowest',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=14,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64]),
        bbox_head=dict(
            type='RotatedDecoupleBBoxHead',
            in_channels=256,
            use_vit=True,
            roi_feat_size=14
        )
    )
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
