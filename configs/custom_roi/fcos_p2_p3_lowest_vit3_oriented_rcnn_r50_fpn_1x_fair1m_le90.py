_base_ = [
    '../custom_rpn/fcos/fcos_p2_p3_sample2_oriented_rcnn_r50_fpn_1x_fair1m_le90.py']
find_unused_parameters=True
angle_version = 'le90'
model = dict(
    roi_head=dict(
        type='OrientedDecoupleHeadRoIHead',
        bbox_roi_extractor=dict(
            type='DecoupleRotatedSingleRoIExtractor',
            cls_feat='lowest',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64]),
        bbox_head=dict(
            type='RotatedDecoupleBBoxHead',
            in_channels=256,
            use_vit=True
        )
    )
)
