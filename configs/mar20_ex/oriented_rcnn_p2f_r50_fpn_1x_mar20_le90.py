_base_ = ['./oriented_rcnn_r50_fpn_1x_mar20_le90.py']

model = dict(
    roi_head=dict(
        bbox_roi_extractor=dict(
            type='CustomRotatedSingleRoIExtractor',
            fusion_feat='lowest',
            aggregation='concat',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=512,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            in_channels=512,
            type='RotatedConvFCBBoxHead',
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0
        )
    )
)
