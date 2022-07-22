_base_ = [
    '../custom_rpn/fcos/fcos_p2_p3_sample2_oriented_rcnn_r50_fpn_1x_fair1m_le90.py']
angle_version = 'le90'
model = dict(
    roi_head=dict(
        bbox_roi_extractor=dict(
            type='CustomRotatedSingleRoIExtractor',
            fusion_feat='lowest',
            aggregation='concat',
            post_process='CBAM',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=512,
            featmap_strides=[4, 8, 16, 32, 64]),
        bbox_head=dict(
            in_channels=512
        )
    )
)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2
)
