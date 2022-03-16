_base_ = ['../oriented_rcnn/oriented_rcnn_r50_fpn_1x_fair1m_le90.py']

model = dict(
    roi_head=dict(
        bbox_roi_extractor=dict(
            roi_layer=dict(
                out_size=15
            )
        ),
        bbox_head=dict(
            roi_feat_size=15
        )
    )
)
