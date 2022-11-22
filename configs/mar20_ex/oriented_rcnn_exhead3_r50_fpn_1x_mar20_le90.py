_base_ = ['./oriented_rcnn_r50_fpn_1x_mar20_le90.py']

find_unused_parameters = True
model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='ExperimentBBoxHead'
        )
    )
)
