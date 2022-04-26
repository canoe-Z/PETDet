_base_ = [
    '../fcos/fcos_p2_p3_sample2_oriented_rcnn_r50_fpn_1x_fair1m_le90.py']
angle_version = 'le90'
course_label_table = [0, 0, 0, 0, 0, 3,
                      3, 0, 0, 0,
                      0, 4, 2, 0, 2,
                      1, 2, 1,
                      2, 1, 3, 4,
                      1, 1, 1, 4,
                      2, 3, 2, 2, 2,
                      1, 2, 1, 0, 1, 2]
model = dict(
    type='CARPNRotatedTwoStageDetector',
    course_label_table=course_label_table,
    rpn_head=dict(
        type='RotatedFCOSCARPNHead',
        num_classes=5
    )
)
