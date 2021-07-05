_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_gd1024_rotate.py',
    '../_base_/datasets/gd1024_detection_4classes_box_aug_v4.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]
optimizer = dict(lr=0.0025)