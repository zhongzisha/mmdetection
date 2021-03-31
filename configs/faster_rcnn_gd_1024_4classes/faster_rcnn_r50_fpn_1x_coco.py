_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_gd1024_4classes.py',
    '../_base_/datasets/gd1024_detection_4classes.py',
    '../_base_/schedules/schedule_1x_gd1024_4classes.py',
    '../_base_/default_runtime_gd1024_4classes.py'
]
