_base_ = 'fcos_r50_caffe_fpn_gn-head_1x_coco.py'

model = dict(
    pretrained='open-mmlab://detectron2/resnet50_caffe',
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    bbox_head=dict(
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=True,
        center_sampling=True,
        conv_bias=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)),
    # training and testing settings
    test_cfg=dict(nms=dict(type='nms', iou_threshold=0.6)))

optimizer_config = dict(_delete_=True, grad_clip=None)

lr_config = dict(warmup='linear')
