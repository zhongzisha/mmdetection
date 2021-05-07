

rem --cls-weights E:/cls_results/resnet50/bs32_lr0.001000_epochs20/best.pt
set PYTHONPATH=F:\gd\
set SUBSET=val
set IMGSIZE=800
set NET=yolov5l_768_4classes
set CONFIG=retinanet_r50_fpn_1x_coco
set CONFIG=faster_rcnn_r50_fpn_ohem_1x_coco
set CONFIG=cascade_rcnn_r50_fpn_1x_coco
set CONFIG=faster_rcnn_r50_fpn_1x_coco
set CONFIG=faster_rcnn_r50_fpn_2x_coco
set CONFIG=faster_rcnn_r50_fpn_ohem_1x_coco
set CONFIG=cascade_rcnn_r50_fpn_1x_coco
set CONFIG=faster_rcnn_r50_fpn_1x_coco
set CONFIG=retinanet_r50_fpn_1x_coco
set CONFIG=faster_rcnn_r50_fpn_1x_coco_lr0.001
set CONFIG=faster_rcnn_r50_fpn_1x_coco_lr0.00075
set CONFIG=faster_rcnn_r50_fpn_1x_coco_lr0.00025
set CONFIG=cascade_rcnn_r50_fpn_2x_coco_lr0.001
set CONFIG=retinanet_r50_fpn_dc5_2x_coco_lr0.0008
set CONFIG=faster_rcnn_r101_fpn_dc5_1x_coco_lr0.001
set CONFIG=faster_rcnn_r50_fpn_dc5_1x_coco_lr0.001_nms
set CONFIG=faster_rcnn_r50_fpn_dc5_1x_coco_lr0.001_nms2
set CONFIG=faster_rcnn_r50_fpn_dc5_2x_coco_lr0.001_nms2
set CONFIG=faster_rcnn_r50_fpn_dc5_ohem_1x_coco_lr0.001_nms2_anchor1
set CONFIG=faster_rcnn_r50_fpn_dc5_ohem_1x_coco_lr0.001_nms2_anchor2
set CONFIG=faster_rcnn_r50_fpn_dc5_1x_coco_lr0.001
set CONFIG=faster_rcnn_r50_fpn_dc5_1x_coco_lr0.001_800

set EPOCHNAME=latest
set EPOCHNAME=epoch_11


python demo\detect_gd1024_4classes.py ^
    --source E:\%SUBSET%_list.txt ^
    --config E:\mmdetection\work_dirs\%CONFIG%\%CONFIG%.py ^
    --weights E:\mmdetection\work_dirs\%CONFIG%\%EPOCHNAME%.pth ^
    --score-thres 0.1 ^
    --iou-thres 0.5 ^
    --hw-thres 10 ^
    --img-size %IMGSIZE% ^
    --gap 32 ^
    --save-txt ^
    --device 0 ^
    --project E:\mmdetection\work_dirs\%CONFIG%\outputs_%SUBSET%_%IMGSIZE%_%EPOCHNAME% ^
    --gt-xml-dir E:\gd_gt_combined_4classes\ ^
    --gt-subsize 5120 ^
    --gt-gap 128 ^
    --big-subsize 10240 ^
    --batchsize 4
















