

rem --cls-weights E:/cls_results/resnet50/bs32_lr0.001000_epochs20/best.pt
set PYTHONPATH=F:\gd\
set SUBSET=val
set NET=yolov5l_768_4classes
set CONFIG=retinanet_r50_fpn_1x_coco
set CONFIG=faster_rcnn_r50_fpn_ohem_1x_coco
set CONFIG=cascade_rcnn_r50_fpn_1x_coco
set CONFIG=faster_rcnn_r50_fpn_1x_coco
set CONFIG=faster_rcnn_r50_fpn_2x_coco
set CONFIG=faster_rcnn_r50_fpn_ohem_1x_coco
set CONFIG=cascade_rcnn_r50_fpn_1x_coco
set CONFIG=faster_rcnn_r50_fpn_1x_coco


python demo\detect_gd1024_4classes.py ^
    --source E:\%SUBSET%_list.txt ^
    --config E:\mmdetection\work_dirs\%CONFIG%\%CONFIG%.py ^
    --weights E:\mmdetection\work_dirs\%CONFIG%\latest.pth ^
    --score-thres 0.1 ^
    --iou-thres 0.5 ^
    --hw-thres 10 ^
    --img-size 1024 ^
    --gap 256 ^
    --save-txt ^
    --device 0 ^
    --project E:\mmdetection\work_dirs\%CONFIG%\outputs_%SUBSET% ^
    --gt-xml-dir E:\gd_gt_combined_4classes\ ^
    --gt-subsize 5120 ^
    --gt-gap 128 ^
    --big-subsize 10240 ^
    --batchsize 4
















