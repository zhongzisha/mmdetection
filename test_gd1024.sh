
CONFIG=faster_rcnn_r50_fpn_1x_coco
CONFIG=cascade_rcnn_r50_fpn_1x_coco

for i in 0 1 2 3; do

CUDA_VISIBLE_DEVICES=0 \
python demo/detect_gd1024.py \
/media/ubuntu/Temp/gd/data/aerial2/${i}/ \
configs/faster_rcnn_gd_1024/${CONFIG}.py \
work_dirs/${CONFIG}/latest.pth \
--score-thr 0.25 \
--save_root /media/ubuntu/Temp/${CONFIG}/${i}/




done