RUN_TRAIN=1
RUN_TEST=1
GPU_ID=0

# CONFIG_PREFIX=faster_rcnn_r50_fpn_1x_dota1
# CONFIG_PREFIX=faster_rcnn_obb_r50_fpn_1x_dota1_lr1e-2
# CONFIG_PREFIX=faster_rcnn_h-obb_r50_fpn_1x_dota1
# CONFIG_PREFIX=faster_rcnn_RoITrans_r50_fpn_1x_dota1
CONFIG_PREFIX=$1
if [ $RUN_TRAIN -ge 1 ]; then

./tools/dist_train.sh configs/DOTA_new/${CONFIG_PREFIX}.py 2

fi 

if [ $RUN_TEST -ge 1 ]; then
CUDA_VISIBLE_DEVICES=${GPU_ID} python tools/test.py \
configs/DOTA_new/${CONFIG_PREFIX}.py \
work_dirs/${CONFIG_PREFIX}/epoch_12.pth \
--out work_dirs/${CONFIG_PREFIX}/results.pkl \
--eval bbox --subset val

fi



