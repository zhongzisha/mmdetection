
ACTION=$1
GPUID=$2

CONFIG=mask_rcnn_r50_fpn_1x_coco
CONFIG=cascade_mask_rcnn_r50_fpn_20e_coco
# CONFIG=cascade_mask_rcnn_r50_fpn_20e_coco_noResize

if [ "$ACTION" == "train" ]; then
  echo "training ..."
  CUDA_VISIBLE_DEVICES=${GPUID} ./tools/dist_train.sh \
  configs/cell_seg/${CONFIG}.py 2 \
  --work-dir /media/ubuntu/Data/cell_seg/work_dirs/${CONFIG}
fi

if [ "$ACTION" == "test" ]; then
  echo "testing ..."
#  python tools/test_with_show_gt.py \
#  /media/ubuntu/Data/cell_seg/work_dirs/${CONFIG}/${CONFIG}.py \
#  /media/ubuntu/Data/cell_seg/work_dirs/${CONFIG}/latest.pth \
#  --show-dir /media/ubuntu/Data/cell_seg/work_dirs/${CONFIG}/val_results \
#  --json_filename data/cell_seg/coco_val.json --img_postfix .png --eval bbox segm


#  python tools/test_with_show_gt.py \
#  /media/ubuntu/Data/cell_seg/work_dirs/${CONFIG}/${CONFIG}.py \
#  /media/ubuntu/Data/cell_seg/work_dirs/${CONFIG}/latest.pth \
#  --cfg-options \
#  data.test.img_prefix="/media/ubuntu/Data/cell_seg/test/" \
#  data.test.ann_file="/media/ubuntu/Data/cell_seg/coco_test.json" \
#  --show-dir /media/ubuntu/Data/cell_seg/work_dirs/${CONFIG}/test_results \
#  --json_filename data/cell_seg/coco_test.json \
#  --img_postfix .png \
#  --eval bbox segm


CFG_OPTIONS="model.test_cfg.rpn.nms_pre=1500 "
CFG_OPTIONS+="model.test_cfg.rpn.max_per_img=1500 "
CFG_OPTIONS+="model.test_cfg.rcnn.max_per_img=1000 "
CFG_OPTIONS+="model.test_cfg.rcnn.mask_thr_binary=0.5 "
CFG_OPTIONS+="evaluation.proposal_nums=(100,500,1000) "

  CUDA_VISIBLE_DEVICES=${GPUID} python tools/test_for_cell_seg.py \
  /media/ubuntu/Data/cell_seg/work_dirs/${CONFIG}/${CONFIG}.py \
  /media/ubuntu/Data/cell_seg/work_dirs/${CONFIG}/epoch_20.pth \
  --cfg-options \
  data.test.img_prefix="/media/ubuntu/Data/cell_seg/train/" \
  data.test.ann_file="/media/ubuntu/Data/cell_seg/coco_val.json" \
  ${CFG_OPTIONS} \
  --eval bbox segm
#  --show-dir /media/ubuntu/Data/cell_seg/work_dirs/${CONFIG}/val_results \
#  --json_filename data/cell_seg/coco_val.json \
#  --img_postfix .png \

  CUDA_VISIBLE_DEVICES=${GPUID} python tools/test_for_cell_seg.py \
  /media/ubuntu/Data/cell_seg/work_dirs/${CONFIG}/${CONFIG}.py \
  /media/ubuntu/Data/cell_seg/work_dirs/${CONFIG}/epoch_20.pth \
  --cfg-options \
  data.test.img_prefix="/media/ubuntu/Data/cell_seg/test/" \
  data.test.ann_file="/media/ubuntu/Data/cell_seg/coco_test.json" \
  ${CFG_OPTIONS} \
  --show-dir /media/ubuntu/Data/cell_seg/work_dirs/${CONFIG}/test_results \
  --json_filename data/cell_seg/coco_test.json \
  --img_postfix .png \
#  --eval bbox segm

fi