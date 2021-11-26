
CONFIG=mask_rcnn_r50_caffe_fpn_1x_coco
CONFIG=mask_rcnn_r50_fpn_2x_coco

echo "training ..."
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh \
configs/cell_seg/mask_rcnn/${CONFIG}.py 2 \
--work-dir ./work_dirs/${CONFIG}

echo "training ..."
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh \
configs/cell_seg/cascade_rcnn/${CONFIG}.py 2 \
--work-dir ./work_dirs/${CONFIG}


python tools/test_for_cell_seg.py \
./work_dirs/${CONFIG}/${CONFIG}.py \
./work_dirs/${CONFIG}/epoch_12.pth \
--show-dir ./work_dirs/${CONFIG}/output \
--eval bbox segm \
--img_postfix .png



#python tools/test_for_cell_seg.py \
python tools/test_for_cell_seg.py \
./work_dirs/${CONFIG}/${CONFIG}.py \
./work_dirs/${CONFIG}/epoch_12.pth \
--cfg-options \
data.test.img_prefix="data/cell_seg/" \
data.test.ann_file="data/cell_seg/annotations_val.json" \
--eval bbox segm \
--img_postfix .png \
--json_filename "data/cell_seg/annotations_val.json" \
--show-dir ./work_dirs/${CONFIG}/output


CONFIG=mask_rcnn_swin-t-p4-w7_fpn_1x_coco
echo "training ..."
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh \
configs/cell_seg/swin/${CONFIG}.py 2 \
--work-dir ./work_dirs/${CONFIG}


















