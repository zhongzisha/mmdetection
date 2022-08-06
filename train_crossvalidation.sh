
for i in 0 1 2 3 4; do
unlink data/towers
ln -sf /media/ubuntu/Data/tower_detection/tower_detection_crossvalidation/data_$i data/towers

./tools/dist_train.sh \
configs/faster_rcnn/faster_rcnn_r101_fpn_4x_tower_1class_Aug.py 2 \
--work-dir /media/ubuntu/Data/tower_detection/work_dirs/faster_rcnn_r101_fpn_4x_tower_1class_Aug_lr0.01_cv$i

done