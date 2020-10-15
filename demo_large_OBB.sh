CONFIG_PREFIX=$1

CUDA_VISIBLE_DEVICES=0 \
python demo_large_image_OBB.py \
configs/DOTA_new/${CONFIG_PREFIX}.py \
work_dirs/${CONFIG_PREFIX}/epoch_12.pth \
demo/P0009.jpg demo/P0009_out_OBB.png
