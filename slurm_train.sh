#!/bin/bash
##SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --account=owner-guest
#SBATCH --partition=q04
##SBATCH --gres=gpu:3
#SBATCH --nodelist=gg02
#SBATCH --gres=gpu:1
##SBATCH --mem-per-gpu=30G
##SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH -o /share/home/zhongzisha/cluster_logs/mmdet-towers-%j-%N.out
#SBATCH -e /share/home/zhongzisha/cluster_logs/mmdet-towers-%j-%N.err

echo "job start `date`"
echo "job run at ${HOSTNAME}"
nvidia-smi

df -h
nvidia-smi
ls /usr/local
which nvcc
which gcc
which g++
nvcc --version
gcc --version
g++ --version

env

nvidia-smi

free -g
top -b -n 1

uname -a

sleep 500000000000000000


source /share/home/zhongzisha/venv_test/bin/activate

# the order of export is very important

export LD_LIBRARY_PATH=$HOME/glibc2.30/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/gcc-7.5.0/install/lib64:$LD_LIBRARY_PATH
export PATH=$HOME/gcc-7.5.0/install/bin:$PATH
# export CUDA_ROOT=$HOME/cuda-10.2-cudnn-7.6.5
# export LD_LIBRARY_PATH=$CUDA_ROOT/libs/lib64:$CUDA_ROOT/lib64:$CUDA_ROOT/lib64/stubs:$LD_LIBRARY_PATH
export CUDA_ROOT=$HOME/cuda-10.2-cudnn-8.2.2
export CUDA_PATH=$CUDA_ROOT
export LD_LIBRARY_PATH=$CUDA_ROOT/libs/lib64:$CUDA_ROOT/lib64:$CUDA_ROOT/lib64/stubs:$LD_LIBRARY_PATH
export PATH=$CUDA_ROOT/bin:$PATH
export CUDA_INSTALL_DIR=$HOME/cuda-10.2-cudnn-8.2.2
export CUDNN_INSTALL_DIR=$HOME/cuda-10.2-cudnn-8.2.2
export TRT_LIB_DIR=$HOME/cuda-10.2-cudnn-8.2.2/TensorRT-8.0.1.6/lib


if [ ${HOSTNAME} == "gggg" ]; then

  CONFIG=faster_rcnn_r101_fpn_4x_tower_1class_Aug_base_800_200_noSplit
  CONFIG=faster_rcnn_r101_fpn_4x_tower_1class_Aug_base_800_200
  CONFIG=mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco
  CONFIG=mask_rcnn_s101_fpn_syncbn-backbone+head_mstrain_1x_coco

  CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/cell_seg/cascade_rcnn/${CONFIG}.py 2 \
  --work-dir work_dirs/${CONFIG} \
  || exit

echo "Test"

fi

#if [ ${HOSTNAME} == "g41" ]; then
#  CONFIG=yolov3_d53_mstrain-608_273e_tower_1class_v4_augtimes5
#
#  CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/yolo/${CONFIG}.py 2 \
#  --work-dir work_dirs/${CONFIG} \
#  || exit
#fi





