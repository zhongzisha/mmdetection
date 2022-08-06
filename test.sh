#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --account=owner-guest
#SBATCH --partition=q04
##SBATCH --gres=gpu:3
#SBATCH --nodelist=g41
##SBATCH --gres=gpu:1
##SBATCH --mem-per-gpu=30G
##SBATCH --cpus-per-task=40
##SBATCH --mem=100G
#SBATCH -o /share/home/zhongzisha/cluster_logs/mmdet-job-test-%j-%N.out
#SBATCH -e /share/home/zhongzisha/cluster_logs/mmdet-job-test-%j-%N.err

echo "job start `date`"
echo "job run at ${HOSTNAME}"
nvidia-smi

#mpirun -np $SLURM_NTASKS pw.x -inp $DATADIR/$NAME.in > $DATADIR/$NAME.$SLURM_JOB_ID.outjjjjjjjkkkkkkkkkkkkkkkkkjj:w
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

sleep 30

