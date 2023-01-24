#!/bin/bash -l

# Set Job Name
#SBATCH --job-name=hf_33_acc
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:2
#SBATCH --mail-user=jy1682@princeton.edu

BASE_DIR=/n/fs/nlp-jy1682/learnx2/hf/runs

# Set conda environment
set --
source /n/fs/nlp-jy1682/miniconda3/bin/activate;
conda activate learn;

srun accelerate launch $BASE_DIR/run_33_acc.py \
    > $BASE_DIR/logs/log_33_acc.txt