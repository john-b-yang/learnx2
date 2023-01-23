#!/bin/bash -l

# Set Job Name
#SBATCH --job-name=hf_32
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00    # Run for 7 days
#SBATCH --gres=gpu:1
#SBATCH --mail-user=jy1682@princeton.edu

BASE_DIR=/n/fs/nlp-jy1682/learnx2/hf/runs

# Set conda environment
set --
source /n/fs/nlp-jy1682/miniconda3/bin/activate;
conda activate learn;

srun python $BASE_DIR/run_32.py \
    > $BASE_DIR/logs/log.txt