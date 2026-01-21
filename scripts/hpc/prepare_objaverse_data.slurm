#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --job-name=prepare_objaverse_data
#SBATCH --output=logs/%x_%j.out      # Standard output log (%x=job_name, %j=job_id)
#SBATCH --error=logs/%x_%j.err       # Standard error log
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks
#SBATCH --cpus-per-task=8            # CPU cores (Match this in Python script!)
#SBATCH --mem=32G                    # Memory limit
#SBATCH --gpus-per-task=1            # Number of GPUs
#SBATCH --partition=gpu

ml miniconda3
ml cuda/12.8
# you will need to create enviroment with conda first
source activate /mnt/home/lihao/lihao_project/.conda/envs/pytorch

python /mnt/home/lihao/lihao_project/OpenLRM/scripts/data/objaverse/render.py
