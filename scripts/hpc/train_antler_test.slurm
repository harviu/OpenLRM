#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --job-name=train_antler_test
#SBATCH --output=logs/%x_%j.out      # Standard output log (%x=job_name, %j=job_id)
#SBATCH --error=logs/%x_%j.err       # Standard error log
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks
#SBATCH --cpus-per-task=16           # CPU cores (Match this in Python script!)
#SBATCH --mem=32G                    # Memory limit
#SBATCH --gpus-per-task=2            # Number of GPUs
#SBATCH --nodelist=g[050-052]        # Use H100 GPUs
#SBATCH --partition=gpu

ml miniconda3
ml cuda/12.8
# you will need to create enviroment with conda first
source activate /mnt/home/lihao/lihao_project/.conda/envs/pytorch

# Example usage
ACC_CONFIG="./configs/accelerate-train-single.yaml"
TRAIN_CONFIG="./configs/train-antler-test.yaml"

accelerate launch --config_file $ACC_CONFIG -m openlrm.launch train.lrm --config $TRAIN_CONFIG