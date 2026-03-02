#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --job-name=antler_batch
#SBATCH --output=logs/antler_%A_%a.out
#SBATCH --error=logs/antler_%A_%a.err
#SBATCH --array=0-60 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=gpu

mkdir -p logs

# Define the list of antler IDs
ANTLER_LIST=(1 2 3 4 5 6 7 8 9 11 12 13 16 17 18 19 20 21 22 23 24 26 28 29 30 31 32 33 34 35 36 37 >

# Get the specific ID
CURRENT_ID=${ANTLER_LIST[$SLURM_ARRAY_TASK_ID]}
FORMATTED_ID="antler_${CURRENT_ID}"

echo "=========================================="
echo "Processing: $FORMATTED_ID"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "=========================================="

# Load modules
ml miniconda3
ml cuda/12.8

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate antler_env

# Set paths
WORKING_DIR="$HOME/antler_project/Antler_Photo_Dataset"
OUTPUT_DIR="$HOME/antler_project/antler_test"

# Run the python script for specific antler
python prep_antler_data.py \
    --working_dir "$WORKING_DIR" \
    --view_dir "$OUTPUT_DIR" \
    --antler_ids "$FORMATTED_ID" \
    --save_debug

echo "=========================================="
echo "Finished $FORMATTED_ID"
echo "=========================================="