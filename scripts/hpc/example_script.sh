#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --job-name=example
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --output=example.%j
#SBATCH --mem=16G
#SBATCH --partition=gpu

ml miniconda3
ml cuda/12.8
# you will need to create enviroment with conda first
source activate /mnt/home/lihao/lihao_project/.conda/envs/pytorch

# Example usage
EXPORT_VIDEO=true
EXPORT_MESH=true
INFER_CONFIG="./configs/infer-b.yaml"
MODEL_NAME="zxhezexin/openlrm-mix-base-1.1"
IMAGE_INPUT="./assets/sample_input/owl.png"

python -m openlrm.launch infer.lrm --infer $INFER_CONFIG model_name=$MODEL_NAME image_input=$IMAGE_INPUT export_video=$EXPORT_VIDEO export_mesh=$EXPORT_MESH
