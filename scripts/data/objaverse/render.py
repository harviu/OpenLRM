import sys
import json
import subprocess

# get all downloaded objects
objaverse_path = '/mnt/home/lihao/.objaverse/hf-objaverse-v1/glbs/'
# get the path for all glbs under the objaverse_path
import os
glb_paths = []
for root, dirs, files in os.walk(os.path.expanduser(objaverse_path)):
    for file in files:
        if file.endswith('.glb'):
            glb_paths.append(os.path.join(root, file))
print(f"Found {len(glb_paths)} .glb files in {objaverse_path}")

# View rendering arguments:
for glb_path in glb_paths: 
    print(f"Submitting job for {glb_path}...")
    uid = os.path.basename(glb_path).replace('.glb', '')
    # Submit a SLURM job for rendering
    job_script = f"""#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --job-name=prepare_objaverse_data
#SBATCH --output=logs/%x_%j.out      # Standard output log (%x=job_name, %j=job_id)
#SBATCH --error=logs/%x_%j.err       # Standard error log
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks
#SBATCH --cpus-per-task=4            # CPU cores (Match this in Python script!)
#SBATCH --mem=8G                    # Memory limit
#SBATCH --gpus-per-task=1            # Number of GPUs
#SBATCH --partition=gpu

ml miniconda3
ml cuda/12.8
# you will need to create enviroment with conda first
source activate /mnt/home/lihao/lihao_project/.conda/envs/pytorch

blender -b -P /mnt/home/lihao/lihao_project/OpenLRM/scripts/data/objaverse/blender_script.py -- \
    --object_path {glb_path} \
    --output_dir /mnt/home/lihao/.objaverse/views/
"""

    # Write the job script to a temporary file
    render_job_dir = '/mnt/home/lihao/lihao_project/OpenLRM/scripts/hpc/render_jobs/'
    os.makedirs(render_job_dir, exist_ok=True)
    render_job_name = render_job_dir + f'render_job_{uid}.slurm'
    with open(render_job_name, 'w') as job_file:
        job_file.write(job_script)
    subprocess.run(['sbatch', render_job_name])

    # rendering_cmd = 'blender -b -P /mnt/home/lihao/lihao_project/OpenLRM/scripts/data/objaverse/blender_script.py -- \
    #     --object_path ' + glb_path + ' \
    #     --output_dir /mnt/home/lihao/.objaverse/views/'
    
    # os.system(rendering_cmd)

# randomly select train and val uids and save to json files
num_train = int(len(glb_paths) * 0.8)
train_uids = [os.path.basename(path).replace('.glb', '') for path in glb_paths[:num_train]]
val_uids = [os.path.basename(path).replace('.glb', '') for path in glb_paths[num_train:]]
with open('/mnt/home/lihao/.objaverse/views/train_uids.json', 'w') as f:
    json.dump(train_uids, f)
with open('/mnt/home/lihao/.objaverse/views/val_uids.json', 'w') as f:
    json.dump(val_uids, f)
print(f"Saved {len(train_uids)} train uids and {len(val_uids)} val uids.")