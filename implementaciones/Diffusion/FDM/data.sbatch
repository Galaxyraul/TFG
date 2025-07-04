#!/bin/bash
#SBATCH --job-name=data_ddpm
#-SBATCH --qos=dgx-limit-gpu-2
#-SBATCH --time 00:00:00
#SBATCH --gres=gpu:0
#SBATCH --out outs/data_ddpm.out
#SBATCH --error outs/data_ddpm.err

# Caperta temporal $TMPSHM
echo $(nproc)

eval "$(conda shell.bash hook)"
conda activate ddpm
python dataset_tool.py --source=/raid/rgt00024/tfg/data/drones --dest=/raid/rgt00024/tfg/data/drones_ddpm.zip --resolution=64x64


