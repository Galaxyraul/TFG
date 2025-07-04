#!/bin/bash
#SBATCH --job-name=ddpm
#-SBATCH --qos=dgx-limit-gpu-2
#-SBATCH --time 00:00:00
#SBATCH --gres=gpu:1
#SBATCH --out outs/ddpm.out
#SBATCH --error outs/ddpm.err

# Caperta temporal $TMPSHM
echo $(nproc)

eval "$(conda shell.bash hook)"
conda activate ddpm
python -m tools.train_ddpm


sleep 30
