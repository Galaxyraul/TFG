#!/bin/bash
#SBATCH --job-name=FDM
#-SBATCH --qos=dgx-limit-gpu-2
#-SBATCH --time 00:00:00
#SBATCH --gres=gpu:1
#SBATCH --out outs/FDM.out
#SBATCH --error outs/FDM.err

# Caperta temporal $TMPSHM
echo $(nproc)

eval "$(conda shell.bash hook)"
conda activate ddpm
torchrun --standalone --nproc_per_node=1 train.py --outdir=training-output \
    --data=../../../data/drones_ddpm.zip --cond=0 --arch=ddpmpp \
    --precond=fdm_ve --warmup_ite=200 


