#!/bin/bash
#SBATCH --job-name=eval
#-SBATCH --qos=dgx-limit-gpu-2
#-SBATCH --time 00:00:00
#SBATCH --gres=gpu:1
#SBATCH --out outs/eval.out
#SBATCH --error outs/eval.err

# Caperta temporal $TMPSHM
echo $(nproc)

eval "$(conda shell.bash hook)"
conda activate ddpm
# Generate 50000 images 
torchrun --standalone --nproc_per_node=1 generate.py --outdir=drones_VE\
    --seeds=0-49999 --network=/raid/rgt00024/tfg/from_scratch/ddpm/FDM-main/training-output/00009-drones_ddpm-uncond-ddpmpp-fdm_ve-gpus1-batch16-fp32/network-snapshot-010000.pkl



