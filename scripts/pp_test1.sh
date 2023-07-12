#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=128G       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-016:30     # DD-HH:MM:SS


source ~/slicgw311/bin/activate

module load cuda/11.4

python pp_test1.py
