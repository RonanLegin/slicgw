#!/bin/bash
#SBATCH -p gpu -C h100
#SBATCH --gpus-per-node=1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=24  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=128G       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-24:00:00     # DD-HH:MM:SS

module purge
source ~/envs/slicgw2/bin/activate

module load modules/2.2-20230808
module load cuda/11.8
module load cudnn/8.9.2.26


python test_inference.py \
        --config_path ../data/config.json \
        --dir_name /mnt/home/rlegin/ceph/gw/run1/inference/ \
        --checkpoint_path /mnt/home/rlegin/ceph/gw/run1/checkpoints/ 

