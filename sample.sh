#!/bin/bash
#SBATCH -p gpu -C a100
#SBATCH --gpus-per-node=1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=32G       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-03:00:00     # DD-HH:MM:SS

module purge

source ~/envs/slicgw/bin/activate

module load cuda/11.8

export LD_LIBRARY_PATH=/mnt/sw/nix/store/24ib7yiwhzcwiry2axn4n92wq2k9k6bj-cudnn-8.9.1.23-11.8/lib:$LD_LIBRARY_PATH
                
python run_injection_noise_mc_phic_dist_tc_eta.py
