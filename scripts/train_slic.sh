#!/bin/bash


CONFIGS=("../architectures/config_real_white_noise_both.json")

# # Array of working directories
WORKDIRS=("/home/ronan/scratch/slicgw/scripts/real_white_noise_both/")

# Loop over each configuration and corresponding work directory
for i in "${!WORKDIRS[@]}"; do
    # Submit a job with specific config and workdir
    sbatch <<EOT
#!/bin/bash
#SBATCH --account=ctb-lplevass
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-020:30:00

module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
source /home/ronan/scratch/slicgw/slic_env/bin/activate

python train_slic.py --config_path "${CONFIGS[$i]}" --workdir "${WORKDIRS[$i]}"
EOT
done
