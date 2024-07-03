#!/bin/bash

# Define the total number of simulations and number of jobs
NUM_SIMS=200
NUM_JOBS=20
SIMS_PER_JOB=$((NUM_SIMS / NUM_JOBS))

# Define other parameters
SIM_TYPES=("td_7d")
NOISE_TYPES=("real")
T_MIN_VALUES=(0.3)
WORKDIRS=("/home/ronan/scratch/slicgw/scripts/real_white_noise_both/")
IFOS=("H1" "L1")

# Loop over each working directory
for workdir in "${WORKDIRS[@]}"; do
    # Loop over instrument 
    for ifo in "${IFOS[@]}"; do
        # Loop over each simulation type
        for sim_type in "${SIM_TYPES[@]}"; do
            # Loop over each noise type
            for noise_type in "${NOISE_TYPES[@]}"; do
                # Loop over each t_min value
                for t_min in "${T_MIN_VALUES[@]}"; do
                    # Construct a unique output folder name
                    output_folder_name="${sim_type}_${noise_type}_tmin${t_min}_${ifo}/"

                    # Loop over each job
                    for (( i=0; i<NUM_JOBS; i++ )); do
                        # Calculate start and end indices for this job
                        START_INDEX=$((i * SIMS_PER_JOB + 1))
                        END_INDEX=$((START_INDEX + SIMS_PER_JOB - 1))

                        RANGE_STRING="{$START_INDEX..$END_INDEX}"
                        echo "$RANGE_STRING"
                        index=$(eval echo $RANGE_STRING)
                        echo "Indices after eval: $index"
                        # Submit the job with the index array
                        sbatch --export=ALL,sim_type="$sim_type",noise_type="$noise_type",t_min="$t_min",workdir="$workdir",output_folder_name="$output_folder_name",ifo="$ifo",index="$(eval echo $RANGE_STRING)"<<EOT
#!/bin/bash
#SBATCH --account=ctb-lplevass
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=0-016:30:00

module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
source /home/ronan/scratch/slicgw/slic_env/bin/activate

echo "Running job with the following parameters:"
echo "Sim Type: $sim_type"
echo "Noise Type: $noise_type"
echo "T Min: $t_min"
echo "Work Directory: $workdir"
echo "Output Folder: $output_folder_name"
echo "Indices: $index"

# Run your Python script with the indices
python inference_injection.py \
--sim_type "$sim_type" \
--noise_type "$noise_type" \
--score_type "slic" \
--workdir "$workdir" \
--output_folder_name "$output_folder_name" \
--t_min "$t_min" \
--step_size 0.002 \
--num_walkers 32 \
--num_steps 20000 \
--num_burn_steps 10000 \
--add_diffusion_noise \
--index_array $index \
--ifo $ifo




EOT
                    done
                done
            done
        done
    done
done