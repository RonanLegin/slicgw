#!/bin/bash

OUTPUT_FOLDER_TRAIN="real_white_noise_train/"
OUTPUT_FOLDER_TEST="real_white_noise_test/"
T0=1126259462
DURATION_TRAIN=$((28*24*60*60)) 
DURATION_TEST=$((2*24*60*60)) 
IFO="H1"
SEGLEN_UPFACTOR=2
  
# Function to submit the compute job with specific SLURM directives
submit_compute_job() {
    # Ensure the heredoc starts at the beginning of the line
    sbatch <<EOF
#!/bin/bash
#SBATCH --account=def-hezaveh
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=32G       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-01:30:00     # DD-HH:MM:SS
#SBATCH --output=preprocess-%j.out  # Separate output file

# Load necessary modules or software
source /home/ronan/scratch/slicgw/slic_env/bin/activate

# Run the compute phase of the script
echo "Running training set preprocessing phase..."
# python fetch_gwosc_data.py --operation preprocess --output_folder_name $OUTPUT_FOLDER_TRAIN --t0 $T0 --duration $DURATION_TRAIN --ifo $IFO --seglen_upfactor $SEGLEN_UPFACTOR

echo "Running testing set preprocessing phase..."
python fetch_gwosc_data.py --operation preprocess --output_folder_name $OUTPUT_FOLDER_TEST --t0 $(($T0 + 2 * $DURATION_TRAIN)) --duration $DURATION_TEST --ifo $IFO --seglen_upfactor $SEGLEN_UPFACTOR --whitening_factor_directory $OUTPUT_FOLDER_TRAIN
EOF
}


# Check if the script is running on a compute node or login node

echo "Running download phase on the login node..."
source /home/ronan/scratch/slicgw/slicgwenv/bin/activate

python fetch_gwosc_data.py --operation fetch --output_folder_name $OUTPUT_FOLDER_TRAIN --t0 $T0 --duration $DURATION_TRAIN --ifo $IFO --seglen_upfactor $SEGLEN_UPFACTOR

python fetch_gwosc_data.py --operation fetch --output_folder_name $OUTPUT_FOLDER_TEST --t0 $(($T0 + 2 * $DURATION_TRAIN)) --duration $DURATION_TEST --ifo $IFO --seglen_upfactor $SEGLEN_UPFACTOR

wget -nc -i "../data/${OUTPUT_FOLDER_TRAIN}/gwosc_urls.txt" -P "../data/${OUTPUT_FOLDER_TRAIN}/raw_${IFO}/"

wget -nc -i "../data/${OUTPUT_FOLDER_TEST}/gwosc_urls.txt" -P "../data/${OUTPUT_FOLDER_TEST}/raw_${IFO}/"

echo "Download successful, submitting compute job to SLURM."
submit_compute_job