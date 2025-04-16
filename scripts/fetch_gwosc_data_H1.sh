#!/bin/bash

OUTPUT_FOLDER_TRAIN="real_white_noise_train/"
OUTPUT_FOLDER_TEST="real_white_noise_test/"
T0=1126259462
DURATION_TRAIN=$((28*24*60*60)) 
DURATION_TEST=$((2*24*60*60)) 
IFO="H1"
SEGLEN_UPFACTOR=2

echo "Running download phase on the login node..."
source /home/ronan/scratch/slicgw/slicgwenv/bin/activate

python fetch_gwosc_data.py --operation fetch --output_folder_name $OUTPUT_FOLDER_TRAIN --t0 $T0 --duration $DURATION_TRAIN --ifo $IFO --seglen_upfactor $SEGLEN_UPFACTOR

python fetch_gwosc_data.py --operation fetch --output_folder_name $OUTPUT_FOLDER_TEST --t0 $(($T0 + 2 * $DURATION_TRAIN)) --duration $DURATION_TEST --ifo $IFO --seglen_upfactor $SEGLEN_UPFACTOR

wget -nc -i "../data/${OUTPUT_FOLDER_TRAIN}/gwosc_urls_${IFO}.txt" -P "../data/${OUTPUT_FOLDER_TRAIN}/raw_${IFO}/"

wget -nc -i "../data/${OUTPUT_FOLDER_TEST}/gwosc_urls_${IFO}.txt" -P "../data/${OUTPUT_FOLDER_TEST}/raw_${IFO}/"

