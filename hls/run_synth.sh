#!/usr/bin/env bash
#SBATCH -A C3SE2024-1-4 -p vera
#SBATCH -t 1-12:00:00 			# time limit days-hours:minutes:seconds
#SBATCH -J synth
#SBATCH -o synth%j.out

apptainer exec /cephyr/NOBACKUP/groups/snic2021-23-319/isakbe/apps/vivado_hls4ml.sif /bin/bash -c "conda run -n hls python custom_hls4ml_layer.py"
