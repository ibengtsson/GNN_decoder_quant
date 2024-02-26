#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-353 -p alvis
#SBATCH -t 3-00:00:00 			# time limit days-hours:minutes:seconds
#SBATCH -J update_paper_inference
#SBATCH -o ../job_outputs/inference_paper%j.out
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=A40:1 # GPUs 64GB of RAM; cost factor 1.0

# load modules and environment
module purge
source ~/scripts/load_env.sh

# files
FILES=($(find ../saved_models/update_paper/ -type f))
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

# send script
python3 ../scripts/inference.py -f $FILE