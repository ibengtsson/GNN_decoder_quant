#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-353 -p alvis
#SBATCH -t 0-03:00:00 			# time limit days-hours:minutes:seconds
#SBATCH -J bit_exploration
#SBATCH -o ../job_outputs/bit_exploration%j.out
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=A40:1 # GPUs 64GB of RAM; cost factor 1.0

# load modules and environment
module purge
source ~/scripts/load_env.sh

# send script
python3 ../scripts/bit_exploration.py