#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-353 -p alvis
#SBATCH -t 7-00:00:00 			# time limit days-hours:minutes:seconds
#SBATCH -J update_paper_training
#SBATCH -o ../job_outputs/train_for_paper%j.out
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=A40:1 # GPUs 64GB of RAM; cost factor 1.0

module purge
source ~/scripts/load_env.sh

python3 ../scripts/belief_matching.py