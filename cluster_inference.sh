#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-353 -p alvis
#SBATCH -t 0-03:00:00 			# time limit days-hours:minutes:seconds
#SBATCH -J d7_d_t_5
#SBATCH -o ./job_outputs/d7_d_t_5_id%j.out
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=A40:1 # GPUs 64GB of RAM; cost factor 1.0

module purge
source ~/scripts/load_env.sh
python3 $SLURM_SUBMIT_DIR/inference.py