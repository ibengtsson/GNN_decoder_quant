#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-353 -p alvis
#SBATCH -t 0-03:00:00 			# time limit days-hours:minutes:seconds
#SBATCH -J inference_d7_d_t_11
#SBATCH -o ../job_outputs/inference_d7_d_t_11_id%j.out
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=A40:1 # GPUs 64GB of RAM; cost factor 1.0

module purge
source ~/scripts/load_env.sh
python3 ../scripts/inference.py -f "../saved_models/d7_d_t_11_240109-114444_load_f_d7_d_t_11_240106-153206_load_f_d7_d_t_11_240102-115304.pt"
