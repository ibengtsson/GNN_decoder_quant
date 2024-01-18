#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-353 -p alvis
#SBATCH -t 5-00:00:00 			# time limit days-hours:minutes:seconds
#SBATCH -J nn_training
#SBATCH -o ../job_outputs/nn_training%j.out
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=A40:1 # GPUs 64GB of RAM; cost factor 1.0

# load modules and environment
module purge
source ~/scripts/load_env.sh

# send script
python3 ../scripts/train_nn.py -c ../decoder_config.yaml
