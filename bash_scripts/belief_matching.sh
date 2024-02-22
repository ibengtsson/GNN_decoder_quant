#!/usr/bin/env bash
#SBATCH -A C3SE2023-1-7 -p vera
#SBATCH -t 1-05:00:00                   # time limit days-hours:minutes:seconds
#SBATCH -J belief_matching
#SBATCH -o belief_matching%j.out

module purge
source ~/scripts/load_env.sh

python3 ../scripts/belief_matching.py