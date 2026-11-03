#!/bin/bash
#SBATCH --job-name=hierbio
#SBATCH --partition=mpcg.p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# One run. All arguments are passed to scripts/run.py, e.g.
#   sbatch slurm/run.sh --embedding naturelm --dataset anuraset --method ftm --seed 0
set -e
source slurm/env.sh
python scripts/run.py "$@"
