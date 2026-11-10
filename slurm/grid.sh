#!/bin/bash
#SBATCH --job-name=hierbio_grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.err

# Methods x seeds as a job array. Task arguments are passed through to scripts/run.py.
#   METHODS="baseline hmce ftm tt ftm_tt" SEEDS="0 1 2 3 4" \
#     sbatch --array=0-24 slurm/grid.sh --embedding naturelm --dataset anuraset --label-mode multi
# --array must be 0..(#methods x #seeds - 1).
set -e
source slurm/env.sh
read -ra M <<< "${METHODS:-baseline hmce ftm tt ftm_tt flat_linear flat_mlp mlbce focal_mlbce}"
read -ra S <<< "${SEEDS:-0 1 2 3 4}"
METHOD=${M[$((SLURM_ARRAY_TASK_ID / ${#S[@]}))]}
SEED=${S[$((SLURM_ARRAY_TASK_ID % ${#S[@]}))]}
python scripts/run.py --method "$METHOD" --seed "$SEED" "$@"
