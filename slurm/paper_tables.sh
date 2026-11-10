#!/bin/bash
#SBATCH --job-name=paper_tables
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# Retrains every cell of paper Tables 2-4 with the paper protocol (seed 42), then prints the tables.
# Retrained heads land within about +-1 point of the published numbers (GPU non-determinism).
#   sbatch slurm/paper_tables.sh
set -e
source slurm/env.sh
P="--protocol paper --seed 42"
for EMB in naturelm birdnet; do
  for M in baseline hmce ftm tt ftm_tt; do
    python scripts/run.py $P --embedding $EMB --dataset anuraset --label-mode single --method $M   # Table 2
    python scripts/run.py $P --embedding $EMB --dataset esc50    --label-mode single --method $M   # Table 3
    # Table 4: multi-label on the full set, recordings without any call included (all-zero rows)
    python scripts/run.py $P --embedding $EMB --dataset anuraset --label-mode multi --variant unknown --method $M
  done
done
python scripts/aggregate.py --filter paper
