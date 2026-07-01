#!/bin/bash
#SBATCH --job-name=esc50-ualb-train
#SBATCH --partition=mpcg.p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/user/gaad2403/Bachelor-Thesis/logs/%j.log
#SBATCH --error=/user/gaad2403/Bachelor-Thesis/logs/%j.err

module load hpc-env/13.1 Python/3.11.3-GCCcore-13.1.0
source /fs/dss/home/gaad2403/bachelor_thesis/naturelm-env/bin/activate

python /user/gaad2403/Bachelor-Thesis/scripts/birdnet/esc50/ualb/train_ualb.py