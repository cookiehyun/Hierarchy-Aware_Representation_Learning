#!/bin/bash
#SBATCH --job-name=esc50-triplet-train
#SBATCH --partition=mpcg.p
#SBATCH --exclude=mpcg005
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/user/gaad2403/Bachelor-Thesis/logs/single_NatureLM_esc50_tree_unseen%j.log
#SBATCH --error=/user/gaad2403/Bachelor-Thesis/logs/single_NatureLM_esc50_tree_unseen%j.err

module load hpc-env/13.1 Python/3.11.3-GCCcore-13.1.0
source /fs/dss/home/gaad2403/bachelor_thesis/naturelm-env/bin/activate

python /user/gaad2403/Bachelor-Thesis/scripts/NatureLM/esc50/tree_triplet/train_triplet_single_unseen.py