#!/bin/bash
#SBATCH --job-name=total_esc50
#SBATCH --partition=mpcg.p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/user/gaad2403/Bachelor-Thesis/logs/total_NatureLM_esc50_hf1%j.log
#SBATCH --error=/user/gaad2403/Bachelor-Thesis/logs/total_NatureLM_esc50_hf1%j.err

module load hpc-env/13.1 Python/3.11.3-GCCcore-13.1.0

source /fs/dss/home/gaad2403/bachelor_thesis/naturelm-env/bin/activate

cd /user/gaad2403/Bachelor-Thesis
python scripts/NatureLM/esc50/evaluation.py
