#!/bin/bash
#SBATCH --job-name=esc50-naturelm-embed
#SBATCH --partition=mpcg.p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/fs/dss/home/gaad2403/Bachelor-Thesis/logs/%j.log
#SBATCH --error=/fs/dss/home/gaad2403/Bachelor-Thesis/logs/%j.err
module load hpc-env/13.1 Python/3.11.3-GCCcore-13.1.0
source /fs/dss/home/gaad2403/bachelor_thesis/naturelm-env/bin/activate

cd /fs/dss/home/gaad2403/bachelor_thesis/NatureLM-audio

python /fs/dss/home/gaad2403/Bachelor-Thesis/scripts/esc50/extract_embeddings.py