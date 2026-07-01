#!/bin/bash
#SBATCH --job-name=birdnet-esc50-embed
#SBATCH --partition=mpcg.p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=/fs/dss/home/gaad2403/Bachelor-Thesis/logs/%j.log
#SBATCH --error=/fs/dss/home/gaad2403/Bachelor-Thesis/logs/%j.err

module load hpc-env/13.1 Python/3.11.3-GCCcore-13.1.0
source /fs/dss/home/gaad2403/bachelor_thesis/birdnet-env/bin/activate

cd /fs/dss/home/gaad2403/bachelor_thesis

python /user/gaad2403/Bachelor-Thesis/scripts/birdnet/esc50/baseline/extract_embeddings_birdnet.py