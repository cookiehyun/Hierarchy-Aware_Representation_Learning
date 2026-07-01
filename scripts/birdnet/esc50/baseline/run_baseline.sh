#!/bin/bash
#SBATCH --job-name=birdnet-esc50-probing
#SBATCH --partition=mpcg.p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/fs/dss/home/gaad2403/Bachelor-Thesis/logs/%j.log
#SBATCH --error=/fs/dss/home/gaad2403/Bachelor-Thesis/logs/%j.err

module load hpc-env/13.1 Python/3.11.3-GCCcore-13.1.0
source /fs/dss/home/gaad2403/bachelor_thesis/birdnet-env/bin/activate

cd /fs/dss/home/gaad2403/bachelor_thesis

python /user/gaad2403/Bachelor-Thesis/scripts/birdnet/esc50/baseline/baseline_probing_birdnet.py