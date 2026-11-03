# Cluster environment shared by the sbatch scripts (submit from the repository root).
module load hpc-env/13.1 Python/3.11.3-GCCcore-13.1.0
source /fs/dss/home/gaad2403/bachelor_thesis/naturelm-env/bin/activate
mkdir -p logs
echo "node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
