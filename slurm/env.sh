# Cluster environment shared by the sbatch scripts (submit from the repository root).
# Adapt the module line and the environment path to your cluster.
module load hpc-env/13.1 Python/3.11.3-GCCcore-13.1.0 2>/dev/null || true
source "${HIERBIO_VENV:-.venv}/bin/activate"
mkdir -p logs
echo "node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
