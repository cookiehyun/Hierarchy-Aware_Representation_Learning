#!/bin/bash
#SBATCH --job-name=verify_paper_multi
#SBATCH --partition=mpcg.p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# Paper Table 4 was produced on the negative-inclusive multi-label set (--variant unknown).
set -e
source slurm/env.sh
for EMB in naturelm birdnet; do
  D=$([ $EMB = naturelm ] && echo NatureLM || echo birdnet)
  R=results/multilabel/$D/anuraset
  T="--protocol paper --embedding $EMB --dataset anuraset --label-mode multi --variant unknown"
  python scripts/run.py $T --method baseline
  python scripts/run.py $T --method hmce   --checkpoint $R/wp4_ualb_multilabel_unknown/best_model.pt
  python scripts/run.py $T --method ftm    --checkpoint $R/wp4_ftm_only_multilabel_unknown/best_model.pt
  python scripts/run.py $T --method tt     --checkpoint $R/wp4_tt_only_multilabel_self_unknown/best_model.pt
  python scripts/run.py $T --method ftm_tt --checkpoint $R/wp4_hssn_multilabel_self_unknown/best_model.pt
done
