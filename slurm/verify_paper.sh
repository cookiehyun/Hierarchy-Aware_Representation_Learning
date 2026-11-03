#!/bin/bash
#SBATCH --job-name=verify_paper
#SBATCH --partition=mpcg.p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# Re-evaluate the checkpoints behind the paper tables with the new code (--protocol paper).
# The numbers should match Tables 2-4 for checkpoints that were not retrained afterwards.
set -e
source slurm/env.sh
R=results
ev() { python scripts/run.py --protocol paper "$@"; }

T="--embedding naturelm --dataset anuraset --label-mode single"
ev $T --method baseline
ev $T --method hmce --checkpoint $R/NatureLM/anuraset/wp2_ualb_singlelabel_l0/best_model.pt
ev $T --method ftm  --checkpoint $R/NatureLM/anuraset/wp2_ftm_only_singlelabel/best_model.pt

T="--embedding naturelm --dataset anuraset --label-mode multi"
ev $T --method baseline
ev $T --method hmce   --checkpoint $R/multilabel/NatureLM/anuraset/wp4_ualb_multilabel/best_model.pt
ev $T --method ftm    --checkpoint $R/multilabel/NatureLM/anuraset/wp4_ftm_only_multilabel/best_model.pt
ev $T --method tt     --checkpoint $R/multilabel/NatureLM/anuraset/wp4_tt_only_multilabel_self/best_model.pt
ev $T --method ftm_tt --checkpoint $R/multilabel/NatureLM/anuraset/wp4_hssn_multilabel_self/best_model.pt

T="--embedding naturelm --dataset esc50"
ev $T --method baseline
ev $T --method hmce   --checkpoint $R/NatureLM/esc50/esc50_wp3_ualb_l0/best_model.pt
ev $T --method ftm    --checkpoint $R/NatureLM/esc50/esc50_wp3_ftm_only/best_model.pt
ev $T --method tt     --checkpoint $R/NatureLM/esc50/esc50_wp3_tt_only_self/best_model.pt
ev $T --method ftm_tt --checkpoint $R/NatureLM/esc50/esc50_wp3_hssn_self/best_model.pt
