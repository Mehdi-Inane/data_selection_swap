#!/bin/bash
#SBATCH --job-name=data_selection_experiments_cifar100
#SBATCH --output=logs/cifar100_gradmatch_%A_%a.txt
#SBATCH --error=logs/cifar100_gradmatch_%A_%a.txt
#SBATCH --array=0-19                 # 20 tasks = 4 fractions × 5 seeds
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1           # V100 (sm_70) — compatible with this PyTorch build
#SBATCH --time=06:00:00
#SBATCH --mem=48Gb

set -euo pipefail
mkdir -p logs

module load miniconda/3
set +u
conda activate gradmatch
set -u

# ── Map flat task ID → (fraction, seed) ──────────────────────────────────────
# Layout: task_id = fraction_idx * 5 + seed_idx
# fraction indices: 0-3   →  tasks  0-4, 5-9, 10-14, 15-19
# seed     indices: 0-4   →  within each group of 5

FRACTIONS=(0.1 0.2 0.4 0.5)
SEEDS=(42 43 44 45 46)

FRACTION_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))

FRACTION=${FRACTIONS[$FRACTION_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=================================================="
echo " SLURM Array Task ID : $SLURM_ARRAY_TASK_ID"
echo " Fraction index      : $FRACTION_IDX  →  fraction=$FRACTION"
echo " Seed index          : $SEED_IDX      →  seed=$SEED"
echo "=================================================="

echo ""
echo "--------------------------------------------------"
echo " Running Random Baseline | fraction=$FRACTION | seed=$SEED"
echo "--------------------------------------------------"
srun python train_random_baseline.py \
    --fraction "$FRACTION" \
    -seed      "$SEED"

echo ""
echo "--------------------------------------------------"
echo " Running GradMatch | fraction=$FRACTION | seed=$SEED"
echo "--------------------------------------------------"
srun python gradmatch_selection.py \
    --fraction "$FRACTION" \
    -seed      "$SEED"

echo ""
echo "=================================================="
echo " Task $SLURM_ARRAY_TASK_ID completed successfully!"
echo "=================================================="