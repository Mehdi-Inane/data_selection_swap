#!/bin/bash
#SBATCH --job-name=imagenet_kl_selection
#SBATCH --output=logs/new_kl_select_imagenet_%A_%a.txt
#SBATCH --error=logs/new_kl_select_imagenet_%A_%a.txt
#SBATCH --array=0-19                 # 20 tasks = 1 dataset (ImageNet) × 4 fractions × 5 seeds
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1         # Adjust GPU type if required
#SBATCH --time=15:00:00
#SBATCH --mem=48Gb

set -euo pipefail
mkdir -p logs

# ── Environment ──────────────────────────────────────────────────────────────
module load miniconda/3
set +u
conda activate gradmatch
set -u

# ── Experiment Hyperparameters ───────────────────────────────────────────────
DATASET="imagenet"  # Options: "imagenet" or "cifar100" (override with DATASET_OVERRIDE)
FRACTIONS=(0.1 0.2 0.4 0.5)
SEEDS=(42 43 44 45 46)

MAX_ITERS=2000
BATCH_SIZE=128
LR=0.01
NUM_WORKERS=4

# ── Map flat task ID → (fraction, seed) ─────────────────────────────
# Array is 0-19. Divide by 5 for the fraction index, modulo 5 for the seed index.
if [ -n "${DATASET_OVERRIDE:-}" ]; then
    DATASET="$DATASET_OVERRIDE"
fi

FRACTION_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))

FRACTION=${FRACTIONS[$FRACTION_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# ── Paths Configuration ──────────────────────────────────────────────────────
SCRATCH_BASE="${SCRATCH:-/home/mila/a/ahmedm/scratch}"
CHECKPOINT_DIR="${SCRATCH_BASE}/data_selection_swap/${DATASET}/checkpoints"
LOCAL_DATA="${SLURM_TMPDIR:-./data}/${DATASET}_data"
mkdir -p "$LOCAL_DATA"
mkdir -p "$CHECKPOINT_DIR"

# ── Stage & Extract Dataset onto Node-Local SSD ($SLURM_TMPDIR) ───────────────
source "${SLURM_SUBMIT_DIR}/stage_dataset.sh"
stage_dataset "$DATASET" "$LOCAL_DATA"

echo "=================================================="
echo " SLURM Array Task ID : $SLURM_ARRAY_TASK_ID"
echo " Dataset             : $DATASET"
echo " Fraction            : $FRACTION"
echo " Seed                : $SEED"
echo " Checkpoint Dir      : $CHECKPOINT_DIR"
echo " Local Data Dir      : $LOCAL_DATA"
echo " Max Iterations      : $MAX_ITERS"
echo "=================================================="

echo ""
echo "--------------------------------------------------"
echo " Running KL-Faithful Selection | dataset=$DATASET | fraction=$FRACTION | seed=$SEED"
echo "--------------------------------------------------"

srun python train_kl_selection.py \
    --dataset "$DATASET" \
    --fraction "$FRACTION" \
    --seed "$SEED" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --data_dir "$LOCAL_DATA" \
    --max_iters "$MAX_ITERS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --num_workers "$NUM_WORKERS"

echo ""
echo "=================================================="
echo " Task $SLURM_ARRAY_TASK_ID completed successfully!"
echo "=================================================="