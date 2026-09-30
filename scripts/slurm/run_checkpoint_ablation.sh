#!/bin/bash
#SBATCH --job-name=kl_checkpoint_ablation
#SBATCH --output=logs/kl_ckpt_ablation_%A_%a.txt
#SBATCH --error=logs/kl_ckpt_ablation_%A_%a.txt
#SBATCH --array=0-99                 # 100 tasks = 4 fractions × 5 seeds × 5 checkpoint counts
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1         # Adjust GPU type if required
#SBATCH --time=15:00:00
#SBATCH --mem=48Gb

set -euo pipefail

# sbatch runs a spooled copy of this file, so locate the repo from the
# submission directory rather than from BASH_SOURCE.
PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/slurm"
if [ -z "$PROJECT_ROOT" ] || [ ! -f "${SCRIPT_DIR}/stage_dataset.sh" ]; then
    echo "Cannot locate the repository (PROJECT_ROOT='${PROJECT_ROOT}'). Submit from inside the repo or export PROJECT_ROOT." >&2
    exit 1
fi
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${PROJECT_ROOT}/logs"

# ── Environment ──────────────────────────────────────────────────────────────
module load miniconda/3
set +u
conda activate gradmatch
set -u

# ── Experiment Hyperparameters ───────────────────────────────────────────────
DATASET="cifar100"  # Options: "imagenet" or "cifar100" (override with DATASET_OVERRIDE)
FRACTIONS=(0.1 0.2 0.4 0.5)
SEEDS=(42 43 44 45 46)
CHECKPOINTS=(1 3 5 10 20)    # --num_checkpoints values to ablate
CHECKPOINT_SELECTION="${CHECKPOINT_SELECTION:-spread}"
CHECKPOINT_WEIGHTING="${CHECKPOINT_WEIGHTING:-point}"

MAX_ITERS=2000
BATCH_SIZE=128
LR=0.01
NUM_WORKERS=4

# ── Map flat task ID → (fraction, seed, num_checkpoints) ────────────────────
# Array is 0-99. There are NUM_SEEDS*NUM_CKPTS = 25 tasks per fraction, and
# NUM_CKPTS = 5 tasks per (fraction, seed) — one per checkpoint count.
if [ -n "${DATASET_OVERRIDE:-}" ]; then
    DATASET="$DATASET_OVERRIDE"
fi

NUM_SEEDS=${#SEEDS[@]}
NUM_CKPTS=${#CHECKPOINTS[@]}
TASKS_PER_FRACTION=$(( NUM_SEEDS * NUM_CKPTS ))

FRACTION_IDX=$(( SLURM_ARRAY_TASK_ID / TASKS_PER_FRACTION ))
REMAINDER=$(( SLURM_ARRAY_TASK_ID % TASKS_PER_FRACTION ))
SEED_IDX=$(( REMAINDER / NUM_CKPTS ))
CKPT_IDX=$(( REMAINDER % NUM_CKPTS ))

FRACTION=${FRACTIONS[$FRACTION_IDX]}
SEED=${SEEDS[$SEED_IDX]}
NUM_CHECKPOINTS=${CHECKPOINTS[$CKPT_IDX]}

# ── Paths Configuration ──────────────────────────────────────────────────────
SCRATCH_BASE="${SCRATCH:-/home/mila/a/ahmedm/scratch}"
CHECKPOINT_DIR="${SCRATCH_BASE}/data_selection_swap/${DATASET}/checkpoints"
LOCAL_DATA="${SLURM_TMPDIR:-./data}/${DATASET}_data"
mkdir -p "$LOCAL_DATA"
mkdir -p "$CHECKPOINT_DIR"

# ── Stage & Extract Dataset onto Node-Local SSD ($SLURM_TMPDIR) ───────────────
source "${SCRIPT_DIR}/stage_dataset.sh"
stage_dataset "$DATASET" "$LOCAL_DATA"

echo "=================================================="
echo " SLURM Array Task ID : $SLURM_ARRAY_TASK_ID"
echo " Dataset             : $DATASET"
echo " Fraction            : $FRACTION"
echo " Seed                : $SEED"
echo " Num Checkpoints (K) : $NUM_CHECKPOINTS"
echo " Checkpoint Selection: $CHECKPOINT_SELECTION"
echo " Checkpoint Weighting: $CHECKPOINT_WEIGHTING"
echo " Checkpoint Dir      : $CHECKPOINT_DIR"
echo " Local Data Dir      : $LOCAL_DATA"
echo " Max Iterations      : $MAX_ITERS"
echo "=================================================="

echo ""
echo "--------------------------------------------------"
echo " Running KL-Faithful Selection | dataset=$DATASET | fraction=$FRACTION | seed=$SEED | K=$NUM_CHECKPOINTS"
echo "--------------------------------------------------"

srun python "${PROJECT_ROOT}/scripts/select/kl.py" \
    --dataset "$DATASET" \
    --fraction "$FRACTION" \
    --seed "$SEED" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --data_dir "$LOCAL_DATA" \
    --num_checkpoints "$NUM_CHECKPOINTS" \
    --checkpoint_selection "$CHECKPOINT_SELECTION" \
    --checkpoint_weighting "$CHECKPOINT_WEIGHTING" \
    --max_iters "$MAX_ITERS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --num_workers "$NUM_WORKERS"

echo ""
echo "=================================================="
echo " Task $SLURM_ARRAY_TASK_ID completed successfully!"
echo "=================================================="
