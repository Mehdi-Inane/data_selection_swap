#!/bin/bash
#SBATCH --job-name=kl_selection
#SBATCH --output=logs/kl_select_lrw_%A_%a.txt
#SBATCH --error=logs/kl_select_lrw_%A_%a.txt
#SBATCH --array=0-19                 # 20 tasks = 4 fractions x 5 seeds (one dataset per submission)
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1         # Adjust GPU type if required
#SBATCH --time=15:00:00
#SBATCH --mem=48Gb

# NOTE: run `mkdir -p logs` before sbatch; SLURM opens the log file before this script starts.

set -euo pipefail
mkdir -p logs

# ── Environment ──────────────────────────────────────────────────────────────
module load miniconda/3
set +u
conda activate gradmatch
set -u

# ── Experiment Hyperparameters ───────────────────────────────────────────────
DATASET="cifar100"  # Options: "imagenet" or "cifar100" (override with DATASET_OVERRIDE)
FRACTIONS=(0.1 0.2 0.4 0.5)
SEEDS=(42 43 44 45 46)

MAX_ITERS=2000
BATCH_SIZE=128
LR=0.01              # LR for retraining on the selected subset
NUM_WORKERS=4
NUM_CHECKPOINTS=0   # Use only the first K checkpoints in CHECKPOINT_DIR to build the
                     # trajectory kernel (0 = use every checkpoint found).

# Learning-rate weighting of the trajectory kernel (P_0 = sum_k eta_k G_k G_k^T).
REF_LR=0.01          # Base LR the reference trajectory was trained with (its --lr).
                     # Must match, otherwise the reconstructed cosine schedule is wrong.
LR_WEIGHTING=1       # 1 = weight checkpoints by eta_k (files get an "_lrw" suffix)
                     # 0 = previous unweighted objective (override with LR_WEIGHTING=0 sbatch ...)
LR_WEIGHTING="${LR_WEIGHTING_OVERRIDE:-$LR_WEIGHTING}"

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
# Override with CHECKPOINT_DIR=... to point at e.g. .../${DATASET}/${ARCH}/checkpoints
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/data_selection_swap/${DATASET}/checkpoints}"
LOCAL_DATA="${SLURM_TMPDIR:-./data}/${DATASET}_data"
mkdir -p "$LOCAL_DATA"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: checkpoint directory not found: $CHECKPOINT_DIR" >&2
    exit 1
fi

# ── Optional flags ───────────────────────────────────────────────────────────
EXTRA_ARGS=()
if [ "$LR_WEIGHTING" = "0" ]; then
    EXTRA_ARGS+=(--no_lr_weighting)
fi

# ── Stage & Extract Dataset onto Node-Local SSD ($SLURM_TMPDIR) ───────────────
source "${SLURM_SUBMIT_DIR}/stage_dataset.sh"
stage_dataset "$DATASET" "$LOCAL_DATA"

echo "=================================================="
echo " SLURM Array Task ID : $SLURM_ARRAY_TASK_ID"
echo " Dataset             : $DATASET"
echo " Fraction            : $FRACTION"
echo " Seed                : $SEED"
echo " Checkpoint Dir      : $CHECKPOINT_DIR"
echo " Num Checkpoints (K) : $NUM_CHECKPOINTS"
echo " LR weighting        : $LR_WEIGHTING (ref_lr=$REF_LR)"
echo " Local Data Dir      : $LOCAL_DATA"
echo " Max Iterations      : $MAX_ITERS"
echo "=================================================="

echo ""
echo "--------------------------------------------------"
echo " Running KL-Faithful Selection | dataset=$DATASET | fraction=$FRACTION | seed=$SEED | K=$NUM_CHECKPOINTS | lr_weighting=$LR_WEIGHTING"
echo "--------------------------------------------------"

srun python train_kl_selection.py \
    --dataset "$DATASET" \
    --fraction "$FRACTION" \
    --seed "$SEED" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --data_dir "$LOCAL_DATA" \
    --num_checkpoints "$NUM_CHECKPOINTS" \
    --max_iters "$MAX_ITERS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --ref_lr "$REF_LR" \
    --num_workers "$NUM_WORKERS" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo ""
echo "=================================================="
echo " Task $SLURM_ARRAY_TASK_ID completed successfully!"
echo "=================================================="