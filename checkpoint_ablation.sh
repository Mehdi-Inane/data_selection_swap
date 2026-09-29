#!/bin/bash
#SBATCH --job-name=kl_checkpoint_ablation
#SBATCH --output=logs/kl_ckpt_ablation_%A_%a.txt
#SBATCH --error=logs/kl_ckpt_ablation_%A_%a.txt
#SBATCH --array=0-599                # 6 (dataset,arch) combos × 4 fractions × 5 seeds × 5 checkpoint counts
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
DATASETS=(cifar10 cifar100)
ARCHS=(resnet18 vgg16 inception)
FRACTIONS=(0.1 0.2 0.4 0.5)
SEEDS=(42 43 44 45 46)
CHECKPOINTS=(1 3 5 10 20)    # --num_checkpoints values to ablate: use only the
                             # first K checkpoints (indexed in CHECKPOINT_DIR)

MAX_ITERS=2000
BATCH_SIZE=128
LR=0.01
NUM_WORKERS=4

# ── Map flat task ID → (dataset, arch, fraction, seed, num_checkpoints) ─────
# Per (dataset,arch): 4 fractions × 5 seeds × 5 checkpoint counts = 100 tasks.

NUM_SEEDS=${#SEEDS[@]}
NUM_CKPTS=${#CHECKPOINTS[@]}
TASKS_PER_FRACTION=$(( NUM_SEEDS * NUM_CKPTS ))
TASKS_PER_COMBO=$(( ${#FRACTIONS[@]} * TASKS_PER_FRACTION ))

IDX="${SLURM_ARRAY_TASK_ID:-0}"
COMBO_IDX=$(( IDX / TASKS_PER_COMBO ))
COMBO_REM=$(( IDX % TASKS_PER_COMBO ))
FRACTION_IDX=$(( COMBO_REM / TASKS_PER_FRACTION ))
REMAINDER=$(( COMBO_REM % TASKS_PER_FRACTION ))
SEED_IDX=$(( REMAINDER / NUM_CKPTS ))
CKPT_IDX=$(( REMAINDER % NUM_CKPTS ))

DATASET="${DATASETS[$(( COMBO_IDX / ${#ARCHS[@]} ))]}"
ARCH="${ARCHS[$(( COMBO_IDX % ${#ARCHS[@]} ))]}"
FRACTION=${FRACTIONS[$FRACTION_IDX]}
SEED=${SEEDS[$SEED_IDX]}
NUM_CHECKPOINTS=${CHECKPOINTS[$CKPT_IDX]}
if [ -n "${DATASET_OVERRIDE:-}" ]; then DATASET="$DATASET_OVERRIDE"; fi
if [ -n "${ARCH_OVERRIDE:-}" ]; then ARCH="$ARCH_OVERRIDE"; fi

# ── Paths Configuration ──────────────────────────────────────────────────────
SCRATCH_BASE="${SCRATCH:-/home/mila/a/ahmedm/scratch}"
CHECKPOINT_DIR="${SCRATCH_BASE}/data_selection_swap/${DATASET}/${ARCH}/checkpoints"
LOCAL_DATA="${SLURM_TMPDIR:-./data}/${DATASET}_data"
mkdir -p "$LOCAL_DATA"
mkdir -p "$CHECKPOINT_DIR"

# ── Stage & Extract Dataset onto Node-Local SSD ($SLURM_TMPDIR) ───────────────
source "${SLURM_SUBMIT_DIR}/stage_dataset.sh"
stage_dataset "$DATASET" "$LOCAL_DATA"

echo "=================================================="
echo " SLURM Array Task ID : $SLURM_ARRAY_TASK_ID"
echo " Dataset             : $DATASET"
echo " Architecture        : $ARCH"
echo " Fraction            : $FRACTION"
echo " Seed                : $SEED"
echo " Num Checkpoints (K) : $NUM_CHECKPOINTS"
echo " Checkpoint Dir      : $CHECKPOINT_DIR"
echo " Local Data Dir      : $LOCAL_DATA"
echo " Max Iterations      : $MAX_ITERS"
echo "=================================================="

echo ""
echo "--------------------------------------------------"
echo " Running KL-Faithful Selection | dataset=$DATASET | fraction=$FRACTION | seed=$SEED | K=$NUM_CHECKPOINTS"
echo "--------------------------------------------------"

srun python train_kl_selection.py \
    --dataset "$DATASET" \
    --arch "$ARCH" \
    --fraction "$FRACTION" \
    --seed "$SEED" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --data_dir "$LOCAL_DATA" \
    --num_checkpoints "$NUM_CHECKPOINTS" \
    --max_iters "$MAX_ITERS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --num_workers "$NUM_WORKERS"

echo ""
echo "=================================================="
echo " Task $SLURM_ARRAY_TASK_ID completed successfully!"
echo "=================================================="