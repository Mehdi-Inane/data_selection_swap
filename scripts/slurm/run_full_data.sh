#!/bin/bash
#SBATCH --job-name=train_full_ref
#SBATCH --output=train_full_ref_%A_%a.txt
#SBATCH --error=train_full_ref_%A_%a.txt
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=15:00:00
#SBATCH --mem=48Gb

# Fail on error
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${PROJECT_ROOT}/logs"

# ── Environment ──────────────────────────────────────────────────────────────
module load miniconda/3
set +u
conda activate gradmatch
set -u

# ── Parameters ───────────────────────────────────────────────────────────────
DATASET="${DATASET_OVERRIDE:-cifar100}"
ARCHITECTURE="${ARCHITECTURE:-resnet18}"
SEEDS=(42 43 44 45 46)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
SCRATCH_BASE="${SCRATCH:-/home/mila/a/ahmedm/scratch}"
SAVE_DIR="${SCRATCH_BASE}/data_selection_swap/${DATASET}/${ARCHITECTURE}/checkpoints/seed_${SEED}"

LOCAL_DATA="${SLURM_TMPDIR:-./data}/${DATASET}_data"
mkdir -p "$LOCAL_DATA"
mkdir -p "$SAVE_DIR"

RESUME_ARGS=()
if [ "${RESUME:-0}" = "1" ]; then
    RESUME_ARGS=(--resume)
fi

# ── Stage & Extract Dataset onto Node-Local SSD ($SLURM_TMPDIR) ───────────────
source "${SCRIPT_DIR}/stage_dataset.sh"
stage_dataset "$DATASET" "$LOCAL_DATA"

echo "=================================================="
echo " Starting Full Reference Trajectory Run"
echo " Dataset        : $DATASET"
echo " Architecture   : $ARCHITECTURE"
echo " Seed           : $SEED"
echo " Local Data Dir : $LOCAL_DATA"
echo " Save Dir       : $SAVE_DIR"
echo "=================================================="

# ── Run the training script ──────────────────────────────────────────────────
srun python "${PROJECT_ROOT}/scripts/train/full_data.py" \
    --dataset "$DATASET" \
    --architecture "$ARCHITECTURE" \
    --data_dir "$LOCAL_DATA" \
    --save_dir "$SAVE_DIR" \
    --download \
    --batch_size 128 \
    --num_workers 4 \
    --seed "$SEED" \
    "${RESUME_ARGS[@]}"

echo "=================================================="
echo "Trajectory generation complete!"
echo "=================================================="
