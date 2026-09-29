#!/bin/bash
#SBATCH --job-name=train_full_ref
#SBATCH --output=logs/train_full_ref_%A_%a.txt
#SBATCH --error=logs/train_full_ref_%A_%a.txt
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=15:00:00
#SBATCH --mem=48Gb

# 6 array tasks = 2 datasets x 3 architectures:
#   0: cifar10  resnet18    3: cifar100 resnet18
#   1: cifar10  vgg16       4: cifar100 vgg16
#   2: cifar10  inception   5: cifar100 inception
# Run a subset with e.g.  sbatch --array=1,4 train_full_ref.sh
# NOTE: run `mkdir -p logs` before sbatch; SLURM opens the log file before this script starts.

# Fail on error
set -euo pipefail

mkdir -p logs

# ── Environment ──────────────────────────────────────────────────────────────
module load miniconda/3
set +u
conda activate gradmatch
set -u

# ── Parameters (derived from the array index) ────────────────────────────────
DATASETS=(cifar10 cifar100)
ARCHS=(resnet18 vgg16 inception)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
DATASET="${DATASETS[$((IDX / ${#ARCHS[@]}))]}"
ARCH="${ARCHS[$((IDX % ${#ARCHS[@]}))]}"

SCRATCH_BASE="${SCRATCH:-/home/mila/a/ahmedm/scratch}"
SAVE_DIR="${SCRATCH_BASE}/data_selection_swap/${DATASET}/${ARCH}/checkpoints"

LOCAL_DATA="${SLURM_TMPDIR:-./data}/${DATASET}_data"
mkdir -p "$LOCAL_DATA"
mkdir -p "$SAVE_DIR"

# ── Stage & Extract Dataset onto Node-Local SSD ($SLURM_TMPDIR) ───────────────
source "${SLURM_SUBMIT_DIR}/stage_dataset.sh"
stage_dataset "$DATASET" "$LOCAL_DATA"

echo "=================================================="
echo " Starting Full Reference Trajectory Run"
echo " Array task     : ${IDX}"
echo " Dataset        : $DATASET"
echo " Architecture   : $ARCH"
echo " Local Data Dir : $LOCAL_DATA"
echo " Save Dir       : $SAVE_DIR"
echo "=================================================="

# ── Run the training script ──────────────────────────────────────────────────
# --resume continues from the latest checkpoint in SAVE_DIR if the job was
# preempted or hit the time limit; just resubmit the same array index.
srun python train_full_data.py \
    --dataset "$DATASET" \
    --arch "$ARCH" \
    --data_dir "$LOCAL_DATA" \
    --save_dir "$SAVE_DIR" \
    --download \
    --batch_size 128 \
    --num_workers 4 \
    --seed 42 \
    --lr 0.01 \

echo "=================================================="
echo "Trajectory generation complete: ${DATASET} / ${ARCH}"
echo "=================================================="