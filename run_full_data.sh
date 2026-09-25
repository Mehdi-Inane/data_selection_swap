#!/bin/bash
#SBATCH --job-name=train_full_ref
#SBATCH --output=logs/train_full_ref_%j.txt
#SBATCH --error=logs/train_full_ref_%j.txt
#SBATCH --cpus-per-task=4            
#SBATCH --gres=gpu:rtx8000:1                 
#SBATCH --time=15:00:00              
#SBATCH --mem=48Gb

# Fail on error
set -euo pipefail

mkdir -p logs

# ── Environment ──────────────────────────────────────────────────────────────
module load miniconda/3
set +u                                
conda activate gradmatch             
set -u

# ── Parameters ───────────────────────────────────────────────────────────────
DATASET="imagenet"  # Options: "imagenet" or "cifar100"
SCRATCH_BASE="${SCRATCH:-/home/mila/a/ahmedm/scratch}"
SAVE_DIR="${SCRATCH_BASE}/data_selection_swap/${DATASET}/checkpoints"

LOCAL_DATA="${SLURM_TMPDIR:-./data}/${DATASET}_data"
mkdir -p "$LOCAL_DATA"
mkdir -p "$SAVE_DIR"

# ── Stage & Extract Dataset onto Node-Local SSD ($SLURM_TMPDIR) ───────────────
source "${SLURM_SUBMIT_DIR}/stage_dataset.sh"
stage_dataset "$DATASET" "$LOCAL_DATA"

echo "=================================================="
echo " Starting Full Reference Trajectory Run"
echo " Dataset        : $DATASET"
echo " Local Data Dir : $LOCAL_DATA"
echo " Save Dir       : $SAVE_DIR"
echo "=================================================="

# ── Run the training script ──────────────────────────────────────────────────
srun python train_full_data.py \
    --dataset "$DATASET" \
    --data_dir "$LOCAL_DATA" \
    --save_dir "$SAVE_DIR" \
    --download \
    --batch_size 128 \
    --num_workers 4 \
    --seed 42

echo "=================================================="
echo "Trajectory generation complete!"
echo "=================================================="