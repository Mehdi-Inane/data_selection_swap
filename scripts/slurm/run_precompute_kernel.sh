#!/bin/bash
#SBATCH --job-name=precompute_kernel
#SBATCH --output=logs/precompute_kernel_%A_%a.txt
#SBATCH --error=logs/precompute_kernel_%A_%a.txt
#SBATCH --array=0-29
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64Gb

# One task per (dataset, architecture, seed): 2 x 3 x 5 = 30 (array 0-29).
# Stages the dataset onto $SLURM_TMPDIR, then stores P and r for that seed.
# q(m) = -2 r / m is applied later by scripts/analyze/score_kernel.py,
# which does not need the images.
#
#   sbatch scripts/slurm/run_precompute_kernel.sh
#   PROJ_DIM=4096 DATASETS=cifar10 ARCHITECTURES=resnet18 sbatch --array=0-4 scripts/slurm/run_precompute_kernel.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/slurm"
if [ -z "$PROJECT_ROOT" ] || [ ! -f "${SCRIPT_DIR}/stage_dataset.sh" ]; then
    echo "Cannot locate the repository (PROJECT_ROOT='${PROJECT_ROOT}'). Submit from inside the repo or export PROJECT_ROOT." >&2
    exit 1
fi
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

module load miniconda/3
CONDA_ENV="${CONDA_ENV:-kl_select}"
set +u
conda activate "$CONDA_ENV"
set -u

DATASET_GRID="${DATASETS:-cifar10 cifar100}"
ARCHITECTURE_GRID="${ARCHITECTURES:-resnet18 vgg16 inception}"
read -r -a DATASETS <<< "$DATASET_GRID"
read -r -a ARCHITECTURES <<< "$ARCHITECTURE_GRID"
SEEDS=(42 43 44 45 46)

N_SEEDS=${#SEEDS[@]}
N_ARCHITECTURES=${#ARCHITECTURES[@]}
TOTAL_TASKS=$(( ${#DATASETS[@]} * N_ARCHITECTURES * N_SEEDS ))
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_TASKS )); then
    echo "Task ${SLURM_ARRAY_TASK_ID} is outside configured grid 0-$((TOTAL_TASKS - 1))." >&2
    exit 1
fi
IDX=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$(( IDX % N_SEEDS ))]}; IDX=$(( IDX / N_SEEDS ))
ARCHITECTURE=${ARCHITECTURES[$(( IDX % N_ARCHITECTURES ))]}; IDX=$(( IDX / N_ARCHITECTURES ))
DATASET=${DATASETS[$IDX]}

SCRATCH_BASE="${SCRATCH:-/home/mila/a/ahmedm/scratch}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/data_selection_swap/${DATASET}/${ARCHITECTURE}/checkpoints}"
LOCAL_DATA="${SLURM_TMPDIR:-./data}/${DATASET}_data"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Missing reference trajectory: $CHECKPOINT_DIR" >&2
    exit 1
fi

mkdir -p "$LOCAL_DATA" logs
source "${SCRIPT_DIR}/stage_dataset.sh"
stage_dataset "$DATASET" "$LOCAL_DATA"

echo "Precompute kernel: dataset=$DATASET architecture=$ARCHITECTURE seed=$SEED proj_dim=${PROJ_DIM:-4096}"
srun python "${PROJECT_ROOT}/scripts/analyze/precompute_kernel.py" \
    --dataset "$DATASET" \
    --architecture "$ARCHITECTURE" \
    --seed "$SEED" \
    --data_dir "$LOCAL_DATA" \
    --scratch "$SCRATCH_BASE" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --num_checkpoints "${NUM_CHECKPOINTS:-0}" \
    --checkpoint_selection "${CHECKPOINT_SELECTION:-spread}" \
    --checkpoint_weighting "${CHECKPOINT_WEIGHTING:-point}" \
    --weighting "${WEIGHTING:-sgd}" \
    --metric "${METRIC:-auto}" \
    --scoring_transform "${SCORING_TRANSFORM:-test}" \
    --proj_dim "${PROJ_DIM:-4096}" \
    --proj_seed "${PROJ_SEED:-0}" \
    --example_batch_size "${EXAMPLE_BATCH_SIZE:-8}" \
    --batch_size 128 \
    --num_workers 4 \
    --download
