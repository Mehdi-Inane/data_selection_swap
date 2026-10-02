#!/bin/bash
#SBATCH --job-name=score_kernel
#SBATCH --output=logs/score_kernel_%A_%a.txt
#SBATCH --error=logs/score_kernel_%A_%a.txt
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=48Gb

# One task per (dataset, architecture): 2 x 3 = 6 (array 0-5).
# Reads the kernels written by run_precompute_kernel.sh and scores every
# saved subset. No dataset staging: P and r are already on scratch.
#
#   sbatch scripts/slurm/run_score_kernel.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
if [ -z "$PROJECT_ROOT" ] || [ ! -f "${PROJECT_ROOT}/scripts/analyze/score_kernel.py" ]; then
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

N_ARCHITECTURES=${#ARCHITECTURES[@]}
TOTAL_TASKS=$(( ${#DATASETS[@]} * N_ARCHITECTURES ))
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_TASKS )); then
    echo "Task ${SLURM_ARRAY_TASK_ID} is outside configured grid 0-$((TOTAL_TASKS - 1))." >&2
    exit 1
fi
IDX=$SLURM_ARRAY_TASK_ID
ARCHITECTURE=${ARCHITECTURES[$(( IDX % N_ARCHITECTURES ))]}; IDX=$(( IDX / N_ARCHITECTURES ))
DATASET=${DATASETS[$IDX]}

SCRATCH_BASE="${SCRATCH:-/home/mila/a/ahmedm/scratch}"
echo "Score kernel: dataset=$DATASET architecture=$ARCHITECTURE proj_dim=${PROJ_DIM:-4096}"
srun python "${PROJECT_ROOT}/scripts/analyze/score_kernel.py" \
    --dataset "$DATASET" \
    --architecture "$ARCHITECTURE" \
    --scratch "$SCRATCH_BASE" \
    --num_checkpoints "${NUM_CHECKPOINTS:-0}" \
    --checkpoint_selection "${CHECKPOINT_SELECTION:-spread}" \
    --checkpoint_weighting "${CHECKPOINT_WEIGHTING:-point}" \
    --weighting "${WEIGHTING:-sgd}" \
    --metric "${METRIC:-auto}" \
    --scoring_transform "${SCORING_TRANSFORM:-test}" \
    --proj_dim "${PROJ_DIM:-4096}" \
    --proj_seed "${PROJ_SEED:-0}"
