#!/bin/bash
#SBATCH --job-name=transfer_d
#SBATCH --output=logs/transfer_d_%A_%a.txt
#SBATCH --error=logs/transfer_d_%A_%a.txt
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=48Gb

# One task per (dataset, TARGET architecture): 2 x 3 = 6 (array 0-5).
# Each task builds that target's trajectory once per seed and scores every
# source architecture's saved subsets against it, so the output is one column
# of the source-by-target D_rel matrix. Concatenate the six CSVs (or rerun with
# --target_architectures unset) to get the full matrices.
#
# Scoring knobs must match the scored tables you compare against, so they
# default to the same values run_score_subsets.sh uses.
#
#   sbatch scripts/slurm/run_transfer_d.sh
#   FRACTION=0.2 sbatch scripts/slurm/run_transfer_d.sh
#   DATASETS=cifar10 ARCHITECTURES=resnet18 sbatch --array=0 scripts/slurm/run_transfer_d.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/slurm"
if [ -z "$PROJECT_ROOT" ] || [ ! -f "${SCRIPT_DIR}/stage_dataset.sh" ]; then
    echo "Cannot locate the repository (PROJECT_ROOT='${PROJECT_ROOT}'). Submit from inside the repo or export PROJECT_ROOT." >&2
    exit 1
fi
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
source "${SCRIPT_DIR}/result_layout.sh"
OPTIMIZER="${OPTIMIZER:-sgd}"
VARIANT="$(layout_variant "$OPTIMIZER")"
if [ -z "$VARIANT" ]; then
    VARIANT_ARG=sgd
    WEIGHTING="${WEIGHTING:-sgd}"
    METRIC="${METRIC:-none}"
else
    VARIANT_ARG="$VARIANT"
    WEIGHTING="${WEIGHTING:-adam}"
    METRIC="${METRIC:-auto}"
fi

module load miniconda/3
CONDA_ENV="${CONDA_ENV:-kl_select}"
set +u
conda activate "$CONDA_ENV"
set -u

DATASET_GRID="${DATASETS:-cifar10 cifar100}"
ARCHITECTURE_GRID="${ARCHITECTURES:-resnet18 vgg16 inception}"
read -r -a DATASETS <<< "$DATASET_GRID"
read -r -a ARCHITECTURES <<< "$ARCHITECTURE_GRID"
read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-42 43 44 45 46}"
# The subsets being scored. Defaults to the same grid, giving the full square.
read -r -a SOURCES <<< "${SOURCE_ARCHITECTURES:-$ARCHITECTURE_GRID}"

N_ARCHITECTURES=${#ARCHITECTURES[@]}
TOTAL_TASKS=$(( ${#DATASETS[@]} * N_ARCHITECTURES ))
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_TASKS )); then
    echo "Task ${SLURM_ARRAY_TASK_ID} is outside configured grid 0-$((TOTAL_TASKS - 1))." >&2
    exit 1
fi
IDX=$SLURM_ARRAY_TASK_ID
TARGET=${ARCHITECTURES[$(( IDX % N_ARCHITECTURES ))]}; IDX=$(( IDX / N_ARCHITECTURES ))
DATASET=${DATASETS[$IDX]}

SCRATCH_BASE="${SCRATCH:-/home/mila/a/ahmedm/scratch}"
SWAP_ROOT="$(layout_root "$SCRATCH_BASE" data_selection_swap "$VARIANT")"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SWAP_ROOT}/${DATASET}/${TARGET}/checkpoints}"
LOCAL_DATA="${SLURM_TMPDIR:-./data}/${DATASET}_data"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Missing reference trajectory for the target architecture: $CHECKPOINT_DIR" >&2
    exit 1
fi

mkdir -p "$LOCAL_DATA" logs
source "${SCRIPT_DIR}/stage_dataset.sh"
stage_dataset "$DATASET" "$LOCAL_DATA"

FRACTION="${FRACTION:-0.1}"
echo "Transfer D: dataset=$DATASET target=$TARGET sources=${SOURCES[*]} fraction=$FRACTION"
srun python "${PROJECT_ROOT}/scripts/analyze/transfer_d.py" \
    --dataset "$DATASET" \
    --source_architectures "${SOURCES[@]}" \
    --target_architectures "$TARGET" \
    --data_dir "$LOCAL_DATA" \
    --checkpoint_dir "${CHECKPOINT_DIR}/" \
    --scratch "$SCRATCH_BASE" \
    --fractions "$FRACTION" \
    --seeds "${SEEDS[@]}" \
    --num_checkpoints "${NUM_CHECKPOINTS:-0}" \
    --checkpoint_selection "${CHECKPOINT_SELECTION:-spread}" \
    --checkpoint_weighting "${CHECKPOINT_WEIGHTING:-point}" \
    --weighting "$WEIGHTING" \
    --metric "$METRIC" \
    --variant "$VARIANT_ARG" \
    --scoring_transform "${SCORING_TRANSFORM:-test}" \
    --n_random "${N_RANDOM:-3}" \
    --gbar_cache_dir "${SWAP_ROOT}/gbar_cache" \
    --batch_size 256 \
    --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
    --download
