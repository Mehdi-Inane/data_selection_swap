#!/bin/bash
#SBATCH --job-name=selected_examples
#SBATCH --output=logs/selected_examples_%A_%a.txt
#SBATCH --error=logs/selected_examples_%A_%a.txt
#SBATCH --array=0-1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --mem=8Gb

# One task per dataset (default cifar10, cifar100; array 0-1).
# Stages that dataset onto $SLURM_TMPDIR, then draws one 8-by-6 grid per
# architecture: six images from the 10% seed-42 subset of full-gradient
# Gradient Swap at K=1 and of each baseline.
#
#   sbatch scripts/slurm/run_selected_examples.sh
#   DATASETS=cifar10 sbatch --array=0 scripts/slurm/run_selected_examples.sh
#   ARCHITECTURES=resnet18 FRACTION=0.2 SEED=43 sbatch scripts/slurm/run_selected_examples.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/slurm"
if [ -z "$PROJECT_ROOT" ] || [ ! -f "${SCRIPT_DIR}/stage_dataset.sh" ]; then
    echo "Cannot locate the repository (PROJECT_ROOT='${PROJECT_ROOT}'). Submit from inside the repo or export PROJECT_ROOT." >&2
    exit 1
fi
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${PROJECT_ROOT}/logs"
source "${SCRIPT_DIR}/result_layout.sh"
OPTIMIZER="${OPTIMIZER:-sgd}"
VARIANT="$(layout_variant "$OPTIMIZER")"
if [ -z "$VARIANT" ]; then
    VARIANT_ARG=sgd
else
    VARIANT_ARG="$VARIANT"
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

TOTAL_TASKS=${#DATASETS[@]}
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_TASKS )); then
    echo "Task ${SLURM_ARRAY_TASK_ID} is outside configured grid 0-$((TOTAL_TASKS - 1))." >&2
    exit 1
fi
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

SCRATCH_BASE="${SCRATCH:-/home/mila/a/ahmedm/scratch}"
LOCAL_DATA="${SLURM_TMPDIR:-./data}/${DATASET}_data"
FRACTION="${FRACTION:-0.1}"
SEED="${SEED:-42}"
N_EXAMPLES="${N_EXAMPLES:-6}"
EXAMPLE_SEED="${EXAMPLE_SEED:-0}"

mkdir -p "$LOCAL_DATA" logs
source "${SCRIPT_DIR}/stage_dataset.sh"
stage_dataset "$DATASET" "$LOCAL_DATA"

for ARCHITECTURE in "${ARCHITECTURES[@]}"; do
    echo "Selected examples: dataset=$DATASET architecture=$ARCHITECTURE fraction=$FRACTION seed=$SEED"
    srun python "${PROJECT_ROOT}/scripts/plot/selected_examples.py" \
        --dataset "$DATASET" \
        --architecture "$ARCHITECTURE" \
        --data_dir "$LOCAL_DATA" \
        --scratch "$SCRATCH_BASE" \
        --fraction "$FRACTION" \
        --seed "$SEED" \
        --n_examples "$N_EXAMPLES" \
        --example_seed "$EXAMPLE_SEED" \
        --variant "$VARIANT_ARG" \
        --download
done
