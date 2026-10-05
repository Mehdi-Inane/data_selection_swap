#!/bin/bash
#SBATCH --job-name=kl_warm_score
#SBATCH --output=logs/kl_warm_score_%j.txt
#SBATCH --error=logs/kl_warm_score_%j.txt
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64Gb

# Score all-checkpoint G_T for the CIFAR-10 ResNet-18 fraction-0.1 warm start
# and write the before/after table and slope plot. Run after
# scripts/slurm/run_kl_full_from_baselines.sh. Baseline G_T already in the
# result-plot cache is reused; missing rows, including every warm start, are scored.

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

DATASET=cifar10
ARCHITECTURE=resnet18
FRACTION=0.1
SCRATCH_BASE="${SCRATCH:-/home/mila/a/ahmedm/scratch}"
SWAP_ROOT="$(layout_root "$SCRATCH_BASE" data_selection_swap "$VARIANT")"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SWAP_ROOT}/${DATASET}/${ARCHITECTURE}/checkpoints}"
LOCAL_DATA="${SLURM_TMPDIR:-./data}/${DATASET}_data"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Missing reference trajectory: $CHECKPOINT_DIR" >&2
    exit 1
fi

mkdir -p "$LOCAL_DATA"
source "${SCRIPT_DIR}/stage_dataset.sh"
stage_dataset "$DATASET" "$LOCAL_DATA"

echo "Warm-start report: dataset=$DATASET architecture=$ARCHITECTURE fraction=$FRACTION"
srun python "${PROJECT_ROOT}/scripts/plot/baseline_warmstart.py" \
    --dataset "$DATASET" \
    --architecture "$ARCHITECTURE" \
    --fraction "$FRACTION" \
    --data_dir "$LOCAL_DATA" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --scratch "$SCRATCH_BASE" \
    --variant "$VARIANT_ARG" \
    --download
