#!/bin/bash
# One task from the controlled optimizer/LR experiment.  Array bounds and
# dependencies are supplied by submit_optimizer_lr.sh rather than duplicated
# in an SBATCH header.
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DRIVER="${PROJECT_ROOT}/scripts/experiments/optimizer_lr.py"

: "${OPT_LR_CONFIG:?Set OPT_LR_CONFIG to an optimizer_lr JSON config}"
: "${OPT_LR_STAGE:?Set OPT_LR_STAGE to reference, selection, or retrain}"
: "${SLURM_ARRAY_TASK_ID:?This launcher must run as a Slurm array}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if command -v module >/dev/null 2>&1; then
    module load "${OPT_LR_MINICONDA_MODULE:-miniconda/3}"
fi
if command -v conda >/dev/null 2>&1; then
    set +u
    conda activate "${OPT_LR_CONDA_ENV:-gradmatch}"
    set -u
fi

DATASET="$(python "$DRIVER" task-field \
    --config "$OPT_LR_CONFIG" \
    --stage "$OPT_LR_STAGE" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --field dataset)"
ARCHITECTURE="$(python "$DRIVER" task-field \
    --config "$OPT_LR_CONFIG" \
    --stage "$OPT_LR_STAGE" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --field architecture)"

DOWNLOAD_ARGS=()
if [ -n "${OPT_LR_DATA_DIR:-}" ]; then
    DATA_DIR="$OPT_LR_DATA_DIR"
else
    DATA_DIR="${SLURM_TMPDIR:-${PROJECT_ROOT}/data}/${DATASET}_data"
    mkdir -p "$DATA_DIR"
    if [ "$DATASET" = "cifar100" ] || [ "$DATASET" = "imagenet" ]; then
        source "${SCRIPT_DIR}/stage_dataset.sh"
        stage_dataset "$DATASET" "$DATA_DIR"
    else
        # CIFAR-10 and MNIST are downloaded by torchvision into the job-local
        # directory unless the caller supplied OPT_LR_DATA_DIR.
        DOWNLOAD_ARGS=(--download)
    fi
fi
if [ "${OPT_LR_DOWNLOAD:-0}" = "1" ]; then
    DOWNLOAD_ARGS=(--download)
fi

OUTPUT_ARGS=()
if [ -n "${OPT_LR_OUTPUT_ROOT:-}" ]; then
    OUTPUT_ARGS=(--output-root "$OPT_LR_OUTPUT_ROOT")
fi
DRY_RUN_ARGS=()
if [ "${OPT_LR_DRY_RUN:-0}" = "1" ]; then
    DRY_RUN_ARGS=(--dry-run --allow-cpu)
fi

echo "============================================================"
echo " optimizer/LR stage : $OPT_LR_STAGE"
echo " array task          : $SLURM_ARRAY_TASK_ID"
echo " dataset             : $DATASET"
echo " architecture        : $ARCHITECTURE"
echo " data directory      : $DATA_DIR"
echo " config              : $OPT_LR_CONFIG"
echo " output root         : ${OPT_LR_OUTPUT_ROOT:-driver default}"
echo "============================================================"

srun python "$DRIVER" run-task \
    --config "$OPT_LR_CONFIG" \
    --stage "$OPT_LR_STAGE" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --data-dir "$DATA_DIR" \
    "${OUTPUT_ARGS[@]}" \
    "${DOWNLOAD_ARGS[@]}" \
    "${DRY_RUN_ARGS[@]}"

