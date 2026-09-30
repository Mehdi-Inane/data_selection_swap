#!/bin/bash
# One array task for the paired Algorithm 2 vs Algorithm 3 experiment.
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G

set -euo pipefail

# sbatch runs a spooled copy of this file, so locate the repo from the
# submission directory rather than from BASH_SOURCE.
PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/slurm"
if [ -z "$PROJECT_ROOT" ] || [ ! -f "${SCRIPT_DIR}/stage_dataset.sh" ]; then
    echo "Cannot locate the repository (PROJECT_ROOT='${PROJECT_ROOT}'). Submit from inside the repo or export PROJECT_ROOT." >&2
    exit 1
fi
DRIVER="${PROJECT_ROOT}/scripts/experiments/compare_swap_rules.py"

: "${SWAP_RULE_CONFIG:?Set SWAP_RULE_CONFIG to a swap_rules JSON config}"
: "${SWAP_RULE_STAGE:?Set SWAP_RULE_STAGE to reference, selection, or retrain}"
: "${SLURM_ARRAY_TASK_ID:?This launcher must run as a Slurm array}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if command -v module >/dev/null 2>&1; then
    module load "${SWAP_RULE_MINICONDA_MODULE:-miniconda/3}"
fi
if command -v conda >/dev/null 2>&1; then
    set +u
    conda activate "${SWAP_RULE_CONDA_ENV:-gradmatch}"
    set -u
fi

DATASET="$(python "$DRIVER" task-field \
    --config "$SWAP_RULE_CONFIG" --stage "$SWAP_RULE_STAGE" \
    --task-id "$SLURM_ARRAY_TASK_ID" --field dataset)"
ARCHITECTURE="$(python "$DRIVER" task-field \
    --config "$SWAP_RULE_CONFIG" --stage "$SWAP_RULE_STAGE" \
    --task-id "$SLURM_ARRAY_TASK_ID" --field architecture)"

DOWNLOAD_ARGS=()
if [ -n "${SWAP_RULE_DATA_DIR:-}" ]; then
    DATA_DIR="$SWAP_RULE_DATA_DIR"
else
    DATA_DIR="${SLURM_TMPDIR:-${PROJECT_ROOT}/data}/${DATASET}_data"
    mkdir -p "$DATA_DIR"
    if [ "$DATASET" = "cifar100" ] || [ "$DATASET" = "imagenet" ]; then
        source "${SCRIPT_DIR}/stage_dataset.sh"
        stage_dataset "$DATASET" "$DATA_DIR"
    else
        DOWNLOAD_ARGS=(--download)
    fi
fi
if [ "${SWAP_RULE_DOWNLOAD:-0}" = "1" ]; then
    DOWNLOAD_ARGS=(--download)
fi

OUTPUT_ARGS=()
if [ -n "${SWAP_RULE_OUTPUT_ROOT:-}" ]; then
    OUTPUT_ARGS=(--output-root "$SWAP_RULE_OUTPUT_ROOT")
fi
DRY_RUN_ARGS=()
if [ "${SWAP_RULE_DRY_RUN:-0}" = "1" ]; then
    DRY_RUN_ARGS=(--dry-run --allow-cpu)
fi

echo "============================================================"
echo " swap-rule stage      : $SWAP_RULE_STAGE"
echo " array task          : $SLURM_ARRAY_TASK_ID"
echo " dataset             : $DATASET"
echo " architecture        : $ARCHITECTURE"
echo " data directory      : $DATA_DIR"
echo " config              : $SWAP_RULE_CONFIG"
echo " output root         : ${SWAP_RULE_OUTPUT_ROOT:-driver default}"
echo "============================================================"

srun python "$DRIVER" run-task \
    --config "$SWAP_RULE_CONFIG" \
    --stage "$SWAP_RULE_STAGE" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --data-dir "$DATA_DIR" \
    "${OUTPUT_ARGS[@]}" \
    "${DOWNLOAD_ARGS[@]}" \
    "${DRY_RUN_ARGS[@]}"

