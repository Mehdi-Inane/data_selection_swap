#!/bin/bash
#SBATCH --job-name=config
#SBATCH --output=logs/%x_%A_%a.txt
#SBATCH --error=logs/%x_%A_%a.txt
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --mem=64Gb

# One task of one stage of a configs/*.yaml experiment. Submit a whole stage
# with scripts/slurm/submit_config.sh, or by hand:
#   sbatch --array=0-$((N - 1)) scripts/slurm/run_config.sh CONFIG STAGE
# where N is `python scripts/experiments/run_config.py count CONFIG STAGE`.

set -euo pipefail
if [ $# -ne 2 ]; then
    echo "Usage: sbatch --array=... $0 CONFIG STAGE" >&2
    exit 2
fi
CONFIG="$1"
STAGE="$2"

# sbatch runs a spooled copy of this file, so locate the repo from the
# submission directory rather than from BASH_SOURCE.
PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/slurm"
if [ -z "$PROJECT_ROOT" ] || [ ! -f "${SCRIPT_DIR}/stage_dataset.sh" ]; then
    echo "Cannot locate the repository (PROJECT_ROOT='${PROJECT_ROOT}'). Submit from inside the repo or export PROJECT_ROOT." >&2
    exit 1
fi
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
source "${SCRIPT_DIR}/cluster_env.sh"     # RESULTS_ROOT, optional conda environment
case "$CONFIG" in
    /*) ;;
    *) CONFIG="${PROJECT_ROOT}/${CONFIG}" ;;
esac

RUNNER=(python "${PROJECT_ROOT}/scripts/experiments/run_config.py")
TASK="${SLURM_ARRAY_TASK_ID:?run this as a Slurm array task}"
DATA_ROOT="${SLURM_TMPDIR:-${PROJECT_ROOT}/data}"
DATASET="$("${RUNNER[@]}" task-field "$CONFIG" "$STAGE" "$TASK" dataset --data-root "$DATA_ROOT")"
DATA_DIR="$("${RUNNER[@]}" task-field "$CONFIG" "$STAGE" "$TASK" data_dir --data-root "$DATA_ROOT")"

source "${SCRIPT_DIR}/stage_dataset.sh"
stage_dataset "$DATASET" "$DATA_DIR"

srun "${RUNNER[@]}" run-task "$CONFIG" "$STAGE" "$TASK" --data-root "$DATA_ROOT"
