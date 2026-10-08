#!/bin/bash
#SBATCH --job-name=train_full_ref
#SBATCH --output=train_full_ref_%A_%a.txt
#SBATCH --error=train_full_ref_%A_%a.txt
#SBATCH --array=0-29                 # 2 datasets × 3 architectures × 5 seeds
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --mem=48Gb

# Fail on error
set -euo pipefail

# sbatch runs a spooled copy of this file, so locate the repo from the
# submission directory rather than from BASH_SOURCE.
PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/slurm"
if [ -z "$PROJECT_ROOT" ] || [ ! -f "${SCRIPT_DIR}/stage_dataset.sh" ]; then
    echo "Cannot locate the repository (PROJECT_ROOT='${PROJECT_ROOT}'). Submit from inside the repo or export PROJECT_ROOT." >&2
    exit 1
fi
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${PROJECT_ROOT}/logs"
source "${SCRIPT_DIR}/result_layout.sh"

# OPTIMIZER=sgd keeps the historical trainer and directories.
# OPTIMIZER=adam trains scripts/train/reference.py into the adam/ subfolder.
# SEEDS_OVERRIDE is a space-separated list. The manuscript trains only seed 42:
#   SEEDS_OVERRIDE=42 sbatch --array=0-5 scripts/slurm/run_full_data.sh
# The default array 0-29 is 2 datasets x 3 architectures x 5 seeds.
OPTIMIZER="${OPTIMIZER:-sgd}"
VARIANT="$(layout_variant "$OPTIMIZER")"

# ── Environment ──────────────────────────────────────────────────────────────
source "${SCRIPT_DIR}/cluster_env.sh"

# ── Parameters ───────────────────────────────────────────────────────────────
DATASET_GRID="${DATASETS:-${DATASET_OVERRIDE:-cifar10 cifar100}}"
ARCHITECTURE_GRID="${ARCHITECTURES:-${ARCHITECTURE:-resnet18 vgg16 inception}}"
read -r -a DATASETS <<< "$DATASET_GRID"
read -r -a ARCHITECTURES <<< "$ARCHITECTURE_GRID"
read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-42 43 44 45 46}"
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
SCRATCH_BASE="$RESULTS_ROOT"
SWAP_ROOT="$(layout_root "$SCRATCH_BASE" data_selection_swap "$VARIANT")"
SAVE_DIR="${SWAP_ROOT}/${DATASET}/${ARCHITECTURE}/checkpoints/seed_${SEED}"

LOCAL_DATA="${SLURM_TMPDIR:-./data}/${DATASET}_data"
mkdir -p "$LOCAL_DATA"
mkdir -p "$SAVE_DIR"

RESUME_ARGS=()
if [ "${RESUME:-0}" = "1" ]; then
    RESUME_ARGS=(--resume)
fi

# ── Stage & Extract Dataset onto Node-Local SSD ($SLURM_TMPDIR) ───────────────
source "${SCRIPT_DIR}/stage_dataset.sh"
stage_dataset "$DATASET" "$LOCAL_DATA"

echo "=================================================="
echo " Starting Full Reference Trajectory Run"
echo " Dataset        : $DATASET"
echo " Architecture   : $ARCHITECTURE"
echo " Seed           : $SEED"
echo " Optimizer      : $OPTIMIZER"
echo " Local Data Dir : $LOCAL_DATA"
echo " Save Dir       : $SAVE_DIR"
echo "=================================================="

# ── Run the training script ──────────────────────────────────────────────────
if [ "$VARIANT" = adam ]; then
    srun python "${PROJECT_ROOT}/scripts/train/reference.py" \
        --dataset "$DATASET" \
        --architecture "$ARCHITECTURE" \
        --optimizer "$OPTIMIZER" \
        --data_dir "$LOCAL_DATA" \
        --save_dir "$SAVE_DIR" \
        --download \
        --batch_size 128 \
        --num_workers 4 \
        --seed "$SEED" \
        --checkpoint_schedule full_data \
        "${RESUME_ARGS[@]}"
else
    srun python "${PROJECT_ROOT}/scripts/train/full_data.py" \
        --dataset "$DATASET" \
        --architecture "$ARCHITECTURE" \
        --data_dir "$LOCAL_DATA" \
        --save_dir "$SAVE_DIR" \
        --download \
        --batch_size 128 \
        --num_workers 4 \
        --seed "$SEED" \
        "${RESUME_ARGS[@]}"
fi

echo "=================================================="
echo "Trajectory generation complete!"
echo "=================================================="
