#!/bin/bash
#SBATCH --job-name=kl_full_selection
#SBATCH --output=logs/kl_full_selection_%A_%a.txt
#SBATCH --error=logs/kl_full_selection_%A_%a.txt
#SBATCH --array=0-119
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --mem=64Gb

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
source "${SCRIPT_DIR}/result_layout.sh"
OPTIMIZER="${OPTIMIZER:-sgd}"
VARIANT="$(layout_variant "$OPTIMIZER")"
LR="$(layout_retrain_lr "$OPTIMIZER")"

module load miniconda/3
CONDA_ENV="${CONDA_ENV:-kl_select}"
set +u
conda activate "$CONDA_ENV"
set -u

DATASET_GRID="${DATASETS:-${DATASET_OVERRIDE:-cifar10 cifar100}}"
ARCHITECTURE_GRID="${ARCHITECTURES:-${ARCHITECTURE:-resnet18 vgg16 inception}}"
read -r -a DATASETS <<< "$DATASET_GRID"
read -r -a ARCHITECTURES <<< "$ARCHITECTURE_GRID"
FRACTIONS=(0.1 0.2 0.4 0.5)
SEEDS=(42 43 44 45 46)

N_SEEDS=${#SEEDS[@]}
N_FRACTIONS=${#FRACTIONS[@]}
N_ARCHITECTURES=${#ARCHITECTURES[@]}
TOTAL_TASKS=$(( ${#DATASETS[@]} * N_ARCHITECTURES * N_FRACTIONS * N_SEEDS ))
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_TASKS )); then
    echo "Task ${SLURM_ARRAY_TASK_ID} is outside configured grid 0-$((TOTAL_TASKS - 1))." >&2
    exit 1
fi
IDX=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$(( IDX % N_SEEDS ))]}; IDX=$(( IDX / N_SEEDS ))
FRACTION=${FRACTIONS[$(( IDX % N_FRACTIONS ))]}; IDX=$(( IDX / N_FRACTIONS ))
ARCHITECTURE=${ARCHITECTURES[$(( IDX % N_ARCHITECTURES ))]}; IDX=$(( IDX / N_ARCHITECTURES ))
DATASET=${DATASETS[$IDX]}

SCRATCH_BASE="${SCRATCH:-/home/mila/a/ahmedm/scratch}"
SWAP_ROOT="$(layout_root "$SCRATCH_BASE" data_selection_swap "$VARIANT")"
if [ "$VARIANT" = adam ]; then
    CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SWAP_ROOT}/${DATASET}/${ARCHITECTURE}/checkpoints/seed_${SEED}}"
else
    CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SWAP_ROOT}/${DATASET}/${ARCHITECTURE}/checkpoints/seed_${SEED}}}"
fi
LOCAL_DATA="${SLURM_TMPDIR:-./data}/${DATASET}_data"
if [ -z "${INIT_CKPT_DIR+x}" ]; then
    INIT_CKPT_DIR="${SWAP_ROOT}/init"
fi
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Missing seed-matched reference trajectory: $CHECKPOINT_DIR" >&2
    exit 1
fi

mkdir -p "$LOCAL_DATA"
source "${SCRIPT_DIR}/stage_dataset.sh"
stage_dataset "$DATASET" "$LOCAL_DATA"

INIT_ARGS=()
if [ -n "$INIT_CKPT_DIR" ]; then
    INIT_ARGS=(--init_ckpt "${INIT_CKPT_DIR}/${DATASET}_${ARCHITECTURE}_seed${SEED}.pth")
fi
RETRAIN_ARGS=()
if [ -n "${RETRAIN_OPTIMIZER:-}" ]; then
    RETRAIN_ARGS=(--retrain_optimizer "$RETRAIN_OPTIMIZER")
fi

echo "Full-network KL: dataset=$DATASET architecture=$ARCHITECTURE fraction=$FRACTION seed=$SEED proj_dim=${PROJ_DIM:-0}"
srun python "${PROJECT_ROOT}/scripts/select/kl_full.py" \
    --dataset "$DATASET" \
    --architecture "$ARCHITECTURE" \
    --fraction "$FRACTION" \
    --seed "$SEED" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --data_dir "$LOCAL_DATA" \
    --num_checkpoints "${NUM_CHECKPOINTS:-4}" \
    --checkpoint_selection "${CHECKPOINT_SELECTION:-spread}" \
    --checkpoint_weighting "${CHECKPOINT_WEIGHTING:-point}" \
    --optimizer "$OPTIMIZER" \
    --metric "${METRIC:-auto}" \
    --max_iters "${MAX_ITERS:-200}" \
    --score_batch_size "${SCORE_BATCH_SIZE:-128}" \
    --batch_size 128 \
    --lr "$LR" \
    --num_workers 4 \
    --download \
    --scoring_transform "${SCORING_TRANSFORM:-test}" \
    --proj_dim "${PROJ_DIM:-0}" \
    --proj_seed "${PROJ_SEED:-0}" \
    "${RETRAIN_ARGS[@]}" \
    "${INIT_ARGS[@]}"
