#!/bin/bash
#SBATCH --job-name=kl_retrain
#SBATCH --output=logs/kl_retrain_%A_%a.txt
#SBATCH --error=logs/kl_retrain_%A_%a.txt
#SBATCH --array=74,267,268,350,353,547,552,553,561
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32Gb

# Retrain-only recovery for kl_full runs that were interrupted after saving
# their selection (wall-clock kills inside retraining): kl_full.py is invoked
# with --retrain_only, which reuses the saved *_indices.pt and writes the
# missing *_model.pth / *_metrics.json under the interrupted run's cfg hash.
#
# Task ids use the SAME grid and decoding as run_kl_full_selection_checkpoints.sh
# (2 datasets x 3 architectures x 4 fractions x 5 K x 5 seeds = 600), so each id
# addresses exactly the cell the sweep would have. The default --array covers the
# nine cells whose metrics.json are missing after the Oct-12/19 timeout wave:
#   74=cifar10/resnet18/f0.4/K15/s46   267,268=cifar10/inception/f0.2/K15/s44,45
#   350,353=cifar100/resnet18/f0.4/K10/s42,45
#   547=cifar100/inception/f0.2/K10/s44 552,553=cifar100/inception/f0.4/K10/s44,45
#   561=cifar100/inception/f0.1/K15/s43

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
LR="$(layout_retrain_lr "$OPTIMIZER")"

module load miniconda/3
CONDA_ENV="${CONDA_ENV:-kl_select}"
set +u
conda activate "$CONDA_ENV"
set -u

# ── Grid (must stay identical to run_kl_full_selection_checkpoints.sh) ───────
DATASET_GRID="${DATASETS:-${DATASET_OVERRIDE:-cifar10 cifar100}}"
ARCHITECTURE_GRID="${ARCHITECTURES:-${ARCHITECTURE:-resnet18 vgg16 inception}}"
K_GRID="${K_VALUES:-1 4 10 15 20}"
read -r -a DATASETS <<< "$DATASET_GRID"
read -r -a ARCHITECTURES <<< "$ARCHITECTURE_GRID"
read -r -a K_VALUES <<< "$K_GRID"
FRACTIONS=(0.1 0.2 0.4 0.5)
SEEDS=(42 43 44 45 46)

N_SEEDS=${#SEEDS[@]}
N_FRACTIONS=${#FRACTIONS[@]}
N_KS=${#K_VALUES[@]}
N_ARCHITECTURES=${#ARCHITECTURES[@]}
TOTAL_TASKS=$(( ${#DATASETS[@]} * N_ARCHITECTURES * N_KS * N_FRACTIONS * N_SEEDS ))
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_TASKS )); then
    echo "Task ${SLURM_ARRAY_TASK_ID} is outside configured grid 0-$((TOTAL_TASKS - 1))." >&2
    exit 1
fi

# Decode the task id: seed varies fastest, then fraction, K, architecture, dataset.
IDX=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$(( IDX % N_SEEDS ))]};                 IDX=$(( IDX / N_SEEDS ))
FRACTION=${FRACTIONS[$(( IDX % N_FRACTIONS ))]};     IDX=$(( IDX / N_FRACTIONS ))
K=${K_VALUES[$(( IDX % N_KS ))]};                    IDX=$(( IDX / N_KS ))
ARCHITECTURE=${ARCHITECTURES[$(( IDX % N_ARCHITECTURES ))]}; IDX=$(( IDX / N_ARCHITECTURES ))
DATASET=${DATASETS[$IDX]}

# ── Paths ───────────────────────────────────────────────────────────────────
SCRATCH_BASE="${SCRATCH:-/home/mila/a/ahmedm/scratch}"
SWAP_ROOT="$(layout_root "$SCRATCH_BASE" data_selection_swap "$VARIANT")"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SWAP_ROOT}/${DATASET}/${ARCHITECTURE}/checkpoints/seed_${SEED}}"
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

echo "KL retrain-only: dataset=$DATASET architecture=$ARCHITECTURE fraction=$FRACTION K=$K seed=$SEED (task $SLURM_ARRAY_TASK_ID/$TOTAL_TASKS) checkpoint_dir=$CHECKPOINT_DIR"
srun python "${PROJECT_ROOT}/scripts/select/kl_full.py" \
    --dataset "$DATASET" \
    --architecture "$ARCHITECTURE" \
    --fraction "$FRACTION" \
    --seed "$SEED" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --data_dir "$LOCAL_DATA" \
    --num_checkpoints "$K" \
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
    --retrain_only \
    "${RETRAIN_ARGS[@]}" \
    "${INIT_ARGS[@]}"
