#!/bin/bash
#SBATCH --job-name=score_subsets
#SBATCH --output=logs/score_subsets_%A_%a.txt
#SBATCH --error=logs/score_subsets_%A_%a.txt
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=48Gb

# One task per (dataset, architecture): 2 x 3 = 6 (array 0-5).
# Scores every saved subset at one fraction with the exact all-parameter
# objective and joins the final-epoch test accuracy those runs recorded.
# The output CSV has one row per (seed, method), which is one point of the
# D-versus-accuracy scatter drawn by scripts/plot/d_accuracy_scatter.py.
#
#   sbatch scripts/slurm/run_score_subsets.sh
#   FRACTION=0.2 sbatch scripts/slurm/run_score_subsets.sh
#   DATASETS=cifar10 ARCHITECTURES=resnet18 sbatch --array=0 scripts/slurm/run_score_subsets.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/slurm"
if [ -z "$PROJECT_ROOT" ] || [ ! -f "${SCRIPT_DIR}/stage_dataset.sh" ]; then
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
read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-42 43 44 45 46}"

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
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/data_selection_swap/${DATASET}/${ARCHITECTURE}/checkpoints}"
LOCAL_DATA="${SLURM_TMPDIR:-./data}/${DATASET}_data"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Missing reference trajectory: $CHECKPOINT_DIR" >&2
    exit 1
fi

mkdir -p "$LOCAL_DATA" logs
source "${SCRIPT_DIR}/stage_dataset.sh"
stage_dataset "$DATASET" "$LOCAL_DATA"

FRACTION="${FRACTION:-0.1}"
echo "Score subsets: dataset=$DATASET architecture=$ARCHITECTURE fraction=$FRACTION"
srun python "${PROJECT_ROOT}/scripts/analyze/score_subsets.py" \
    --dataset "$DATASET" \
    --architecture "$ARCHITECTURE" \
    --data_dir "$LOCAL_DATA" \
    --checkpoint_dir "${CHECKPOINT_DIR}/" \
    --scratch "$SCRATCH_BASE" \
    --fractions "$FRACTION" \
    --seeds "${SEEDS[@]}" \
    --num_checkpoints "${NUM_CHECKPOINTS:-0}" \
    --checkpoint_selection "${CHECKPOINT_SELECTION:-spread}" \
    --checkpoint_weighting "${CHECKPOINT_WEIGHTING:-point}" \
    --weighting "${WEIGHTING:-sgd}" \
    --metric "${METRIC:-none}" \
    --scoring_transform "${SCORING_TRANSFORM:-test}" \
    --n_random "${N_RANDOM:-3}" \
    --gbar_cache_dir "${SCRATCH_BASE}/data_selection_swap/gbar_cache" \
    --batch_size 256 \
    --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
    --download
