#!/bin/bash
#SBATCH --job-name=kl_selection
#SBATCH --output=kl_selection_%A_%a.txt
#SBATCH --error=kl_selection_%A_%a.txt
#SBATCH --array=0-119                # 2 datasets × 3 archs × 4 fractions × 5 seeds
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1         # Adjust GPU type if required
#SBATCH --time=15:00:00
#SBATCH --mem=48Gb

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

# ── Environment ──────────────────────────────────────────────────────────────
module load miniconda/3
set +u
conda activate gradmatch
set -u

# ── Experiment Hyperparameters ───────────────────────────────────────────────
DATASET_GRID="${DATASETS:-${DATASET_OVERRIDE:-cifar10 cifar100}}"
ARCHITECTURE_GRID="${ARCHITECTURES:-${ARCHITECTURE:-resnet18 vgg16 inception}}"
read -r -a DATASETS <<< "$DATASET_GRID"
read -r -a ARCHITECTURES <<< "$ARCHITECTURE_GRID"
FRACTIONS=(0.1 0.2 0.4 0.5)
SEEDS=(42 43 44 45 46)

MAX_ITERS="${MAX_ITERS:-2000}"
BATCH_SIZE=128
LR=0.01
NUM_WORKERS=4
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-4}"   # How many checkpoints build the kernel
                                          # (0 = every checkpoint found).

# Objective weighting (w_k, M_k); see data_selection_swap.trajectory.
#   uniform      w=1, M=I                        — the original runs
#   sgd          w_k = eta_k, M=I                — works off the existing ckpts
#   adam         w_k = eta_k a_k^2,              — the KL weighting; needs an
#                M_k = D_k^2                       Adam reference trajectory
#                                                  from scripts/train/reference.py, so
#                                                  point CHECKPOINT_DIR there
#   adam_update  w_k = eta_k^2 a_k^2,            — squared update mismatch,
#                M_k = D_k^2                       not a KL bound
# 'prefix' takes the first K checkpoints, i.e. epochs 1..K for
# scripts/train/full_data.py's cadence; 'spread' spaces them over the whole run.
CHECKPOINT_SELECTION="${CHECKPOINT_SELECTION:-spread}"
CHECKPOINT_WEIGHTING="${CHECKPOINT_WEIGHTING:-point}"

# A short prefix sits on an almost-flat part of the cosine schedule, so the
# default spreads checkpoints over the run. Adam needs an optimizer-aware
# trajectory; point CHECKPOINT_DIR at the output of scripts/train/reference.py.
OPTIMIZER="${OPTIMIZER:-sgd}"
METRIC="${METRIC:-auto}"          # auto | none | adam
# 'ranked' is Algorithm 3 (stops at its first rejected top-ranked pair);
# 'best' is the exact best-swap search with screening.
RULE="${RULE:-ranked}"
MAX_PAIRS="${MAX_PAIRS:-0}"       # 0 is exhaustive/certifying for --rule best
SCORING_TRANSFORM="${SCORING_TRANSFORM:-test}"
RETRAIN_OPTIMIZER="${RETRAIN_OPTIMIZER:-}"
# Shared retraining initialisation (per seed, set below once SEED is known):
# the first run writes it and the rest load it, so two selectors are compared
# from the same weights. Set INIT_CKPT_DIR="" to opt out.
INIT_CKPT_DIR="${INIT_CKPT_DIR-${SCRATCH:-./scratch}/data_selection_swap/init}"

# ── Map flat task ID → (dataset, architecture, fraction, seed) ───────────────
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

INIT_ARGS=()
if [ -n "$INIT_CKPT_DIR" ]; then
    INIT_ARGS=(--init_ckpt "${INIT_CKPT_DIR}/${DATASET}_${ARCHITECTURE}_seed${SEED}.pth")
fi

# ── Paths Configuration ──────────────────────────────────────────────────────
SCRATCH_BASE="${SCRATCH:-/home/mila/a/ahmedm/scratch}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/data_selection_swap/${DATASET}/${ARCHITECTURE}/checkpoints/seed_${SEED}}"
LOCAL_DATA="${SLURM_TMPDIR:-./data}/${DATASET}_data"
mkdir -p "$LOCAL_DATA"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Missing seed-matched reference trajectory: $CHECKPOINT_DIR" >&2
    exit 1
fi

# ── Stage & Extract Dataset onto Node-Local SSD ($SLURM_TMPDIR) ───────────────
source "${SCRIPT_DIR}/stage_dataset.sh"
stage_dataset "$DATASET" "$LOCAL_DATA"

echo "=================================================="
echo " SLURM Array Task ID : $SLURM_ARRAY_TASK_ID"
echo " Dataset             : $DATASET"
echo " Architecture        : $ARCHITECTURE"
echo " Fraction            : $FRACTION"
echo " Seed                : $SEED"
echo " Checkpoint Dir      : $CHECKPOINT_DIR"
echo " Num Checkpoints (K) : $NUM_CHECKPOINTS ($CHECKPOINT_SELECTION)"
echo " Checkpoint Weighting: $CHECKPOINT_WEIGHTING"
echo " Objective           : $OPTIMIZER (metric=$METRIC)"
echo " Swap rule           : $RULE"
echo " Max pairs           : $MAX_PAIRS"
echo " Scoring transform   : $SCORING_TRANSFORM"
echo " Shared init         : ${INIT_ARGS[1]:-none}"
echo " Local Data Dir      : $LOCAL_DATA"
echo " Max Iterations      : $MAX_ITERS"
echo "=================================================="

echo ""
echo "--------------------------------------------------"
echo " Running KL-Faithful Selection | dataset=$DATASET | fraction=$FRACTION | seed=$SEED | K=$NUM_CHECKPOINTS"
echo "--------------------------------------------------"

RETRAIN_ARGS=()
if [ -n "$RETRAIN_OPTIMIZER" ]; then
    RETRAIN_ARGS=(--retrain_optimizer "$RETRAIN_OPTIMIZER")
fi

srun python "${PROJECT_ROOT}/scripts/select/kl.py" \
    --dataset "$DATASET" \
    --architecture "$ARCHITECTURE" \
    --fraction "$FRACTION" \
    --seed "$SEED" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --data_dir "$LOCAL_DATA" \
    --num_checkpoints "$NUM_CHECKPOINTS" \
    --checkpoint_selection "$CHECKPOINT_SELECTION" \
    --checkpoint_weighting "$CHECKPOINT_WEIGHTING" \
    --optimizer "$OPTIMIZER" \
    --metric "$METRIC" \
    --rule "$RULE" \
    --max_pairs "$MAX_PAIRS" \
    --max_iters "$MAX_ITERS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --num_workers "$NUM_WORKERS" \
    --download \
    --scoring_transform "$SCORING_TRANSFORM" \
    "${RETRAIN_ARGS[@]}" \
    "${INIT_ARGS[@]}"

echo ""
echo "=================================================="
echo " Task $SLURM_ARRAY_TASK_ID completed successfully!"
echo "=================================================="
