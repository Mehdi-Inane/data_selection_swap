#!/bin/bash
#SBATCH --job-name=selection_baselines
#SBATCH --output=logs/baselines_%A_%a.txt
#SBATCH --error=logs/baselines_%A_%a.txt
#SBATCH --array=0-839                # 2 datasets × 3 archs × 7 methods × 4 fractions × 5 seeds
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1         # match run_kl_selection.sh for comparable timing
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
source "${SCRIPT_DIR}/result_layout.sh"

# OPTIMIZER=adam retrains with Adam (lr 0.001 unless LR is set) and writes
# under the adam/ subfolders. The reference trajectory is the matching tree.
OPTIMIZER="${OPTIMIZER:-sgd}"
VARIANT="$(layout_variant "$OPTIMIZER")"

# ── Environment ──────────────────────────────────────────────────────────────
source "${SCRIPT_DIR}/cluster_env.sh"

# ── Experiment Hyperparameters (identical to alg3.sh) ───────────────────────
DATASET_GRID="${DATASETS:-${DATASET_OVERRIDE:-cifar10 cifar100}}"
ARCHITECTURE_GRID="${ARCHITECTURES:-${ARCHITECTURE:-resnet18 vgg16 inception}}"
read -r -a DATASETS <<< "$DATASET_GRID"
read -r -a ARCHITECTURES <<< "$ARCHITECTURE_GRID"
FRACTIONS=(0.1 0.2 0.4 0.5)
SEEDS=(42 43 44 45 46)
BATCH_SIZE=128
if [ "$VARIANT" = adam ]; then
    RETRAIN_OPTIMIZER="${RETRAIN_OPTIMIZER:-$OPTIMIZER}"
else
    RETRAIN_OPTIMIZER="${RETRAIN_OPTIMIZER:-sgd}"
fi
LR="$(layout_retrain_lr "$RETRAIN_OPTIMIZER")"
NUM_WORKERS=4
SCORING_TRANSFORM="${SCORING_TRANSFORM:-test}"   # must match the KL run

# Canonical (paper-protocol) configuration of each baseline. Override with e.g.
#   METHODS="herding_grad_trajectory el2n_trajectory moso_trajectory" \
#     sbatch --array=0-59 scripts/slurm/run_baselines.sh
# to run the same-information variants.
read -r -a METHODS <<< "${METHODS:-random herding moderate el2n moso graphcut facility_location}"

N_SEEDS=${#SEEDS[@]}
N_FRACTIONS=${#FRACTIONS[@]}
N_METHODS=${#METHODS[@]}
N_ARCHITECTURES=${#ARCHITECTURES[@]}
TOTAL_TASKS=$(( ${#DATASETS[@]} * N_ARCHITECTURES * N_METHODS * N_FRACTIONS * N_SEEDS ))
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_TASKS )); then
    echo "Task ${SLURM_ARRAY_TASK_ID} is outside configured grid 0-$((TOTAL_TASKS - 1))." >&2
    exit 1
fi

IDX=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$(( IDX % N_SEEDS ))]}; IDX=$(( IDX / N_SEEDS ))
FRACTION=${FRACTIONS[$(( IDX % N_FRACTIONS ))]}; IDX=$(( IDX / N_FRACTIONS ))
METHOD=${METHODS[$(( IDX % N_METHODS ))]}; IDX=$(( IDX / N_METHODS ))
ARCHITECTURE=${ARCHITECTURES[$(( IDX % N_ARCHITECTURES ))]}; IDX=$(( IDX / N_ARCHITECTURES ))
DATASET=${DATASETS[$IDX]}

MOSO_GRAD_SPACE="full"

case "$METHOD" in
    random)                      CMD=("${PROJECT_ROOT}/scripts/select/random_selection.py") ;;
    herding)                     CMD=("${PROJECT_ROOT}/scripts/select/herding.py" --kernel features) ;;
    herding_grad)                CMD=("${PROJECT_ROOT}/scripts/select/herding.py" --kernel grad) ;;
    herding_grad_trajectory)     CMD=("${PROJECT_ROOT}/scripts/select/herding.py" --kernel grad_trajectory) ;;
    moderate)                    CMD=("${PROJECT_ROOT}/scripts/select/moderate.py") ;;
    el2n)                        CMD=("${PROJECT_ROOT}/scripts/select/el2n.py" --source probes --num_probes 10 --score_epoch 20) ;;
    el2n_trajectory)             CMD=("${PROJECT_ROOT}/scripts/select/el2n.py" --source trajectory --score_epoch 20) ;;
    moso)                        CMD=("${PROJECT_ROOT}/scripts/select/moso.py" --source surrogates --grad_space "$MOSO_GRAD_SPACE") ;;
    moso_trajectory)             CMD=("${PROJECT_ROOT}/scripts/select/moso.py" --source trajectory --grad_space "$MOSO_GRAD_SPACE") ;;
    moso_trajectory_lastlayer)   CMD=("${PROJECT_ROOT}/scripts/select/moso.py" --source trajectory --grad_space last_layer) ;;
    graphcut)                    CMD=("${PROJECT_ROOT}/scripts/select/submodular.py" --function graphcut --kernel features) ;;
    graphcut_grad_trajectory)    CMD=("${PROJECT_ROOT}/scripts/select/submodular.py" --function graphcut --kernel grad_trajectory) ;;
    facility_location)           CMD=("${PROJECT_ROOT}/scripts/select/submodular.py" --function facility_location --kernel features) ;;
    facility_location_grad_trajectory)
                                 CMD=("${PROJECT_ROOT}/scripts/select/submodular.py" --function facility_location --kernel grad_trajectory) ;;
    *) echo "Unknown method: $METHOD" >&2; exit 1 ;;
esac

# ── Paths Configuration ──────────────────────────────────────────────────────
SCRATCH_BASE="$RESULTS_ROOT"
SWAP_ROOT="$(layout_root "$SCRATCH_BASE" data_selection_swap "$VARIANT")"
BASE_ROOT="$(layout_root "$SCRATCH_BASE" data_selection_baselines "$VARIANT")"
if [ -z "${INIT_CKPT_DIR+x}" ]; then
    INIT_CKPT_DIR="${SWAP_ROOT}/init"
fi
case "$DATASET" in
    cifar10|cifar100) N_FULL=50000 ;;
    mnist)            N_FULL=60000 ;;
    imagenet)         N_FULL=1281167 ;;
    *) echo "Cannot derive the training-pool size for dataset=$DATASET" >&2; exit 1 ;;
esac
N_TRAIN=$(( N_FULL - N_FULL / 10 ))
BUDGET=$(python -c "print(int(float('${FRACTION}') * ${N_TRAIN}))")
RESULT_DIR="${BASE_ROOT}/${METHOD}/${DATASET}/${ARCHITECTURE}/${BUDGET}/seed_${SEED}"
RESULT_MODEL="${RESULT_DIR}/${METHOD}_${BUDGET}_model.pth"
RESULT_METRICS="${RESULT_DIR}/${METHOD}_${BUDGET}_metrics.json"
if [ -s "$RESULT_MODEL" ] && [ -s "$RESULT_METRICS" ]; then
    echo "Skipping completed run: $RESULT_DIR"
    exit 0
fi

if [ "$VARIANT" = adam ]; then
    CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SWAP_ROOT}/${DATASET}/${ARCHITECTURE}/checkpoints/seed_${SEED}}"
else
    # Appendix G.1: one seed-42 reference trajectory for every selection seed.
    CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SWAP_ROOT}/${DATASET}/${ARCHITECTURE}/checkpoints/seed_42}"
fi
LOCAL_DATA="${SLURM_TMPDIR:-./data}/${DATASET}_data"
mkdir -p "$LOCAL_DATA"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Missing reference trajectory: $CHECKPOINT_DIR" >&2
    exit 1
fi

INIT_ARGS=()
if [ -n "$INIT_CKPT_DIR" ]; then
    INIT_ARGS=(--init_ckpt "${INIT_CKPT_DIR}/${DATASET}_${ARCHITECTURE}_seed${SEED}.pth")
fi

source "${SCRIPT_DIR}/stage_dataset.sh"
stage_dataset "$DATASET" "$LOCAL_DATA"

echo "=================================================="
echo " Task $SLURM_ARRAY_TASK_ID | method=$METHOD | dataset=$DATASET | architecture=$ARCHITECTURE | fraction=$FRACTION | seed=$SEED"
echo " Command: ${CMD[*]}"
echo "=================================================="

srun python "${CMD[@]}" \
    --dataset "$DATASET" \
    --architecture "$ARCHITECTURE" \
    --fraction "$FRACTION" \
    --seed "$SEED" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --data_dir "$LOCAL_DATA" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --retrain_optimizer "$RETRAIN_OPTIMIZER" \
    --num_workers "$NUM_WORKERS" \
    --download \
    --scoring_transform "$SCORING_TRANSFORM" \
    "${INIT_ARGS[@]}"
