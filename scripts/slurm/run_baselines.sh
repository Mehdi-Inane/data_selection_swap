#!/bin/bash
#SBATCH --job-name=selection_baselines
#SBATCH --output=baselines_%A_%a.txt
#SBATCH --error=baselines_%A_%a.txt
#SBATCH --array=0-139                # 7 methods × 4 fractions × 5 seeds
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1         # match run_kl_selection.sh for comparable timing
#SBATCH --time=15:00:00
#SBATCH --mem=48Gb

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${PROJECT_ROOT}/logs"

# ── Environment ──────────────────────────────────────────────────────────────
module load miniconda/3
set +u
conda activate gradmatch
set -u

# ── Experiment Hyperparameters (identical to alg3.sh) ───────────────────────
DATASET="${DATASET_OVERRIDE:-cifar100}"
ARCHITECTURE="${ARCHITECTURE:-resnet18}"
FRACTIONS=(0.1 0.2 0.4 0.5)
SEEDS=(42 43 44 45 46)
BATCH_SIZE=128
LR=0.01
RETRAIN_OPTIMIZER="${RETRAIN_OPTIMIZER:-sgd}"
NUM_WORKERS=4
SCORING_TRANSFORM="${SCORING_TRANSFORM:-test}"   # must match the KL run
INIT_CKPT_DIR="${INIT_CKPT_DIR-${SCRATCH:-./scratch}/data_selection_swap/init}"

# Canonical (paper-protocol) configuration of each baseline. Override with e.g.
#   METHODS="herding_grad_trajectory el2n_trajectory moso_trajectory" \
#     sbatch --array=0-59 scripts/slurm/run_baselines.sh
# to run the same-information variants.
read -r -a METHODS <<< "${METHODS:-random herding moderate el2n moso graphcut facility_location}"

N_FS=$(( ${#FRACTIONS[@]} * ${#SEEDS[@]} ))
METHOD=${METHODS[$(( SLURM_ARRAY_TASK_ID / N_FS ))]}
REM=$(( SLURM_ARRAY_TASK_ID % N_FS ))
FRACTION=${FRACTIONS[$(( REM / ${#SEEDS[@]} ))]}
SEED=${SEEDS[$(( REM % ${#SEEDS[@]} ))]}

# MoSo full-network per-sample gradients are affordable on CIFAR-100 but not
# on ImageNet; fall back to the exact last-layer form there.
MOSO_GRAD_SPACE="full"
[ "$DATASET" = "imagenet" ] && MOSO_GRAD_SPACE="last_layer"

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
SCRATCH_BASE="${SCRATCH:-/home/mila/a/ahmedm/scratch}"
CHECKPOINT_DIR="${SCRATCH_BASE}/data_selection_swap/${DATASET}/${ARCHITECTURE}/checkpoints/seed_${SEED}"
LOCAL_DATA="${SLURM_TMPDIR:-./data}/${DATASET}_data"
mkdir -p "$LOCAL_DATA"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Missing seed-matched reference trajectory: $CHECKPOINT_DIR" >&2
    exit 1
fi

INIT_ARGS=()
if [ -n "$INIT_CKPT_DIR" ]; then
    INIT_ARGS=(--init_ckpt "${INIT_CKPT_DIR}/${DATASET}_${ARCHITECTURE}_seed${SEED}.pth")
fi

source "${SCRIPT_DIR}/stage_dataset.sh"
stage_dataset "$DATASET" "$LOCAL_DATA"

echo "=================================================="
echo " Task $SLURM_ARRAY_TASK_ID | method=$METHOD | dataset=$DATASET | fraction=$FRACTION | seed=$SEED"
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
