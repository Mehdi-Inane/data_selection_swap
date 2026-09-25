#!/bin/bash
#SBATCH --job-name=selection_baselines
#SBATCH --output=logs/baselines_%A_%a.txt
#SBATCH --error=logs/baselines_%A_%a.txt
#SBATCH --array=0-139                # 7 methods × 4 fractions × 5 seeds
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1         # keep identical to alg3.sh: timings are only comparable on the same GPU
#SBATCH --time=15:00:00
#SBATCH --mem=48Gb

set -euo pipefail
mkdir -p logs

# ── Environment ──────────────────────────────────────────────────────────────
module load miniconda/3
set +u
conda activate gradmatch
set -u

# ── Experiment Hyperparameters (identical to alg3.sh) ───────────────────────
DATASET="${DATASET_OVERRIDE:-cifar100}"
FRACTIONS=(0.1 0.2 0.4 0.5)
SEEDS=(42 43 44 45 46)
BATCH_SIZE=128
LR=0.01
NUM_WORKERS=4
SCORING_TRANSFORM="${SCORING_TRANSFORM:-train}"   # must match what train_kl_selection.py used

# Canonical (paper-protocol) configuration of each baseline. Override with e.g.
#   METHODS="herding_grad_trajectory el2n_trajectory moso_trajectory" sbatch --array=0-59 run_baselines.sh
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
    random)                      CMD=(train_random_selection.py) ;;
    herding)                     CMD=(train_herding_selection.py --kernel features) ;;
    herding_grad)                CMD=(train_herding_selection.py --kernel grad) ;;
    herding_grad_trajectory)     CMD=(train_herding_selection.py --kernel grad_trajectory) ;;
    moderate)                    CMD=(train_moderate_coreset_selection.py) ;;
    el2n)                        CMD=(train_el2n_selection.py --source probes --num_probes 10 --score_epoch 20) ;;
    el2n_trajectory)             CMD=(train_el2n_selection.py --source trajectory --score_epoch 20) ;;
    moso)                        CMD=(train_moso_selection.py --source surrogates --grad_space "$MOSO_GRAD_SPACE") ;;
    moso_trajectory)             CMD=(train_moso_selection.py --source trajectory --grad_space "$MOSO_GRAD_SPACE") ;;
    moso_trajectory_lastlayer)   CMD=(train_moso_selection.py --source trajectory --grad_space last_layer) ;;
    graphcut)                    CMD=(train_submodular_selection.py --function graphcut --kernel features) ;;
    graphcut_grad_trajectory)    CMD=(train_submodular_selection.py --function graphcut --kernel grad_trajectory) ;;
    facility_location)           CMD=(train_submodular_selection.py --function facility_location --kernel features) ;;
    facility_location_grad_trajectory)
                                 CMD=(train_submodular_selection.py --function facility_location --kernel grad_trajectory) ;;
    *) echo "Unknown method: $METHOD" >&2; exit 1 ;;
esac

# ── Paths Configuration ──────────────────────────────────────────────────────
SCRATCH_BASE="${SCRATCH:-/home/mila/a/ahmedm/scratch}"
CHECKPOINT_DIR="${SCRATCH_BASE}/data_selection_swap/${DATASET}/checkpoints"
LOCAL_DATA="${SLURM_TMPDIR:-./data}/${DATASET}_data"
mkdir -p "$LOCAL_DATA"

source "${SLURM_SUBMIT_DIR}/stage_dataset.sh"
stage_dataset "$DATASET" "$LOCAL_DATA"

echo "=================================================="
echo " Task $SLURM_ARRAY_TASK_ID | method=$METHOD | dataset=$DATASET | fraction=$FRACTION | seed=$SEED"
echo " Command: ${CMD[*]}"
echo "=================================================="

srun python "${CMD[@]}" \
    --dataset "$DATASET" \
    --fraction "$FRACTION" \
    --seed "$SEED" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --data_dir "$LOCAL_DATA" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --num_workers "$NUM_WORKERS" \
    --scoring_transform "$SCORING_TRANSFORM"
