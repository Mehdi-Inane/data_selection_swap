#!/bin/bash
#SBATCH --job-name=d_alpha_corr
#SBATCH --output=d_alpha_corr_%A_%a.out
#SBATCH --error=d_alpha_corr_%A_%a.err
#SBATCH --array=0-29
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=64G

set -euo pipefail

# sbatch runs a spooled copy of this file, so locate the repo from the
# submission directory rather than from BASH_SOURCE.
PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/slurm"
if [ -z "$PROJECT_ROOT" ] || [ ! -f "${SCRIPT_DIR}/stage_dataset.sh" ]; then
    echo "Cannot locate the repository (PROJECT_ROOT='${PROJECT_ROOT}'). Submit from inside the repo or export PROJECT_ROOT." >&2
    exit 1
fi
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
source "${SCRIPT_DIR}/result_layout.sh"

source "${SCRIPT_DIR}/cluster_env.sh"

# The first entry for every dataset is a BN-free primary control. ResNet cells
# explicitly opt into the frozen-BN eval surrogate; they must not be described
# as exact train-mode BN gradients. Override as a space-delimited list of
# dataset:architecture:batchnorm_policy triples and adjust --array accordingly.
DEFAULT_CONFIGS="cifar10:lenet:reject cifar10:resnet18:eval_frozen cifar100:lenet:reject cifar100:resnet18:eval_frozen mnist:lenet:reject mnist:resnet18:eval_frozen"
read -r -a CONFIGS <<< "${CONFIGS_OVERRIDE:-$DEFAULT_CONFIGS}"
read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-42 43 44 45 46}"

N_SEEDS=${#SEEDS[@]}
CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID / N_SEEDS))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % N_SEEDS))
if (( CONFIG_INDEX >= ${#CONFIGS[@]} )); then
    echo "Array index ${SLURM_ARRAY_TASK_ID} exceeds ${#CONFIGS[@]} configs x ${N_SEEDS} seeds" >&2
    exit 2
fi

IFS=: read -r DATASET ARCHITECTURE BN_POLICY <<< "${CONFIGS[$CONFIG_INDEX]}"
SEED=${SEEDS[$SEED_INDEX]}
read -r -a FRACTIONS <<< "${FRACTIONS_OVERRIDE:-0.1 0.2 0.4}"
read -r -a SAMPLERS <<< "${SAMPLERS_OVERRIDE:-uniform stratified easy hard}"

SCRATCH_BASE="$RESULTS_ROOT"
REFERENCE_OPTIMIZER="${REFERENCE_OPTIMIZER:-${OPTIMIZER:-sgd}}"
RETRAIN_OPTIMIZER="${RETRAIN_OPTIMIZER:-$REFERENCE_OPTIMIZER}"
VARIANT="$(layout_variant "$REFERENCE_OPTIMIZER")"
DATA_ROOT_BASE="${DATA_ROOT_BASE:-${SCRATCH_BASE}/datasets}"
DATA_DIR="${DATA_ROOT:-${DATA_ROOT_BASE}/${DATASET}}"
if [ -z "${OUTPUT_DIR+x}" ]; then
    OUTPUT_DIR="$(layout_root "$SCRATCH_BASE" data_selection_swap "$VARIANT")/d_correlation"
fi
STAGE="${STAGE:-all}"
mkdir -p "$DATA_DIR" "$OUTPUT_DIR"

DOWNLOAD_ARGS=()
if [[ "${DOWNLOAD:-0}" == "1" ]]; then
    DOWNLOAD_ARGS=(--download)
fi

echo "D(alpha) correlation cell"
echo "  stage=$STAGE dataset=$DATASET architecture=$ARCHITECTURE split_seed=$SEED"
echo "  batchnorm_policy=$BN_POLICY fractions=${FRACTIONS[*]} samplers=${SAMPLERS[*]}"
echo "  data=$DATA_DIR output=$OUTPUT_DIR"

srun python "${PROJECT_ROOT}/scripts/experiments/run_d_accuracy_correlation.py" \
    --stage "$STAGE" \
    --dataset "$DATASET" \
    --architecture "$ARCHITECTURE" \
    --batchnorm-policy "$BN_POLICY" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --split-seed "$SEED" \
    --reference-seed "$SEED" \
    --reference-optimizer "$REFERENCE_OPTIMIZER" \
    --reference-epochs "${REFERENCE_EPOCHS:-0}" \
    --num-d-checkpoints "${NUM_D_CHECKPOINTS:-5}" \
    --reference-resume-every "${REFERENCE_RESUME_EVERY:-5}" \
    --fractions "${FRACTIONS[@]}" \
    --samplers "${SAMPLERS[@]}" \
    --subsets-per-sampler "${SUBSETS_PER_SAMPLER:-1}" \
    --loss-pool-multiplier "${LOSS_POOL_MULTIPLIER:-2.0}" \
    --d-weighting "${D_WEIGHTING:-uniform}" \
    --d-normalization "${D_NORMALIZATION:-sum}" \
    --retrain-optimizer "$RETRAIN_OPTIMIZER" \
    --retrain-epochs "${RETRAIN_EPOCHS:-0}" \
    --retrain-replicates "${RETRAIN_REPLICATES:-1}" \
    --batch-size "${BATCH_SIZE:-128}" \
    --d-batch-size "${D_BATCH_SIZE:-256}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}" \
    "${DOWNLOAD_ARGS[@]}"
