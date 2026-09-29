#!/bin/bash
#SBATCH --job-name=interpolate_cifar100
#SBATCH --output=logs/interp_cifar100_%j.out
#SBATCH --error=logs/interp_cifar100_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=08:00:00              # ~2-3h full-data training (once) + 4 budgets of grid scans
#SBATCH --mem=32Gb

set -euo pipefail
mkdir -p logs

# ── Environment Setup ────────────────────────────────────────────────────────
module load miniconda/3
set +u
conda activate gradmatch
set -u

# ── Configurations & Paths ───────────────────────────────────────────────────
SCRATCH_BASE="${SCRATCH:-/home/mila/a/ahmedm/scratch}"

CIFAR_TAR_SRC="${CIFAR_TAR_SRC:-/network/datasets/cifar100/cifar-100-python.tar.gz}"
LOCAL_DATA_DIR="${SLURM_TMPDIR}/cifar100_data"

BUDGETS=(4500 9000 18000 22500)
SEED=42
K=1
NUM_EPOCHS=300
BASELINES=("random" "graphcut" "moderate")

# One sub-directory per budget for the interpolation results, plus ONE shared directory for the
# full-data model: it depends only on (seed, num_epochs), not on the budget, so it is trained once.
ROOT_OUT="${SCRATCH_BASE}/connectivity/cifar100_seed${SEED}"
FULL_DIR="${ROOT_OUT}/full"
FULL_NAME="full_seed${SEED}_epoch${NUM_EPOCHS}.pth"     # must match interpolate_checkpoints.py
FULL_CKPT="${FULL_DIR}/${FULL_NAME}"
FULL_METRICS="${FULL_DIR}/${FULL_NAME%.pth}_metrics.json"

mkdir -p "$LOCAL_DATA_DIR" "$FULL_DIR"

# ── Stage & Extract Dataset onto Node-Local SSD ($SLURM_TMPDIR) ───────────────
echo "=================================================="
echo " Staging CIFAR-100 onto node-local storage"
echo " Source Archive : $CIFAR_TAR_SRC"
echo " Target Dir     : $LOCAL_DATA_DIR"
echo "=================================================="

if [ -f "$CIFAR_TAR_SRC" ]; then
    echo "Copying archive to $SLURM_TMPDIR..."
    cp "$CIFAR_TAR_SRC" "$SLURM_TMPDIR/"

    echo "Extracting cifar-100-python.tar.gz into $LOCAL_DATA_DIR..."
    tar -xf "${SLURM_TMPDIR}/$(basename "$CIFAR_TAR_SRC")" -C "$LOCAL_DATA_DIR"
    echo "Extraction complete."
else
    echo "Error: Archive not found at $CIFAR_TAR_SRC" >&2
    exit 1
fi

# ── Run Interpolation Script for every budget ────────────────────────────────
FAILED=()

for BUDGET in "${BUDGETS[@]}"; do
    OUT_DIR="${ROOT_OUT}/${BUDGET}"
    mkdir -p "$OUT_DIR"

    # Resume support: this budget already finished in an earlier run (last file written by the
    # script is the loss PDF). Set FORCE_RERUN=1 to redo it anyway.
    if [ -f "${OUT_DIR}/interp_test_loss.pdf" ] && [ "${FORCE_RERUN:-0}" != "1" ]; then
        echo "[budget=$BUDGET] already done (${OUT_DIR}/interp_test_loss.pdf) -> skipping"
        continue
    fi

    # If the shared full-data model already exists, expose it in this budget's out_dir so that
    # interpolate_checkpoints.py finds it and skips training.
    if [ -f "$FULL_CKPT" ]; then
        ln -sfn "$FULL_CKPT" "${OUT_DIR}/${FULL_NAME}"
    fi

    echo ""
    echo "--------------------------------------------------"
    echo " Running Checkpoint Interpolation"
    echo " Budget    : $BUDGET"
    echo " Baselines : ${BASELINES[*]}"
    echo " Output    : $OUT_DIR"
    echo " Full model: $FULL_CKPT ($([ -f "$FULL_CKPT" ] && echo 'reusing' || echo 'will be trained now'))"
    echo "--------------------------------------------------"

    # `if !` so that one bad budget (e.g. missing ours/baseline checkpoint) doesn't kill the rest
    if ! srun python interpolate_checkpoints.py \
            --data_dir "$LOCAL_DATA_DIR" \
            --baselines "${BASELINES[@]}" \
            --budget "$BUDGET" \
            --seed "$SEED" \
            --K "$K" \
            --num_epochs "$NUM_EPOCHS" \
            --out_dir "$OUT_DIR"; then
        echo "[budget=$BUDGET] FAILED" >&2
        FAILED+=("$BUDGET")
    fi

    # The first budget that gets this far trains the full model inside its own out_dir:
    # publish it to the shared directory and leave a symlink behind for later budgets.
    if [ ! -f "$FULL_CKPT" ] && [ -f "${OUT_DIR}/${FULL_NAME}" ] && [ ! -L "${OUT_DIR}/${FULL_NAME}" ]; then
        echo "[budget=$BUDGET] publishing freshly trained full-data model -> $FULL_CKPT"
        mv "${OUT_DIR}/${FULL_NAME}" "$FULL_CKPT"
        mv "${OUT_DIR}/${FULL_NAME%.pth}_metrics.json" "$FULL_METRICS" 2>/dev/null || true
        ln -sfn "$FULL_CKPT" "${OUT_DIR}/${FULL_NAME}"
    fi
done

echo ""
echo "=================================================="
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo " Finished with failures for budgets: ${FAILED[*]}" >&2
    exit 1
fi
echo " All budgets (${BUDGETS[*]}) completed successfully!"
echo "=================================================="