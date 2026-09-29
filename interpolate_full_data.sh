#!/bin/bash
#SBATCH --job-name=interp_full_runs_cifar100
#SBATCH --output=logs/interp_full_runs_cifar100_%j.out
#SBATCH --error=logs/interp_full_runs_cifar100_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=12:00:00              # 3 x ~2-3h full-data training + one grid scan (N=20)
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

SEED=42                 # first run; the runs use SEED, SEED+1, SEED+2
NUM_RUNS=3
NUM_EPOCHS=300
GRID_RES=20
SPLIT_SEED="${SPLIT_SEED:-}"   # optional: export SPLIT_SEED=42 to keep the train/val split fixed across runs

OUT_DIR="${SCRATCH_BASE}/connectivity/full_runs_seed${SEED}"
mkdir -p "$LOCAL_DATA_DIR" "$OUT_DIR"

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

# ── Resume support ───────────────────────────────────────────────────────────
# Finished per-seed checkpoints are reused automatically by the python script.
# Skip everything if the final plot already exists (FORCE_RERUN=1 to redo the scan/plots).
if [ -f "${OUT_DIR}/interp_test_loss.pdf" ] && [ "${FORCE_RERUN:-0}" != "1" ]; then
    echo "Already done (${OUT_DIR}/interp_test_loss.pdf) -> nothing to do"
    exit 0
fi

echo ""
echo "--------------------------------------------------"
echo " Full-data runs + interpolation"
echo " Seeds     : $(seq -s ' ' "$SEED" $((SEED + NUM_RUNS - 1)))"
echo " Epochs    : $NUM_EPOCHS"
echo " Grid res  : $GRID_RES"
echo " Split seed: ${SPLIT_SEED:-<same as run seed>}"
echo " Output    : $OUT_DIR"
echo "--------------------------------------------------"

EXTRA_ARGS=()
if [ -n "$SPLIT_SEED" ]; then
    EXTRA_ARGS+=(--split_seed "$SPLIT_SEED")
fi

srun python interpolate_full_training.py \
    --data_dir "$LOCAL_DATA_DIR" \
    --seed "$SEED" \
    --num_runs "$NUM_RUNS" \
    --num_epochs "$NUM_EPOCHS" \
    --grid_res "$GRID_RES" \
    --out_dir "$OUT_DIR" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=================================================="
echo " Done. Results in $OUT_DIR"
echo "=================================================="