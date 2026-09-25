#!/bin/bash
#SBATCH --job-name=gradmatch_cifar10_linear_layer
#SBATCH --output=logs/linear_layer_gradmatch_cifar10_%A_%a.txt
#SBATCH --error=logs/linear_layer_gradmatch_cifar10_%A_%a.txt
#SBATCH --array=0-19                 # 20 tasks = 4 fractions × 5 seeds
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=06:00:00              # CIFAR-10 300 epochs is ~1-2 hrs/run; 6 is generous
#SBATCH --mem=24Gb                   # CIFAR-10 is small; 24 GB is enough

set -euo pipefail
mkdir -p logs

# ── Environment ──────────────────────────────────────────────────────────────
module load miniconda/3
set +u
conda activate gradmatch
set -u

# ── Experiment grid ───────────────────────────────────────────────────────────
FRACTIONS=(0.1 0.2 0.4 0.5)
SEEDS=(42 43 44 45 46)

# Layout: task_id = fraction_idx * 5 + seed_idx
FRACTION_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))

FRACTION=${FRACTIONS[$FRACTION_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH_BASE="${SCRATCH:-/home/mila/a/ahmedm/scratch}"

# Where CIFAR-10 is already installed on the shared filesystem.
# torchvision expects: $CIFAR10_CACHE/cifar-10-batches-py/
# Adjust this to wherever you ran `datasets.CIFAR10(download=True)` previously.
CIFAR10_CACHE="/network/datasets/cifar10"

# Node-local SSD destination (fast local I/O during training).
LOCAL_DATA="${SLURM_TMPDIR}/cifar10_data"

echo "=================================================="
echo " SLURM Array Task ID : $SLURM_ARRAY_TASK_ID"
echo " Fraction            : $FRACTION  (index $FRACTION_IDX)"
echo " Seed                : $SEED      (index $SEED_IDX)"
echo " CIFAR-10 cache      : $CIFAR10_CACHE"
echo " Node-local data dir : $LOCAL_DATA"
echo "=================================================="

# ── Stage CIFAR-10 onto node-local SSD ───────────────────────────────────────
# CIFAR-10 is ~170 MB; rsync takes <10 seconds and gives us fast local reads.
echo "[stage] Copying CIFAR-10 from shared FS → SLURM_TMPDIR ..."
mkdir -p "$LOCAL_DATA"

if [ -d "${CIFAR10_CACHE}/cifar-10-batches-py" ]; then
    # Data already extracted; just rsync the directory tree.
    rsync -a --info=progress2 \
        "${CIFAR10_CACHE}/cifar-10-batches-py" \
        "${LOCAL_DATA}/"
    echo "[stage] Done. Using pre-extracted dataset."
elif [ -f "${CIFAR10_CACHE}/cifar-10-python.tar.gz" ]; then
    # Tarball present but not yet extracted; extract into LOCAL_DATA.
    cp "${CIFAR10_CACHE}/cifar-10-python.tar.gz" "${LOCAL_DATA}/"
    tar -xzf "${LOCAL_DATA}/cifar-10-python.tar.gz" -C "${LOCAL_DATA}/"
    rm  "${LOCAL_DATA}/cifar-10-python.tar.gz"
    echo "[stage] Done. Extracted from tarball."
else
    # Fallback: let torchvision download it directly to LOCAL_DATA.
    # This will work but adds ~30 s of network latency on first run.
    echo "[stage] WARNING: no cached CIFAR-10 found at ${CIFAR10_CACHE}."
    echo "[stage] Falling back to torchvision auto-download into ${LOCAL_DATA}."
    DOWNLOAD_FLAG="--download"
fi

echo ""
echo "--------------------------------------------------"
echo " Running GradMatch | dataset=cifar10 | fraction=$FRACTION | seed=$SEED"
echo "--------------------------------------------------"

srun python gradmatch_selection.py \
    --dataset     cifar10           \
    --fraction    "$FRACTION"       \
    -seed         "$SEED"           \
    --data_dir    "$LOCAL_DATA"     \
    --batch_size  128               \
    --lr          0.01              \
    --num_workers 4                 \
    --select_every 20               \
    --kappa       0.0               \
    --lam         0.5               \
    --linear_layer                  \
    ${DOWNLOAD_FLAG:-}

echo ""
echo "=================================================="
echo " Task $SLURM_ARRAY_TASK_ID completed successfully!"
echo "=================================================="