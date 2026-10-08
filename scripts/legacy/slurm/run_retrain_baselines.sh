#!/bin/bash
#SBATCH --job-name=data_selection_like_paper
#SBATCH --output=logs/gradmatch_%A_%a.txt
#SBATCH --error=logs/gradmatch_%A_%a.txt
#SBATCH --cpus-per-task=4            # matches --num-workers below
#SBATCH --gres=gpu:1           # 1 GPU per job
#SBATCH --time=02:00:00              # train (~40 epochs) + feature extraction + Algorithm 3 + baselines + eval-training on 4 subsets
#SBATCH --mem=48Gb



set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
mkdir -p "${PROJECT_ROOT}/logs"


source "${PROJECT_ROOT}/scripts/slurm/cluster_env.sh"

srun python3 "${PROJECT_ROOT}/scripts/legacy/train_baselines.py"
