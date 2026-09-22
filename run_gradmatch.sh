#!/bin/bash
#SBATCH --job-name=data_selection_like_paper
#SBATCH --output=logs/gradmatch_%A_%a.txt
#SBATCH --error=logs/gradmatch_%A_%a.txt
#SBATCH --cpus-per-task=4            # matches --num-workers below
#SBATCH --gres=gpu:1           # 1 GPU per job
#SBATCH --time=02:00:00              # train (~40 epochs) + feature extraction + Algorithm 3 + baselines + eval-training on 4 subsets
#SBATCH --mem=48Gb



set -euo pipefail
mkdir -p logs


module load miniconda/3
set +u                                # conda's activate script references unbound $PS1 under `set -u`
conda activate gradmatch    # conda create -n faithful_selection python=3.12 + pip install -r requirements.txt
set -u

srun python3 train_baselines.py