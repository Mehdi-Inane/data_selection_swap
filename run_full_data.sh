#!/bin/bash
#SBATCH --job-name=train_full_ref
#SBATCH --output=logs/train_full_ref_%j.txt
#SBATCH --error=logs/train_full_ref_%j.txt
#SBATCH --cpus-per-task=4            
#SBATCH --gres=gpu:1                 
#SBATCH --time=06:00:00              
#SBATCH --mem=48Gb

# Fail on error
set -euo pipefail

# Create logs directory if it doesn't exist
mkdir -p logs

# Load environment
module load miniconda/3
set +u                                
conda activate gradmatch             
set -u

echo "=================================================="
echo "Starting Full CIFAR-10 Reference Trajectory Run"
echo "=================================================="

# Run the training script
srun python train_full_data.py

echo "=================================================="
echo "Trajectory generation complete!"
echo "=================================================="