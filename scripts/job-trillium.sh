#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# No --mem: Trillium rejects memory requests (it grants memory proportionally to CPUs).
#SBATCH --time=0:05:00

# Run the job command passed as an argument when submitting the job ('python main.py' for example)
echo "Running command: $@"
srun uv run "$@"
