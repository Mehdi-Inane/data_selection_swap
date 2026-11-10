#!/usr/bin/env bash
# Phase 0 bring-up runner for mila (see docs/llm_less_protocol.md).
# Thin wrapper: repo path + uv env + HF cache, then run_pipeline.py --run with the
# stage selector passed through.
#
# Submit (one L40S):
#   sbatch -p long -t 2-00:00:00 --mem=64G --cpus-per-task=8 --gres=gpu:l40s:1 \
#          --output=/network/scratch/a/ahmedm/less/logs/phase0_%j.out \
#          scripts/llm/run_phase0_mila.sh --stage warmup:seed0
#
# Intended sequence:
#   1. --stage warmup:seed0           # LoRA warmup; then read vendor/out/<job>/train.log
#                                      # for the real checkpoint step numbers.
#   2. (set ckpts/checkpoint_weights in scripts/llm/pipeline.phase0.mila.json if they
#      differ from [105 211 317 420]) and run:
#      --stages "train_grads,val_grads,store,select:"   # substring matching;
#      select stages need no GPU (CPU only, run after grads).
#   3. Selection outputs (selected.jsonl, diagnostics.json) land under runs_dir.
#      Accuracy reproduction (train/eval stages) additionally needs LESS's
#      evaluation/open_instruct installed — see the protocol doc, Phase 0 step 5.
#
set -euo pipefail
cd "$(dirname "$0")/../.."

export UV_PROJECT_ENVIRONMENT=/network/scratch/a/ahmedm/venvs/data_selection_swap
export HF_HOME=/network/scratch/a/ahmedm/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p /network/scratch/a/ahmedm/less/{logs,runs,work}
nvidia-smi || true

uv run --extra llm python scripts/llm/run_pipeline.py \
    --config scripts/llm/pipeline.phase0.mila.json "$@" --run
