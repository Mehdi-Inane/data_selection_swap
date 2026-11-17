#!/usr/bin/env bash
# Fair-comparison swap run: ranked descent initialized at LESS top-k, capped at
# max_iters = m (one replacement opportunity per selected member). Only improving
# swaps are accepted, so f(swap) <= f(LESS) by construction. See
# docs/llm_pythia_phase0_report.md (sec. 8.3 / 10).
#
# Submit (one per task):
#   sbatch -p long -t 1-12:00:00 --mem=250G --cpus-per-task=16 --gres=gpu:l40s:1 \
#          --output=$SCRATCH/less/logs/swapinit_%j.out \
#          scripts/llm/run_swap_less_init.sh mmlu
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

export UV_PROJECT_ENVIRONMENT=/network/scratch/a/ahmedm/venvs/data_selection_swap
export HF_HOME=/network/scratch/a/ahmedm/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    export HF_DATASETS_CACHE="/tmp/dscache-${USER}-${SLURM_JOB_ID}"
    trap 'rm -rf "$HF_DATASETS_CACHE"' EXIT
fi

task="${1:?usage: $0 <task> (mmlu|tydiqa|bbh)}"
MAX_ITERS="${MAX_ITERS:-13534}"   # = subset size m at fraction 0.05 of the 270,679 pool
SEED="${SEED:-0}"                 # selection seed: seed 1 = store-seed1 -> ...:seed1:initless
ROOT=/network/scratch/a/ahmedm/less
store="$ROOT/work/pythia410m/store"
[[ "$SEED" != "0" ]] && store="$store-seed$SEED"

uv run --extra llm python scripts/llm/select_swap.py \
    --store "$store" \
    --out "$ROOT/runs/pythia410m/select:swap:${task}:p0.05:seed${SEED}:initless" \
    --setting targeted --task "$task" --fraction 0.05 \
    --rule ranked --max-iters "$MAX_ITERS" --init less
