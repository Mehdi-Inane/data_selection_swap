#!/usr/bin/env bash
# Phase-0 accuracy gate (docs/llm_less_protocol.md step 5): LoRA fine-tune with
# LESS's own trainer on one selected subset, then evaluate with LESS's own
# open-instruct eval scripts. One (method, task) per job.
#
#   sbatch -p long -t 8:00:00 --mem=64G --cpus-per-task=8 --gres=gpu:l40s:1 \
#          --output=/network/scratch/a/ahmedm/less/logs/gate_%j.out \
#          scripts/llm/run_finetune_eval.sh <less|swap|random> <mmlu|tydiqa|bbh>
#
# swap uses the fair LESS-initialized selections (...:initless). The training
# job name must contain "lora": eval's load_hf_lm_and_tokenizer only applies a
# PEFT adapter when the model path contains "lora" (evaluation/eval/utils.py).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

VENV=/network/scratch/a/ahmedm/venvs/data_selection_swap
export UV_PROJECT_ENVIRONMENT="$VENV"
export PATH="$VENV/bin:$PATH"          # torchrun lives in the venv (lora_train.sh)
export HF_HOME=/network/scratch/a/ahmedm/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    export HF_DATASETS_CACHE="/tmp/dscache-${USER}-${SLURM_JOB_ID}"
    trap 'rm -rf "$HF_DATASETS_CACHE"' EXIT
fi

method="${1:?usage: $0 <less|swap|random> <mmlu|tydiqa|bbh>}"
task="${2:?usage: $0 <less|swap|random> <mmlu|tydiqa|bbh>}"

ROOT=/network/scratch/a/ahmedm/less
MODEL="$HF_HOME/models/pythia-410m"
sel="$ROOT/runs/pythia410m/select:${method}:${task}:p0.05:seed0"
[[ "$method" == "swap" ]] && sel="$sel:initless"     # fair headline subset
selected="$sel/selected.jsonl"
if [[ "${SKIP_TRAIN:-0}" != "1" && ! -f "$selected" ]]; then
    echo "missing $selected" >&2; exit 1
fi

job="pythia410m-lora-${method}-${task}-p0.05-seed0"

# 1) fine-tune with LESS's own defaults (less/scripts/train/base_training_args.sh)
#    SKIP_TRAIN=1 re-evaluates an existing adapter (smoke tests, re-scores).
cd vendor/LESS
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
    bash less/scripts/train/lora_train.sh "$selected" "$MODEL" "$job"
fi
out_dir="$PWD/../out/$job"
[[ -f "$out_dir/adapter_config.json" ]] || { echo "adapter not saved in $out_dir" >&2; exit 1; }

# eval.sh clobbers DATA_DIR with its author's path (referencing an unbound $n,
# so source it with nounset off), and export ours only after sourcing.
cd evaluation
set +u
source "eval_${task}.sh"
set -u
export DATA_DIR="$ROOT/DATA/eval"
"eval_${task}" "$out_dir"
"extract_${task}" "$out_dir" | tee "$out_dir/eval/${task}/score.txt"
