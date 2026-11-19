#!/bin/bash
# Submit a configs/*.yaml experiment as Slurm arrays, one task per grid
# point: every stage, or only the named one. Extra arguments go to sbatch,
# e.g. a pinned GPU type:
#   scripts/slurm/submit_config.sh configs/paper/main_comparison.yaml
#   scripts/slurm/submit_config.sh configs/paper/main_comparison.yaml gradient_swap \
#       --gres=gpu:rtx8000:1
set -euo pipefail
if [ $# -lt 1 ]; then
    echo "Usage: $0 CONFIG [STAGE] [sbatch options...]" >&2
    exit 2
fi
CONFIG="$1"
shift

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER=(python "${PROJECT_ROOT}/scripts/experiments/run_config.py")
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
    STAGES=("$1")
    shift
else
    read -r -a STAGES <<< "$("${RUNNER[@]}" stages "$CONFIG")"
fi

# Refuse invalid or planned configs before anything is queued.
for STAGE in "${STAGES[@]}"; do
    "${RUNNER[@]}" run-task "$CONFIG" "$STAGE" 0 --dry-run > /dev/null
done
mkdir -p "${PROJECT_ROOT}/logs"
cd "$PROJECT_ROOT"
for STAGE in "${STAGES[@]}"; do
    N="$("${RUNNER[@]}" count "$CONFIG" "$STAGE")"
    sbatch --array="0-$((N - 1))" --job-name="$(basename "$CONFIG" .yaml)-${STAGE}" "$@" \
        scripts/slurm/run_config.sh "$CONFIG" "$STAGE"
done
