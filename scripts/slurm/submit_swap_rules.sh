#!/bin/bash
# Submit paired Algorithm 2/Algorithm 3 arrays with strict dependencies.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DRIVER="${PROJECT_ROOT}/scripts/experiments/compare_swap_rules.py"
LAUNCHER="${SCRIPT_DIR}/run_swap_rules.sh"

CONFIG_INPUT="${1:-${SWAP_RULE_CONFIG:-}}"
if [ -z "$CONFIG_INPUT" ]; then
    echo "Usage: $0 CONFIG_JSON [OUTPUT_ROOT]" >&2
    exit 2
fi
CONFIG_DIR="$(cd "$(dirname "$CONFIG_INPUT")" && pwd)"
CONFIG="${CONFIG_DIR}/$(basename "$CONFIG_INPUT")"
if [ ! -f "$CONFIG" ]; then
    echo "Configuration not found: $CONFIG" >&2
    exit 2
fi

OUTPUT_ROOT="${2:-${SWAP_RULE_OUTPUT_ROOT:-${RESULTS_ROOT:-${SCRATCH:-${PROJECT_ROOT}/results}}/data_selection_swap/swap_rules}}"
MAX_PARALLEL="${SWAP_RULE_MAX_PARALLEL:-8}"
LOG_DIR="${PROJECT_ROOT}/logs/swap_rules"
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_ROOT"

python "$DRIVER" validate --config "$CONFIG"
N_REFERENCE="$(python "$DRIVER" count --config "$CONFIG" --stage reference)"
N_SELECTION="$(python "$DRIVER" count --config "$CONFIG" --stage selection)"
N_RETRAIN="$(python "$DRIVER" count --config "$CONFIG" --stage retrain)"
COMMON_EXPORT="ALL,SWAP_RULE_CONFIG=${CONFIG},SWAP_RULE_OUTPUT_ROOT=${OUTPUT_ROOT}"

REFERENCE_JOB="$(sbatch --parsable \
    --job-name=swap_ref \
    --array="0-$((N_REFERENCE - 1))%${MAX_PARALLEL}" \
    --output="${LOG_DIR}/reference_%A_%a.log" \
    --error="${LOG_DIR}/reference_%A_%a.log" \
    --export="${COMMON_EXPORT},SWAP_RULE_STAGE=reference" \
    "$LAUNCHER")"
REFERENCE_DEP="${REFERENCE_JOB%%;*}"

SELECTION_JOB="$(sbatch --parsable \
    --job-name=swap_select \
    --dependency="afterok:${REFERENCE_DEP}" \
    --array="0-$((N_SELECTION - 1))%${MAX_PARALLEL}" \
    --output="${LOG_DIR}/selection_%A_%a.log" \
    --error="${LOG_DIR}/selection_%A_%a.log" \
    --export="${COMMON_EXPORT},SWAP_RULE_STAGE=selection" \
    "$LAUNCHER")"
SELECTION_DEP="${SELECTION_JOB%%;*}"

RETRAIN_JOB="$(sbatch --parsable \
    --job-name=swap_retrain \
    --dependency="afterok:${SELECTION_DEP}" \
    --array="0-$((N_RETRAIN - 1))%${MAX_PARALLEL}" \
    --output="${LOG_DIR}/retrain_%A_%a.log" \
    --error="${LOG_DIR}/retrain_%A_%a.log" \
    --export="${COMMON_EXPORT},SWAP_RULE_STAGE=retrain" \
    "$LAUNCHER")"

echo "Submitted paired swap-rule experiment:"
echo "  reference : ${REFERENCE_JOB} (${N_REFERENCE} tasks)"
echo "  selection : ${SELECTION_JOB} (${N_SELECTION} tasks; afterok:${REFERENCE_JOB})"
echo "  retrain   : ${RETRAIN_JOB} (${N_RETRAIN} tasks; afterok:${SELECTION_JOB})"
echo "  output    : ${OUTPUT_ROOT}"
echo "After completion:"
echo "  python ${DRIVER} summarize --config ${CONFIG} --output-root ${OUTPUT_ROOT}"
