# Shared cluster setup. Source this after PROJECT_ROOT is set.
#
# Results land in RESULTS_ROOT, otherwise SCRATCH, otherwise
# $PROJECT_ROOT/results. Set CONDA_MODULE and/or CONDA_ENV to load a
# module and activate a conda environment. When those are unset, the
# Python already on PATH is used.
#
# Older launchers may still set OPT_LR_MINICONDA_MODULE, OPT_LR_CONDA_ENV,
# SWAP_RULE_MINICONDA_MODULE, or SWAP_RULE_CONDA_ENV. Those are honored
# when CONDA_MODULE / CONDA_ENV are empty.

if [ -z "${PROJECT_ROOT:-}" ]; then
    echo "cluster_env.sh requires PROJECT_ROOT." >&2
    return 1
fi

RESULTS_ROOT="${RESULTS_ROOT:-${SCRATCH:-${PROJECT_ROOT}/results}}"
export RESULTS_ROOT

if [ -z "${CONDA_MODULE:-}" ]; then
    CONDA_MODULE="${OPT_LR_MINICONDA_MODULE:-${SWAP_RULE_MINICONDA_MODULE:-}}"
fi
if [ -z "${CONDA_ENV:-}" ]; then
    CONDA_ENV="${OPT_LR_CONDA_ENV:-${SWAP_RULE_CONDA_ENV:-}}"
fi

if [ -n "$CONDA_MODULE" ] && command -v module >/dev/null 2>&1; then
    module load "$CONDA_MODULE"
fi
if [ -n "$CONDA_ENV" ] && command -v conda >/dev/null 2>&1; then
    set +u
    conda activate "$CONDA_ENV"
    set -u
fi
