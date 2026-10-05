# Shared result layout. Source after SCRIPT_DIR is known.
# OPTIMIZER=sgd (the default) keeps the historical directories.
# OPTIMIZER=adam or adamw writes under an adam/ subfolder of each root.

layout_variant() {
    case "${1:-sgd}" in
        adam|adamw|adam_update) printf 'adam' ;;
        sgd|uniform|'') printf '' ;;
        *)
            echo "Unknown OPTIMIZER '${1}'. Choose sgd or adam." >&2
            return 1
            ;;
    esac
}

layout_root() {
    # layout_root <scratch> <root name> <variant>
    if [ -n "${3:-}" ]; then
        printf '%s/%s/%s' "$1" "$2" "$3"
    else
        printf '%s/%s' "$1" "$2"
    fi
}

layout_retrain_lr() {
    # Default subset-retraining learning rate. An explicit LR= overrides it.
    if [ -n "${LR:-}" ]; then
        printf '%s' "$LR"
        return
    fi
    case "${1:-sgd}" in
        adam|adamw|adam_update) printf '0.001' ;;
        *) printf '0.01' ;;
    esac
}
