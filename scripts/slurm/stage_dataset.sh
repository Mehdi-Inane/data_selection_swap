#!/bin/bash
# stage_dataset.sh
#
# Shared helper for staging/extracting a dataset onto node-local SSD
# ($SLURM_TMPDIR). Source this file, then call:
#
#   stage_dataset "$DATASET" "$LOCAL_DATA"
#
# Supported datasets: "cifar10", "cifar100", "mnist", "imagenet"

IMAGENET_RAW_DIR="/network/datasets/imagenet"
CIFAR100_ARCHIVE="/network/datasets/cifar100/cifar-100-python.tar.gz"

stage_dataset() {
    local dataset="$1"
    local local_data="$2"

    mkdir -p "$local_data"

    if [ "$dataset" = "cifar100" ]; then
        if [ -f "$CIFAR100_ARCHIVE" ]; then
            echo "Extracting CIFAR-100 archive to $local_data ..."
            tar -xzf "$CIFAR100_ARCHIVE" -C "$local_data"
        else
            echo "Warning: $CIFAR100_ARCHIVE not found. Python script will attempt automatic download." >&2
        fi

    elif [ "$dataset" = "cifar10" ] || [ "$dataset" = "mnist" ]; then
        # torchvision performs an integrity-checked download into local_data.
        # Keeping this branch explicit means the launchers pass the same root
        # to every stage instead of silently falling through as an unknown set.
        echo "$dataset will be integrity-checked/downloaded by torchvision in $local_data"

    elif [ "$dataset" = "imagenet" ]; then
        if [ ! -d "$IMAGENET_RAW_DIR" ]; then
            echo "Error: ImageNet directory not found at $IMAGENET_RAW_DIR" >&2
            return 1
        fi

        echo "=== Staging ImageNet from Raw Tar Archives ==="

        # 1. Extract Training Set
        if [ ! -f "$local_data/.imagenet_train_complete" ]; then
            if [ -d "$local_data/train" ] && [ -n "$(find "$local_data/train" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
                echo "Error: partial ImageNet train staging found without completion marker." >&2
                echo "Use a fresh node-local destination; refusing to certify a partial directory." >&2
                return 1
            fi
            echo "1/2 Unpacking ILSVRC2012_img_train.tar..."
            mkdir -p "$local_data/train"
            tar -xf "$IMAGENET_RAW_DIR/ILSVRC2012_img_train.tar" -C "$local_data/train"

            echo "Extracting 1000 class archives in parallel..."
            find "$local_data/train" -name "*.tar" | xargs -n 1 -P 16 -I {} sh -c '
                dir="${1%.tar}"
                mkdir -p "$dir"
                tar -xf "$1" -C "$dir"
                rm -f "$1"
            ' _ {}
            class_count=$(find "$local_data/train" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
            if [ "$class_count" -ne 1000 ]; then
                echo "Error: expected 1000 ImageNet train classes, found $class_count" >&2
                return 1
            fi
            touch "$local_data/.imagenet_train_complete"
        else
            echo "1/2 Train set has a completion marker, skipping."
        fi

        # 2. Extract Validation Set & Organize into ImageFolder format
        if [ ! -f "$local_data/.imagenet_val_complete" ]; then
            if [ -d "$local_data/val" ] && [ -n "$(find "$local_data/val" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
                echo "Error: partial ImageNet validation staging found without completion marker." >&2
                return 1
            fi
            echo "2/2 Unpacking ILSVRC2012_img_val.tar..."
            mkdir -p "$local_data/val"
            tar -xf "$IMAGENET_RAW_DIR/ILSVRC2012_img_val.tar" -C "$local_data/val"

            echo "Sorting validation images into class subdirectories..."
            if [ -z "${IMAGENET_VALPREP:-}" ] || [ ! -f "$IMAGENET_VALPREP" ]; then
                echo "Error: set IMAGENET_VALPREP to a reviewed local valprep.sh file." >&2
                return 1
            fi
            if [ -z "${IMAGENET_VALPREP_SHA256:-}" ]; then
                echo "Error: set IMAGENET_VALPREP_SHA256 to pin the reviewed script." >&2
                return 1
            fi
            actual_sha=$(shasum -a 256 "$IMAGENET_VALPREP" | awk '{print $1}')
            if [ "$actual_sha" != "$IMAGENET_VALPREP_SHA256" ]; then
                echo "Error: ImageNet valprep checksum mismatch." >&2
                return 1
            fi
            (cd "$local_data/val" && bash "$IMAGENET_VALPREP")
            class_count=$(find "$local_data/val" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
            if [ "$class_count" -ne 1000 ]; then
                echo "Error: expected 1000 ImageNet validation classes, found $class_count" >&2
                return 1
            fi
            touch "$local_data/.imagenet_val_complete"
        else
            echo "2/2 Validation set has a completion marker, skipping."
        fi

    else
        echo "Error: Unknown dataset '$dataset'. Choose cifar10, cifar100, mnist, or imagenet." >&2
        return 1
    fi
}
