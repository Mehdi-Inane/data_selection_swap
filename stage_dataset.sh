#!/bin/bash
# stage_dataset.sh
#
# Shared helper for staging/extracting a dataset onto node-local SSD
# ($SLURM_TMPDIR). Source this file, then call:
#
#   stage_dataset "$DATASET" "$LOCAL_DATA"
#
# Supported datasets: "cifar100", "imagenet"

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

    elif [ "$dataset" = "imagenet" ]; then
        if [ ! -d "$IMAGENET_RAW_DIR" ]; then
            echo "Error: ImageNet directory not found at $IMAGENET_RAW_DIR" >&2
            return 1
        fi

        echo "=== Staging ImageNet from Raw Tar Archives ==="

        # 1. Extract Training Set
        if [ ! -d "$local_data/train" ]; then
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
        else
            echo "1/2 Train set already extracted at $local_data/train, skipping."
        fi

        # 2. Extract Validation Set & Organize into ImageFolder format
        if [ ! -d "$local_data/val" ]; then
            echo "2/2 Unpacking ILSVRC2012_img_val.tar..."
            mkdir -p "$local_data/val"
            tar -xf "$IMAGENET_RAW_DIR/ILSVRC2012_img_val.tar" -C "$local_data/val"

            echo "Sorting validation images into class subdirectories..."
            (cd "$local_data/val" && wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash)
        else
            echo "2/2 Val set already extracted at $local_data/val, skipping."
        fi

    else
        echo "Error: Unknown dataset '$dataset'. Choose 'cifar100' or 'imagenet'." >&2
        return 1
    fi
}