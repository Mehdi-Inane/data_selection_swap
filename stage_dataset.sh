#!/bin/bash
# stage_dataset.sh
#
# Shared helper for staging/extracting a dataset onto node-local SSD
# ($SLURM_TMPDIR). Source this file, then call:
#
#   stage_dataset "$DATASET" "$LOCAL_DATA"
#
# Supported datasets: "cifar10", "cifar100", "imagenet"

IMAGENET_RAW_DIR="/network/datasets/imagenet"
CIFAR10_ARCHIVE="/network/datasets/cifar10/cifar-10-python.tar.gz"
CIFAR100_ARCHIVE="/network/datasets/cifar100/cifar-100-python.tar.gz"

# Extract a CIFAR tarball into $local_data unless it is already there.
# Args: <pretty name> <archive path> <folder created by the archive> <local_data>
_stage_cifar() {
    local name="$1"
    local archive="$2"
    local marker="$3"
    local local_data="$4"

    if [ -d "$local_data/$marker" ]; then
        echo "$name already extracted at $local_data/$marker, skipping."
    elif [ -f "$archive" ]; then
        echo "Extracting $name archive to $local_data ..."
        tar -xzf "$archive" -C "$local_data"
    else
        echo "Warning: $archive not found. Python script will attempt automatic download." >&2
    fi
}

stage_dataset() {
    local dataset="$1"
    local local_data="$2"

    mkdir -p "$local_data"

    if [ "$dataset" = "cifar10" ]; then
        _stage_cifar "CIFAR-10" "$CIFAR10_ARCHIVE" "cifar-10-batches-py" "$local_data"

    elif [ "$dataset" = "cifar100" ]; then
        _stage_cifar "CIFAR-100" "$CIFAR100_ARCHIVE" "cifar-100-python" "$local_data"

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
        echo "Error: Unknown dataset '$dataset'. Choose 'cifar10', 'cifar100' or 'imagenet'." >&2
        return 1
    fi
}