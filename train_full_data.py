import argparse
import glob
import json
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as tvm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.models.googlenet import BasicConv2d


# ───────────────────────────── Utilities ─────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def atomic_save(obj, path):
    """Write to a temp file then rename, so a preempted job never leaves a corrupt file."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=4)
    os.replace(tmp, path)


def get_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


# ───────────────────────────── Data ──────────────────────────────────────────
# Folder torchvision expects inside the dataset root for each dataset.
DATA_MARKERS = {
    "cifar10": ["cifar-10-batches-py"],
    "cifar100": ["cifar-100-python"],
    #"imagenet": ["train", "val"],
}


def resolve_data_root(dataset, data_dir, download=False):
    """Locate the dataset root, defaulting to $SLURM_TMPDIR and trying common sub-folders."""
    base = data_dir or os.environ.get("SLURM_TMPDIR")
    if base is None:
        raise ValueError("Pass --data_dir or run inside a SLURM job where $SLURM_TMPDIR is set.")

    candidates = [base] + [os.path.join(base, sub) for sub in (
        f"{dataset}_data", dataset, "data", f"{dataset}-data",
        *(("ILSVRC", "ILSVRC/Data/CLS-LOC") if dataset == "imagenet" else ()),
    )]
    for root in candidates:
        if all(os.path.isdir(os.path.join(root, m)) for m in DATA_MARKERS[dataset]):
            return root

    if download and dataset != "imagenet":
        root = os.path.join(base, f"{dataset}_data")
        os.makedirs(root, exist_ok=True)
        return root

    raise FileNotFoundError(
        f"Could not find '{dataset}' (looking for {DATA_MARKERS[dataset]}). Tried:\n  "
        + "\n  ".join(candidates)
    )


def get_dataset_config(dataset, data_dir, download=False):
    """Return hyper-params plus train-transform and eval-transform versions of the train set."""
    root = resolve_data_root(dataset, data_dir, download)

    if dataset in ("cifar10", "cifar100"):
        if dataset == "cifar10":
            cls, n_cls = datasets.CIFAR10, 10
            mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        else:
            cls, n_cls = datasets.CIFAR100, 100
            mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        tf_tr = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        tf_te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return dict(
            root=root, num_classes=n_cls, num_epochs=300, cifar_style=True,
            full_train=cls(root, train=True, download=download, transform=tf_tr),
            full_train_eval=cls(root, train=True, download=False, transform=tf_te),
            testset=cls(root, train=False, download=download, transform=tf_te),
        )

    if dataset == "imagenet":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        tf_tr = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        tf_te = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        return dict(
            root=root, num_classes=1000, num_epochs=350, cifar_style=False,
            full_train=datasets.ImageFolder(os.path.join(root, "train"), transform=tf_tr),
            full_train_eval=datasets.ImageFolder(os.path.join(root, "train"), transform=tf_te),
            testset=datasets.ImageFolder(os.path.join(root, "val"), transform=tf_te),
        )

    raise ValueError(f"Unknown dataset: {dataset!r}")


# ───────────────────────────── Models ────────────────────────────────────────
ARCHS = ["resnet18", "vgg16", "inception"]


def build_model(arch, num_classes, cifar_style):
    """
    resnet18  : torchvision ResNet-18
    vgg16     : torchvision VGG-16 with BatchNorm
    inception : torchvision GoogLeNet (Inception-v1), no auxiliary heads
    For CIFAR-sized inputs the stems / heads are adapted to 32x32 images.
    """
    if arch == "resnet18":
        model = tvm.resnet18(num_classes=num_classes)
        if cifar_style:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()

    elif arch == "vgg16":
        model = tvm.vgg16_bn(num_classes=num_classes)
        if cifar_style:
            model.avgpool = nn.AdaptiveAvgPool2d(1)  # 32x32 -> 1x1x512 after 5 pools
            model.classifier = nn.Sequential(
                nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(0.5),
                nn.Linear(512, num_classes),
            )

    elif arch == "inception":
        model = tvm.googlenet(aux_logits=False, init_weights=True,
                              num_classes=num_classes)
        if cifar_style:
            model.conv1 = BasicConv2d(3, 64, kernel_size=3, stride=1, padding=1)
            model.maxpool1 = nn.Identity()

    else:
        raise ValueError(f"Unknown arch: {arch!r}. Choose from {ARCHS}.")
    return model


# ───────────────────────────── Train / eval ──────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(inputs)
        total_loss += criterion(outputs, targets).item() * targets.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        n += targets.size(0)
    return total_loss / n, correct / n


def main():
    p = argparse.ArgumentParser(description="Train full reference model for data selection")
    p.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100", "imagenet"])
    p.add_argument("--arch", default="resnet18", choices=ARCHS)
    p.add_argument("--data_dir", type=str, default=None,
                   help="Dataset root. Defaults to $SLURM_TMPDIR (sub-folders are searched).")
    p.add_argument("-seed", "--seed", default=42, type=int)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--download", action="store_true", default=False,
                   help="Download CIFAR if not present (not supported for ImageNet)")
    p.add_argument("--batch_size", default=128, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--lr", default=0.01, type=float)
    p.add_argument("--epochs", default=None, type=int,
                   help="Override the per-dataset default (300 CIFAR / 350 ImageNet)")
    p.add_argument("--resume", action="store_true", default=False,
                   help="Resume from the latest checkpoint in --save_dir if one exists")
    args = p.parse_args()

    set_seed(args.seed)

    # ── 1. Directory & logging ───────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(os.path.join(args.save_dir, "train.log"))],
    )
    logger = logging.getLogger(__name__)

    # ── 2. Data ──────────────────────────────────────────────────────────────
    cfg = get_dataset_config(args.dataset, args.data_dir, download=args.download)
    logger.info(f"Dataset={args.dataset} (root={cfg['root']}) | Arch={args.arch} | "
                f"Saving to: {args.save_dir}")

    n_total = len(cfg["full_train"])
    n_val = int(0.1 * n_total)
    n_train = n_total - n_val
    split = random_split(range(n_total), [n_train, n_val],
                         generator=torch.Generator().manual_seed(args.seed))
    train_idx, val_idx = list(split[0].indices), list(split[1].indices)

    # Train subset uses augmentation; val subset uses the deterministic eval transform.
    trainset = Subset(cfg["full_train"], train_idx)
    valset = Subset(cfg["full_train_eval"], val_idx)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=args.num_workers)
    valloader = DataLoader(valset, batch_size=256, shuffle=False,
                           pin_memory=True, num_workers=args.num_workers)

    # Save the split and config once, so the trajectory can be reproduced exactly.
    atomic_json({"train_indices": train_idx, "val_indices": val_idx},
                os.path.join(args.save_dir, "split_indices.json"))
    atomic_json({**vars(args), "num_epochs": args.epochs or cfg["num_epochs"],
                 "num_classes": cfg["num_classes"], "data_root": cfg["root"]},
                os.path.join(args.save_dir, "config.json"))

    # ── 3. Model & optimizer ─────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_EPOCHS = args.epochs or cfg["num_epochs"]

    model = build_model(args.arch, cfg["num_classes"], cfg["cifar_style"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # ── 4. Metrics history (rewritten every epoch) ───────────────────────────
    metrics = {k: [] for k in ("epoch", "lr", "train_loss", "train_acc",
                               "val_loss", "val_acc", "epoch_seconds")}
    metrics["device"] = torch.cuda.get_device_name() if device == "cuda" else "cpu"
    metrics_path = os.path.join(args.save_dir, "metrics.json")
    start_epoch = 1

    if args.resume:
        ckpts = sorted(glob.glob(os.path.join(args.save_dir, "checkpoint_*.pth")))
        if ckpts:
            ckpt = torch.load(ckpts[-1], map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            set_rng_state(ckpt["rng_state"])
            metrics = ckpt["metrics"]
            start_epoch = ckpt["epoch"] + 1
            logger.info(f"Resumed from {ckpts[-1]} (next epoch: {start_epoch})")

    if start_epoch == 1:
        # Epoch 0 = initialization, useful as the start of the trajectory.
        atomic_save({
            "epoch": 0, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "lr": optimizer.param_groups[0]["lr"],
            "rng_state": get_rng_state(), "metrics": metrics, "args": vars(args),
        }, os.path.join(args.save_dir, "checkpoint_000.pth"))

    logger.info(f"Starting training on {device} for {NUM_EPOCHS} epochs...")

    # ── 5. Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        lr_used = optimizer.param_groups[0]["lr"]  # LR used during this epoch
        model.train()
        running_loss, correct, seen = 0.0, 0, 0

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            seen += targets.size(0)

        scheduler.step()
        train_loss, train_acc = running_loss / seen, correct / seen
        val_loss, val_acc = evaluate(model, valloader, criterion, device)

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        for k, v in (("epoch", epoch), ("lr", lr_used), ("train_loss", train_loss),
                     ("train_acc", train_acc), ("val_loss", val_loss),
                     ("val_acc", val_acc), ("epoch_seconds", elapsed)):
            metrics[k].append(v)
        metrics["total_seconds"] = sum(metrics["epoch_seconds"])
        atomic_json(metrics, metrics_path)

        logger.info(f"Epoch {epoch:03d}/{NUM_EPOCHS} | lr {lr_used:.6f} | "
                    f"train {train_loss:.4f}/{train_acc:.4f} | "
                    f"val {val_loss:.4f}/{val_acc:.4f} | {elapsed:.1f}s")

        # Save everything, every epoch.
        save_path = os.path.join(args.save_dir, f"checkpoint_{epoch:03d}.pth")
        atomic_save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),   # momentum buffers + current lr
            "scheduler": scheduler.state_dict(),   # last_epoch, base_lrs, T_max, ...
            "lr": lr_used,
            "rng_state": get_rng_state(),
            "metrics": metrics,                    # history up to this epoch
            "args": vars(args),
        }, save_path)
        logger.info(f"--> Saved checkpoint: {save_path}")

    # ── 6. Final test evaluation ─────────────────────────────────────────────
    testloader = DataLoader(cfg["testset"], batch_size=256, shuffle=False,
                            pin_memory=True, num_workers=args.num_workers)
    test_loss, test_acc = evaluate(model, testloader, criterion, device)
    metrics["final_test_loss"], metrics["final_test_acc"] = test_loss, test_acc
    atomic_json(metrics, metrics_path)
    logger.info(f"Finished trajectory training. Test loss {test_loss:.4f} | acc {test_acc:.4f}")


if __name__ == "__main__":
    main()