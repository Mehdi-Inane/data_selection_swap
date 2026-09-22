"""
Compare CIFAR-10 data selection methods (GradMatch, Alg3, Random, K-Center)
by training a fresh ResNet-18 model directly on each SELECTED subset 
and evaluating on the test set.

Hyperparameters (per paper setup):
  - Optimizer: SGD (LR=0.01, Momentum=0.9, Weight Decay=5e-4)
  - Scheduler: CosineAnnealingLR (T_max=300)
  - Architecture: ResNet-18
  - Epochs: 300
"""

import logging
import os
import shutil
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# ─── Config ──────────────────────────────────────────────────────────────────
FRAC = 0.3


BUDGET = int(FRAC * 50000) 

INDEX_PATHS: Dict[str, str] = {
    "GradMatch": f"gradmatch_cifar10_idxs_budget_{BUDGET}.npy",
    "Alg3": "/home/mila/a/ahmedm/scratch/data_selection_swap/cifar10/selection/20000_points/alg3_indices.npy",
    "Random": "/home/mila/a/ahmedm/scratch/data_selection_swap/cifar10/selection/20000_points/random_indices.npy",
    "K-Center": "/home/mila/a/ahmedm/scratch/data_selection_swap/cifar10/selection/20000_points/kcenter_indices.npy",
}

EPOCHS = 300
BATCH_SIZE = 128
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_CLASSES = 10
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Dataset Setup ───────────────────────────────────────────────────────────

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Copy raw tarball to SLURM_TMPDIR once
source_tar = "/network/datasets/cifar10/cifar-10-python.tar.gz"
data_dir = os.environ.get("SLURM_TMPDIR", "./data")
target_tar = os.path.join(data_dir, "cifar-10-python.tar.gz")

if not os.path.exists(target_tar):
    logger.info(f"Copying dataset to local storage ({data_dir})...")
    os.makedirs(data_dir, exist_ok=True)
    shutil.copy(source_tar, target_tar)

full_train = torchvision.datasets.CIFAR10(
    root=data_dir, train=True, download=True, transform=train_transform
)
test_set = torchvision.datasets.CIFAR10(
    root=data_dir, train=False, download=True, transform=test_transform
)

test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
)

# ─── Helper Functions ────────────────────────────────────────────────────────

def get_fresh_model() -> nn.Module:
    """Instantiates a fresh ResNet-18 model."""
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * targets.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, 100.0 * correct / total


def train_on_subset(method_name: str, idxs_path: str) -> Tuple[float, int]:
    """Loads specified subset indices and trains on them directly."""
    logger.info("=" * 60)
    logger.info(f"Starting training for method: {method_name} (Training on SUBSET)")
    logger.info("=" * 60)

    if not os.path.exists(idxs_path):
        logger.error(f"File not found: {idxs_path}. Skipping {method_name}.")
        return 0.0, 0

    # Load indices for the subset
    selected_idxs = np.load(idxs_path).tolist()
    subset_size = len(selected_idxs)
    selected_subset = Subset(full_train, selected_idxs)

    train_loader = DataLoader(
        selected_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    logger.info(f"Loaded {subset_size} samples for '{method_name}'.")
    logger.info(f"Training on selected subset for {EPOCHS} epochs.")

    # Fresh model & optimizer instance
    model = get_fresh_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    best_test_acc = 0.0
    checkpoint_name = f"best_model_subset_{method_name.lower().replace('-', '_')}.pth"

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = eval_epoch(model, test_loader, criterion)
        scheduler.step()

        is_best = test_acc > best_test_acc
        if is_best:
            best_test_acc = test_acc
            torch.save(model.state_dict(), checkpoint_name)

        if epoch % 25 == 0 or epoch == EPOCHS or is_best:
            logger.info(
                f"[{method_name}] Epoch {epoch:3d}/{EPOCHS} | "
                f"Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | "
                f"Test Loss {test_loss:.4f} Acc {test_acc:.2f}%"
                + (" ← best" if is_best else "")
            )

    logger.info(f"Finished {method_name} (Subset). Best Test Accuracy: {best_test_acc:.2f}%\n")
    return best_test_acc, subset_size

# ─── Main Driver ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}

    for method_name, idxs_path in INDEX_PATHS.items():
        best_acc, sample_count = train_on_subset(method_name, idxs_path)
        results[method_name] = {
            "best_acc": best_acc,
            "samples": sample_count,
        }

    # ─── Final Summary Table ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info(f"{'Selection Method':<18} | {'Subset Size':<12} | {'Best Test Acc (%)':<18}")
    logger.info("-" * 60)
    for method, info in results.items():
        logger.info(f"{method:<18} | {info['samples']:<12d} | {info['best_acc']:<18.2f}")
    logger.info("=" * 60)