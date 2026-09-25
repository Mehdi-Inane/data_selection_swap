import argparse
import json
import os
import random
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Assuming you are using the same ResNet18 from cords
from cords.utils.models import ResNet18

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataset_config(dataset: str, data_dir: str, download: bool = False) -> dict:
    """Return per-dataset hyper-params, transforms, and dataset objects."""
    if dataset == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        tf_tr = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        tf_te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return dict(
            num_classes=100, num_epochs=300, cifar_style=True,
            full_train=datasets.CIFAR100(root=data_dir, train=True,  download=download, transform=tf_tr),
            testset   =datasets.CIFAR100(root=data_dir, train=False, download=download, transform=tf_te),
        )

    elif dataset == 'imagenet':
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
            num_classes=1000, num_epochs=350, cifar_style=False,
            full_train=datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=tf_tr),
            testset   =datasets.ImageFolder(os.path.join(data_dir, 'val'),   transform=tf_te),
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose 'cifar100' or 'imagenet'.")

def main():
    p = argparse.ArgumentParser(description="Train full reference model for KL-Faithful selection")
    p.add_argument('--dataset', default='cifar100', choices=['cifar100', 'imagenet'])
    p.add_argument('--data_dir', required=True, type=str,
                   help='Root of the staged dataset (e.g. $SLURM_TMPDIR/cifar100_data)')
    p.add_argument('-seed', '--seed', default=42, type=int)
    p.add_argument('--save_dir', type=str, required=True,
                   help='Directory to save model checkpoints')
    p.add_argument('--download', action='store_true', default=False,
                   help='Download dataset if not present')
    p.add_argument('--batch_size', default=128, type=int)
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--lr', default=0.01, type=float)
    args = p.parse_args()

    set_seed(args.seed)

    # ── 1. Directory & Logging Setup ──────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(args.save_dir, "train.log"))]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Dataset={args.dataset} | Saving checkpoints to: {args.save_dir}")

    # ── 2. Data Preparation & Split ────────────────────────────────────────────
    cfg = get_dataset_config(args.dataset, args.data_dir, download=args.download)
    full_train = cfg['full_train']
    
    n_val   = int(0.1 * len(full_train))
    n_train = len(full_train) - n_val

    # Ensure the exact same trainset is used to compute the trajectory
    trainset, valset = random_split(
        full_train, 
        [n_train, n_val], 
        generator=torch.Generator().manual_seed(args.seed)
    )

    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, 
        pin_memory=True, num_workers=args.num_workers
    )

    # ── 3. Model & Optimizer Setup ─────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = cfg['num_classes']
    NUM_EPOCHS = cfg['num_epochs']
    
    model = ResNet18(num_classes=NUM_CLASSES)
    if cfg['cifar_style']:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    logger.info(f"Starting full-dataset training on {device} for {NUM_EPOCHS} epochs...")

    # ── 4. Training Loop & Checkpoint Saving ───────────────────────────────────
    # Per-epoch wall-clock lets aggregate_selection_comparison.py charge each
    # selection method for the prefix of this trajectory it consumes.
    epoch_seconds = []
    timing_path = os.path.join(args.save_dir, "train_time.json")
    for epoch in range(1, NUM_EPOCHS + 1):
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.train()
        running_loss = 0.0

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        if device == 'cuda':
            torch.cuda.synchronize()
        epoch_seconds.append(time.perf_counter() - t0)
        with open(timing_path, 'w') as fh:
            json.dump({"epoch_seconds": epoch_seconds,
                       "total_seconds": sum(epoch_seconds),
                       "device": torch.cuda.get_device_name() if device == 'cuda' else 'cpu'},
                      fh, indent=4)
        avg_loss = running_loss / len(trainloader)

        # Logging
        if epoch % 10 == 0 or epoch <= 10:
            logger.info(f"Epoch {epoch:03d}/{NUM_EPOCHS} | Train Loss: {avg_loss:.4f}")

        # Checkpoint Saving Logic: epochs 1 to 10, then every 10 epochs
        if epoch <= 10 or epoch % 10 == 0:
            # Zero-pad epoch number so glob.glob() sorts them correctly alphabetically
            save_path = os.path.join(args.save_dir, f"checkpoint_{epoch:03d}.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"--> Saved checkpoint: {save_path}")

    logger.info("Finished trajectory training.")

if __name__ == "__main__":
    main()