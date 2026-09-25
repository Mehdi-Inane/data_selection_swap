"""
selection_common.py
Shared infrastructure for the data-selection baseline comparison suite.

train_kl_selection.py (the reference "KL-Faithful" algorithm) is the
source of truth for the experimental protocol. Every baseline script
in this suite —

    train_herding_selection.py
    train_moderate_coreset_selection.py
    train_moso_selection.py
    train_el2n_selection.py
    train_graphcut_facloc_selection.py

— imports from here, so that dataset splits, model architecture,
optimizer/schedule, and evaluation protocol are byte-for-byte identical
across methods. The ONLY thing that is allowed to differ between
scripts is the selection rule that produces `selected_indices`.

See EXPERIMENTAL_PROTOCOL.md for the reasoning behind what is shared
and what is deliberately allowed to differ (e.g. EL2N's need for K
independently-initialised short probe runs, vs. MoSo / KL-Faithful's
shared use of the same training trajectory, vs. Herding / Moderate
Coreset / GraphCut / Facility Location's use of a single converged
embedding).
"""

import glob
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split

from cords.utils.models import ResNet18

FEAT_DIM = 512  # ResNet18 penultimate layer width, all datasets below


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Dataset configuration  (identical to train_kl_selection.py)
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Model  (identical CIFAR-stem swap as train_kl_selection.py)
# ─────────────────────────────────────────────────────────────────────────────
def build_model(cfg: dict, num_classes: int, device: str) -> nn.Module:
    model = ResNet18(num_classes=num_classes)
    if cfg['cifar_style']:
        model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders — identical split logic/seed across every method
# ─────────────────────────────────────────────────────────────────────────────
def make_loaders(cfg: dict, seed: int, batch_size: int, num_workers: int):
    n_val   = int(0.1 * len(cfg['full_train']))
    n_train = len(cfg['full_train']) - n_val
    trainset, _ = random_split(
        cfg['full_train'], [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    # shuffle=False: sample i of eval_loader always corresponds to selected_indices[i]
    eval_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False,
                              pin_memory=True, num_workers=num_workers)
    testloader = DataLoader(cfg['testset'], batch_size=batch_size, shuffle=False,
                             pin_memory=True, num_workers=num_workers)
    return trainset, eval_loader, testloader, n_train


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoints
# ─────────────────────────────────────────────────────────────────────────────
def list_checkpoints(checkpoint_dir: str):
    if not os.path.isdir(checkpoint_dir):
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
    ckpt_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pth")))
    if not ckpt_paths:
        raise ValueError(f"No .pth checkpoints in {checkpoint_dir}")
    return ckpt_paths


def resolve_embedding_checkpoint(checkpoint_dir: str, override: str = None) -> str:
    """
    One-shot methods (Herding / Moderate Coreset / GraphCut / Facility
    Location) need a single reference embedding, not a full trajectory.
    Default: the LAST checkpoint in --checkpoint_dir, i.e. exactly the
    converged network the KL-Faithful / MoSo trajectory also ends on —
    this is the fairest single point of comparison, since it costs no
    extra training compute beyond what every other method already uses.
    """
    if override is not None:
        if not os.path.isfile(override):
            raise ValueError(f"--embedding_checkpoint not found: {override}")
        return override
    return list_checkpoints(checkpoint_dir)[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Feature / probability extraction
#   Every baseline that needs penultimate features and/or softmax
#   probabilities reads them through this single hook-based pass, so
#   "what the model sees" is defined identically everywhere (same hook
#   point as extract_factored_features in train_kl_selection.py).
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_probs_features_targets(model: nn.Module, dataloader, device: str):
    """One forward pass -> (probs [n,C], phi [n,F], targets [n])."""
    model.eval()
    feats = {}

    def _hook(_, inp, __):
        feats['phi'] = inp[0]

    handle = model.linear.register_forward_hook(_hook)

    probs_buf, phi_buf, targets_buf = [], [], []
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs_buf.append(torch.softmax(outputs, dim=1).cpu())
        phi_buf.append(feats['phi'].cpu())
        targets_buf.append(targets.clone())

    handle.remove()
    return torch.cat(probs_buf), torch.cat(phi_buf), torch.cat(targets_buf)


# ─────────────────────────────────────────────────────────────────────────────
# Per-class budget helper
#   Used by methods that are natively defined per class (Herding,
#   Moderate Coreset, GraphCut / Facility Location), so every class
#   contributes budget * (class_count / n_train) samples, matching the
#   public reference implementations linked for each method. Uses
#   largest-remainder rounding so quotas sum exactly to `budget`.
# ─────────────────────────────────────────────────────────────────────────────
def per_class_budget(targets: torch.Tensor, num_classes: int, budget: int) -> torch.Tensor:
    counts = torch.bincount(targets, minlength=num_classes).float()
    raw    = counts * (budget / counts.sum())
    quota  = raw.floor().long()
    remainder = int(budget - quota.sum().item())
    if remainder > 0:
        frac_order = torch.argsort(raw - quota.float(), descending=True)
        for c in frac_order[:remainder]:
            quota[c] += 1
    return quota  # LongTensor [num_classes]


# ─────────────────────────────────────────────────────────────────────────────
# Save-path / logging convention (one line change — method name — vs.
# train_kl_selection.py's own base_dir / logging setup)
# ─────────────────────────────────────────────────────────────────────────────
def get_save_paths(method: str, dataset: str, budget: int, seed: int) -> dict:
    scratch_path = os.environ.get("SCRATCH", "/home/mila/a/ahmedm/scratch")
    base_dir = f"{scratch_path}/data_selection_baselines/{method}/{dataset}/{budget}/seed_{seed}"
    os.makedirs(base_dir, exist_ok=True)
    return dict(
        base_dir=base_dir,
        model  =os.path.join(base_dir, f"{method}_{budget}_model.pth"),
        indices=os.path.join(base_dir, f"{method}_{budget}_indices.pt"),
        metrics=os.path.join(base_dir, f"{method}_{budget}_metrics.json"),
        scores =os.path.join(base_dir, f"{method}_{budget}_scores.json"),
        log    =os.path.join(base_dir, f"{method}_{budget}_training.log"),
    )


def get_logger(name: str, log_path: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return logging.getLogger(name)


# ─────────────────────────────────────────────────────────────────────────────
# Shared argparser — every baseline adds only its method-specific flags
# ─────────────────────────────────────────────────────────────────────────────
def base_argparser(description: str):
    import argparse
    p = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fraction',       default=0.3,         type=float)
    p.add_argument('--seed', '-seed',  default=42,          type=int)
    p.add_argument('--checkpoint_dir', required=True,       type=str,
                   help='Same trajectory directory used for train_kl_selection.py')
    p.add_argument('--dataset',        default='cifar100',  choices=['cifar100', 'imagenet'])
    p.add_argument('--data_dir',       required=True,       type=str)
    p.add_argument('--batch_size',     default=128,         type=int)
    p.add_argument('--lr',             default=0.01,        type=float,
                   help='Learning rate for the final retrain-on-subset stage')
    p.add_argument('--num_workers',    default=4,           type=int)
    p.add_argument('--download',       action='store_true', default=False)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Shared retrain-on-subset loop (identical to steps 8-9 of
# train_kl_selection.py) — the only stage that consumes GPU-hours
# proportional to the *final* budget, run identically for every method
# so that accuracy differences trace back to `selected_indices` alone.
# ─────────────────────────────────────────────────────────────────────────────
def retrain_on_subset(cfg, trainset, selected_indices, device, args, logger, paths):
    trainloader = DataLoader(
        Subset(trainset, selected_indices),
        batch_size=args.batch_size, shuffle=True, pin_memory=True,
        num_workers=args.num_workers,
    )
    testloader = DataLoader(cfg['testset'], batch_size=args.batch_size, shuffle=False,
                             pin_memory=True, num_workers=args.num_workers)

    model = build_model(cfg, cfg['num_classes'], device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                           weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_epochs'])

    logger.info(f"Retraining on subset of size {len(selected_indices)} ...")
    accuracy_history = {}

    for epoch in range(1, cfg['num_epochs'] + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        if epoch % 50 == 0 or epoch == cfg['num_epochs']:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in testloader:
                    x, y = x.to(device), y.to(device)
                    correct += model(x).argmax(1).eq(y).sum().item()
                    total   += y.size(0)
            acc = 100.0 * correct / total
            accuracy_history[epoch] = acc
            logger.info(f"Epoch {epoch:03d}/{cfg['num_epochs']} | "
                        f"Test Acc: {acc:.2f}% | "
                        f"Train Loss: {running_loss / len(trainloader):.4f}")

    torch.save(model.state_dict(), paths['model'])
    with open(paths['metrics'], 'w') as fh:
        json.dump(accuracy_history, fh, indent=4)
    logger.info(f"Saved all artefacts to {paths['base_dir']}")
    return accuracy_history