"""
selection_common.py
Shared infrastructure for the data-selection baseline comparison suite.

train_kl_selection.py (the reference "KL-Faithful" algorithm) is the
source of truth for the experimental protocol. Every baseline script
in this suite —

    train_random_selection.py
    train_herding_selection.py
    train_moderate_coreset_selection.py
    train_moso_selection.py
    train_el2n_selection.py
    train_submodular_selection.py      (GraphCut / Facility Location)

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

import copy
import glob
import json
import logging
import math
import os
import random
import re
import time
from contextlib import contextmanager

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
def scoring_subset(cfg: dict, trainset: Subset, scoring_transform: str = 'train') -> Subset:
    """
    The dataset view that selection methods *score* (not train on).
    'train' reproduces train_kl_selection.py as-is (augmented views);
    'test' scores un-augmented images, as the Moderate-DS and data_diet
    reference implementations do. Indices are identical either way.
    """
    if scoring_transform == 'train':
        return trainset
    clean = copy.copy(cfg['full_train'])
    clean.transform = cfg['testset'].transform
    return Subset(clean, trainset.indices)


def make_loaders(cfg: dict, seed: int, batch_size: int, num_workers: int,
                 scoring_transform: str = 'train'):
    n_val   = int(0.1 * len(cfg['full_train']))
    n_train = len(cfg['full_train']) - n_val
    trainset, _ = random_split(
        cfg['full_train'], [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    # shuffle=False: sample i of eval_loader always corresponds to selected_indices[i]
    eval_loader = DataLoader(scoring_subset(cfg, trainset, scoring_transform),
                             batch_size=batch_size, shuffle=False,
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


def checkpoint_epoch(path: str) -> int:
    """Epoch number of a train_full_data.py checkpoint (checkpoint_XXX.pth)."""
    m = re.search(r'(\d+)\.pth$', os.path.basename(path))
    if m is None:
        raise ValueError(f"Cannot parse epoch from checkpoint name: {path}")
    return int(m.group(1))


def reference_epochs_used(ckpt_paths) -> int:
    """How far into the shared trajectory a method had to train (0 if none)."""
    epochs = []
    for path in ckpt_paths:
        try:
            epochs.append(checkpoint_epoch(path))
        except ValueError:
            pass
    return max(epochs, default=0)


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


def cosine_lr(base_lr: float, epoch: int, num_epochs: int) -> float:
    """LR used *during* 1-indexed `epoch` by train_full_data.py's CosineAnnealingLR."""
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * (epoch - 1) / num_epochs))


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


def logit_grads(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """L = softmax - onehot: the logit-gradient factor used by KL-Faithful."""
    one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)
    return probs - one_hot


# ─────────────────────────────────────────────────────────────────────────────
# Per-class kernels
#   Herding and GraphCut / Facility Location are pairwise methods run
#   within each class. They can consume either
#     'features'        K = Φ Φ^T                 at the embedding checkpoint
#     'grad'            K = (L L^T) ⊙ (Φ Φ^T)     at the embedding checkpoint
#     'grad_trajectory' K = Σ_t (L_t L_t^T) ⊙ (Φ_t Φ_t^T)  over every checkpoint
#   'grad_trajectory' is exactly the kernel KL-Faithful optimises over,
#   restricted to one class — the "same information" ablation.
# ─────────────────────────────────────────────────────────────────────────────
def build_class_kernels(kind: str, model: nn.Module, eval_loader, checkpoint_dir: str,
                         embedding_checkpoint: str, num_classes: int, device: str):
    """Return (kernels: list[[n_c,n_c] CPU tensor], class_idx: list[LongTensor], ckpts_used)."""
    if kind == 'grad_trajectory':
        ckpts = list_checkpoints(checkpoint_dir)
    else:
        ckpts = [resolve_embedding_checkpoint(checkpoint_dir, embedding_checkpoint)]

    kernels = class_idx = None
    for ckpt in ckpts:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        probs, phi, targets = extract_probs_features_targets(model, eval_loader, device)
        if class_idx is None:
            class_idx = [torch.where(targets == c)[0] for c in range(num_classes)]
            kernels = [None] * num_classes
        L = logit_grads(probs, targets) if kind != 'features' else None
        for c, idx in enumerate(class_idx):
            phi_c = phi[idx].to(device)
            K = phi_c @ phi_c.T
            if L is not None:
                L_c = L[idx].to(device)
                K = K * (L_c @ L_c.T)
            K = K.cpu()
            kernels[c] = K if kernels[c] is None else kernels[c] + K
    return kernels, class_idx, ckpts


def kernel_to_similarity(K: torch.Tensor) -> torch.Tensor:
    """
    Non-negative similarity for submodular functions, following apricot's
    own convention for metric='euclidean':  S = max(D) - D, with D the
    RKHS distance D_ij = sqrt(K_ii + K_jj - 2 K_ij).
    """
    diag = torch.diagonal(K)
    D = (diag.unsqueeze(0) + diag.unsqueeze(1) - 2.0 * K).clamp_min(0.0).sqrt()
    return D.max() - D


# ─────────────────────────────────────────────────────────────────────────────
# Per-class budget helper
#   Used by methods that are natively defined per class (Herding,
#   MoSo, GraphCut / Facility Location), so every class contributes
#   budget * (class_count / n_train) samples, matching the public
#   reference implementations linked for each method. Uses
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


def per_class_topk(scores: torch.Tensor, targets: torch.Tensor, num_classes: int,
                   budget: int) -> torch.Tensor:
    quota = per_class_budget(targets, num_classes, budget)
    selected = []
    for c in range(num_classes):
        idx = torch.where(targets == c)[0]
        q = min(int(quota[c].item()), idx.numel())
        if q > 0:
            selected.append(idx[torch.topk(scores[idx], q).indices])
    return torch.cat(selected)


# ─────────────────────────────────────────────────────────────────────────────
# Short full-data training (EL2N probes, MoSo surrogates)
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        criterion(model(inputs), targets).backward()
        optimizer.step()


# ─────────────────────────────────────────────────────────────────────────────
# Selection-time accounting
#   Wall-clock, CUDA-synchronised, split into phases:
#     extra_training : training the method needs beyond the shared
#                      full-data trajectory (EL2N probes, MoSo surrogates)
#     scoring        : checkpoint loading + forward/backward passes over
#                      the training set to build features/scores/kernels
#     selection      : the combinatorial rule that turns scores into indices
#   `reference_epochs_used` records how much of the shared trajectory the
#   method consumed, so aggregate_selection_comparison.py can add the
#   amortised cost of those epochs from train_full_data.py's timing file.
# ─────────────────────────────────────────────────────────────────────────────
class SelectionTimer:
    def __init__(self, device: str):
        self.device = device
        self.phases = {}

    def _sync(self):
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def phase(self, name: str):
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.phases[name] = self.phases.get(name, 0.0) + time.perf_counter() - t0

    def total(self) -> float:
        return sum(self.phases.values())


def save_timing(path: str, timer: SelectionTimer, logger, **extra):
    record = dict(
        phases_seconds=timer.phases,
        selection_seconds=timer.total(),
        device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',
        **extra,
    )
    with open(path, 'w') as fh:
        json.dump(record, fh, indent=4)
    logger.info(f"Selection time: {timer.total():.1f}s  phases="
                + ", ".join(f"{k}={v:.1f}s" for k, v in timer.phases.items()))


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
        timing =os.path.join(base_dir, f"{method}_{budget}_timing.json"),
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
    p.add_argument('--scoring_transform', default='train', choices=['train', 'test'],
                   help="Transform applied to the training set when scoring it. "
                        "'train' matches train_kl_selection.py; use the same value "
                        "for every method in a comparison.")
    p.add_argument('--selection_only', action='store_true', default=False,
                   help='Stop after saving indices + timing (skip retraining).')
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


def finish(cfg, trainset, selected_indices, device, args, logger, paths):
    """Save indices, then retrain unless --selection_only."""
    torch.save(selected_indices, paths['indices'])
    if args.selection_only:
        logger.info("--selection_only set: skipping retraining.")
        return
    retrain_on_subset(cfg, trainset, selected_indices, device, args, logger, paths)
