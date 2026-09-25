"""
train_gradmatch_selection.py
GradMatch baseline (Killamsetty et al., 2021a) for CIFAR-10, CIFAR-100 & ImageNet.

Can be run interactively (no SLURM required):

    python train_gradmatch_selection.py \
        --dataset     cifar10           \
        --fraction    0.1               \
        --seed        42                \
        --data_dir    ./data            \
        --output_dir  ./outputs         \
        --download                      \
        --linear_layer

Matches the paper's Appendix C.2 protocol: SGD lr=0.01, momentum=0.9,
weight_decay=5e-4, nesterov, cosine annealing, 300 epochs (CIFAR-10/100)
/ 350 epochs (ImageNet), R=20, kappa=0.5 (GRAD-MATCH-WARM), lam=0.5.
"""

import argparse
import json
import logging
import os
import random
import warnings

import numpy as np
import torch
from typing import Optional
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from dotmap import DotMap
from torch.utils.data import DataLoader, random_split

from cords.utils.data.dataloader.SL.adaptive import GradMatchDataLoader
from cords.utils.models import ResNet18


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Dataset configuration
# ─────────────────────────────────────────────────────────────────────────────

def get_dataset_config(dataset: str, data_dir: str, download: bool = False) -> dict:
    """Return per-dataset hyper-params, transforms, and dataset objects."""

    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)
        tf_tr = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        tf_te = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return dict(
            num_classes=10,
            num_epochs=300,
            cifar_style=True,
            full_train=datasets.CIFAR10(
                root=data_dir, train=True,  download=download, transform=tf_tr),
            testset=datasets.CIFAR10(
                root=data_dir, train=False, download=download, transform=tf_te),
        )

    elif dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std  = (0.2675, 0.2565, 0.2761)
        tf_tr = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        tf_te = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return dict(
            num_classes=100,
            num_epochs=300,
            cifar_style=True,
            full_train=datasets.CIFAR100(
                root=data_dir, train=True,  download=download, transform=tf_tr),
            testset=datasets.CIFAR100(
                root=data_dir, train=False, download=download, transform=tf_te),
        )

    elif dataset == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)
        tf_tr = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        tf_te = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return dict(
            num_classes=1000,
            num_epochs=350,
            cifar_style=False,
            full_train=datasets.ImageFolder(
                os.path.join(data_dir, 'train'), transform=tf_tr),
            testset=datasets.ImageFolder(
                os.path.join(data_dir, 'val'),   transform=tf_te),
        )

    else:
        raise ValueError(
            f"Unknown dataset: {dataset!r}. "
            "Choose 'cifar10', 'cifar100', or 'imagenet'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Output-directory resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_base_dir(output_dir: Optional[str],
                     dataset: str, budget: int, seed: int) -> str:
    """
    Resolve where artefacts are written, in priority order:

      1. --output_dir (explicit, works everywhere including interactive runs)
      2. $SCRATCH     (set automatically on Mila/SLURM clusters)
      3. ./outputs    (safe local fallback, no env-var required)
    """
    if output_dir:
        root = output_dir
    elif "SCRATCH" in os.environ:
        root = os.path.join(os.environ["SCRATCH"], "gradmatch_baseline")
    else:
        root = os.path.join(os.getcwd(), "outputs", "gradmatch_baseline")

    base = os.path.join(root, dataset, str(budget), f"seed_{seed}")
    os.makedirs(base, exist_ok=True)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    # ── Core ──────────────────────────────────────────────────────────────────
    p.add_argument('--fraction',      default=0.3,        type=float,
                   help='Subset fraction of the training set.')
    p.add_argument('--seed', '-seed', default=42,         type=int,
                   help='Global random seed.')
    p.add_argument('--dataset',       default='cifar10',
                   choices=['cifar10', 'cifar100', 'imagenet'])
    p.add_argument('--data_dir',      required=True,      type=str,
                   help='Root directory of the dataset.  '
                        'For CIFAR-10/100, this is the folder that contains '
                        '(or will contain after --download) the '
                        'cifar-10-batches-py/ subdirectory.  '
                        'For ImageNet, must have train/ and val/ sub-folders.')
    p.add_argument('--output_dir',    default=None,       type=str,
                   help='Where to write logs, model checkpoints, and metrics.  '
                        'Defaults to $SCRATCH/gradmatch_baseline (on cluster) '
                        'or ./outputs/gradmatch_baseline (local interactive run).')
    p.add_argument('--batch_size',    default=128,        type=int)
    p.add_argument('--lr',            default=0.01,       type=float)
    p.add_argument('--num_workers',   default=4,          type=int)
    p.add_argument('--download',      action='store_true', default=False,
                   help='Let torchvision download the dataset into --data_dir '
                        'if it is not already present.  Safe to set by default '
                        'for interactive runs; leave unset on clusters where '
                        'the data is already staged.')
    p.add_argument('--num_epochs',    default=None,       type=int,
                   help='Override the default epoch count for the dataset '
                        '(useful for quick smoke-tests, e.g. --num_epochs 5).')

    # ── GradMatch / OMP hyperparameters (paper Appendix C.2–C.3) ─────────────
    p.add_argument('--select_every',  default=20,         type=int,
                   help='R: epochs between OMP subset re-selections.')
    p.add_argument('--kappa',         default=0.5,        type=float,
                   help='Warm-start fraction κ.  '
                        '0.0 = plain GRAD-MATCH; '
                        '0.5 = GRAD-MATCH-WARM (paper headline numbers).')
    p.add_argument('--lam',           default=0.5,        type=float,
                   help='OMP regularisation coefficient λ.')
    p.add_argument('--eps',           default=1e-10,      type=float,
                   help='OMP gradient-error tolerance ε.')
    p.add_argument('--linear_layer',  action='store_true', default=False,
                   help='[CRITICAL] Restrict OMP to last-layer gradients.  '
                        'Must be set to reproduce paper results.  '
                        'Omitting it makes OMP intractable on ResNet-18 and '
                        'causes selection quality to collapse below random at '
                        'small fractions.')
    p.add_argument('--valid',         action='store_true', default=False,
                   help='Match against validation gradients instead of training '
                        'gradients (class-imbalance setting only).')

    args = p.parse_args()

    # ── Guard: warn loudly if --linear_layer was forgotten ───────────────────
    if not args.linear_layer:
        warnings.warn(
            "\n"
            "  --linear_layer is NOT set.\n"
            "  OMP will run in full-network gradient space (~11 M dims for\n"
            "  ResNet-18).  This is computationally intractable and causes\n"
            "  selection quality to fall below random at small fractions.\n"
            "  Add --linear_layer to your command to reproduce the paper.\n",
            RuntimeWarning,
            stacklevel=2,
        )

    set_seed(args.seed)

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    cfg         = get_dataset_config(args.dataset, args.data_dir, download=args.download)
    NUM_CLASSES = cfg['num_classes']
    NUM_EPOCHS  = args.num_epochs if args.num_epochs is not None else cfg['num_epochs']

    n_val   = int(0.1 * len(cfg['full_train']))
    n_train = len(cfg['full_train']) - n_val
    budget  = int(args.fraction * n_train)

    base_dir = resolve_base_dir(args.output_dir, args.dataset, budget, args.seed)

    # ── 2. Logging ────────────────────────────────────────────────────────────
    log_path = os.path.join(base_dir, f"gradmatch_{budget}_training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),          # also prints to terminal
        ],
    )
    logger = logging.getLogger(__name__)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(
        f"dataset={args.dataset}  budget={budget}  seed={args.seed}  "
        f"epochs={NUM_EPOCHS}  kappa={args.kappa}  "
        f"linear_layer={args.linear_layer}  device={device}"
    )
    logger.info(f"Artefacts will be written to: {base_dir}")

    BATCH_SIZE = args.batch_size
    LR         = args.lr

    MODEL_SAVE_PATH   = os.path.join(base_dir, f"gradmatch_{budget}_model.pth")
    INDICES_SAVE_PATH = os.path.join(base_dir, f"gradmatch_{budget}_indices.pt")
    WEIGHTS_SAVE_PATH = os.path.join(base_dir, f"gradmatch_{budget}_weights.pt")
    METRICS_SAVE_PATH = os.path.join(base_dir, f"gradmatch_{budget}_metrics.json")

    # ── 3. Reproducible train / val split ────────────────────────────────────
    trainset, valset = random_split(
        cfg['full_train'], [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    trainloader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=(device == 'cuda'), num_workers=args.num_workers,
    )
    valloader = DataLoader(
        valset, batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=(device == 'cuda'), num_workers=args.num_workers,
    )
    testloader = DataLoader(
        cfg['testset'], batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=(device == 'cuda'), num_workers=args.num_workers,
    )

    # ── 4. Model ──────────────────────────────────────────────────────────────
    model = ResNet18(num_classes=NUM_CLASSES)
    if cfg['cifar_style']:
        model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model = model.to(device)

    criterion_nored = nn.CrossEntropyLoss(reduction='none')

    optimizer = optim.SGD(
        model.parameters(), lr=LR,
        momentum=0.9, weight_decay=5e-4, nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # ── 5. GradMatch DataLoader ───────────────────────────────────────────────
    dss_args = DotMap({
        'type'           : 'GradMatch',
        'model'          : model,
        'loss'           : criterion_nored,
        'eta'            : LR,
        'num_classes'    : NUM_CLASSES,
        'num_epochs'     : NUM_EPOCHS,
        'device'         : device,
        'valid'          : args.valid,
        'fraction'       : args.fraction,
        'select_every'   : args.select_every,
        'kappa'          : args.kappa,
        'linear_layer'   : args.linear_layer,   # True required for paper results
        'selection_type' : 'PerClassPerGradient',
        'greedy'         : 'Stochastic',
        'collate_fn'     : None,
        # known gradient accumulation bug in several released commits.
        'lam'            : args.lam,
        'eps'            : args.eps,
        'v1'             : False,
    })

    logger.info(
        f"GradMatch dss_args: select_every={args.select_every}  "
        f"kappa={args.kappa}  lam={args.lam}  "
        f"linear_layer={args.linear_layer}  isValid={args.valid}"
    )

    dataloader = GradMatchDataLoader(
        trainloader, valloader, dss_args, logger,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(device == 'cuda'),
    )

    # ── 6. Training loop ──────────────────────────────────────────────────────
    logger.info("Starting GradMatch training loop ...")
    accuracy_history: dict[int, float] = {}

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for inputs, targets, weights in dataloader:
            inputs  = inputs.to(device)
            targets = targets.to(device)
            weights = weights.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            losses  = criterion_nored(outputs, targets)
            loss    = torch.dot(losses, weights / weights.sum())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        if epoch % 50 == 0 or epoch == NUM_EPOCHS:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in testloader:
                    x, y = x.to(device), y.to(device)
                    pred  = model(x).argmax(dim=1)
                    correct += pred.eq(y).sum().item()
                    total   += y.size(0)

            acc      = 100.0 * correct / total
            avg_loss = running_loss / len(dataloader)
            accuracy_history[epoch] = acc
            logger.info(
                f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
                f"Test Acc: {acc:.2f}% | Train Loss: {avg_loss:.4f}"
            )

    # ── 7. Save artefacts ─────────────────────────────────────────────────────
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info(f"Saved model weights  → {MODEL_SAVE_PATH}")

    idxes  = torch.tensor(dataloader.subset_indices)
    gammas = torch.tensor(dataloader.subset_weights)
    torch.save(idxes,  INDICES_SAVE_PATH)
    torch.save(gammas, WEIGHTS_SAVE_PATH)
    logger.info(f"Saved subset indices → {INDICES_SAVE_PATH}")
    logger.info(f"Saved subset weights → {WEIGHTS_SAVE_PATH}")

    with open(METRICS_SAVE_PATH, 'w') as f:
        json.dump(accuracy_history, f, indent=4)
    logger.info(f"Saved accuracy log   → {METRICS_SAVE_PATH}")
    logger.info(f"All artefacts saved to: {base_dir}")


if __name__ == "__main__":
    main()