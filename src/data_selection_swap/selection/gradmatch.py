"""GradMatch baseline for CIFAR-10, CIFAR-100, and ImageNet.

Can be run interactively (no SLURM required):

    python scripts/select/gradmatch.py \
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
import math
import os
import time
import warnings

import torch
from typing import Optional
import torch.nn as nn
import torch.optim as optim
from dotmap import DotMap
from torch.utils.data import DataLoader

from cords.utils.data.dataloader.SL.adaptive import GradMatchDataLoader

from .. import common
from ..paths import default_results_root


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Backward-compatible alias for the shared seed helper."""
    common.set_seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset configuration
# ─────────────────────────────────────────────────────────────────────────────

def get_dataset_config(dataset: str, data_dir: str, download: bool = False) -> dict:
    """Backward-compatible alias for the shared dataset configuration."""
    return common.get_dataset_config(dataset, data_dir, download=download)


# ─────────────────────────────────────────────────────────────────────────────
# Output-directory resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_base_dir(output_dir: Optional[str],
                     dataset: str, budget: int, seed: int) -> str:
    """
    Resolve where artefacts are written, in priority order:

      1. --output_dir
      2. RESULTS_ROOT, else SCRATCH, else ./results
    """
    if output_dir:
        root = output_dir
    else:
        root = os.path.join(default_results_root(), "gradmatch_baseline")

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
                   choices=list(common.DATASETS))
    p.add_argument('--architecture',  default=None, choices=list(common.ARCHITECTURES))
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
    p.add_argument('--selection_only', action='store_true', default=False,
                   help='Save the final GradMatch subset without the controlled fresh retrain.')
    p.add_argument('--retrain_optimizer', default='sgd',
                   choices=['sgd', 'adam', 'adamw'])
    p.add_argument('--init_ckpt', default=None,
                   help='Shared initialization for the fresh subset retrain.')
    p.add_argument('--weight_decay', default=5e-4, type=float)
    p.add_argument('--beta1', default=0.9, type=float)
    p.add_argument('--beta2', default=0.999, type=float)
    p.add_argument('--adam_eps', default=1e-8, type=float)

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

    args = p.parse_args()
    if not math.isfinite(args.fraction) or not 0.0 < args.fraction < 1.0:
        p.error('--fraction must be finite and strictly between 0 and 1')

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

    common.set_seed(args.seed)

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    cfg         = common.get_dataset_config(args.dataset, args.data_dir, download=args.download,
                                            architecture=args.architecture)
    NUM_CLASSES = cfg['num_classes']
    NUM_EPOCHS  = args.num_epochs if args.num_epochs is not None else cfg['num_epochs']
    cfg = dict(cfg)
    cfg['num_epochs'] = NUM_EPOCHS

    trainset = common.make_train_set(cfg)
    n_train  = len(trainset)
    budget   = int(args.fraction * n_train)

    if args.output_dir:
        base_dir = resolve_base_dir(args.output_dir, args.dataset, budget, args.seed)
        paths = {
            'base_dir': base_dir,
            'model': os.path.join(base_dir, f'gradmatch_{budget}_model.pth'),
            'indices': os.path.join(base_dir, f'gradmatch_{budget}_indices.pt'),
            'metrics': os.path.join(base_dir, f'gradmatch_{budget}_metrics.json'),
            'timing': os.path.join(base_dir, f'gradmatch_{budget}_timing.json'),
            'log': os.path.join(base_dir, f'gradmatch_{budget}_training.log'),
        }
    else:
        paths = common.get_save_paths(
            'gradmatch', args.dataset, cfg['architecture'], budget, args.seed,
            variant=common.storage_variant(args.retrain_optimizer))
        base_dir = paths['base_dir']

    # ── 2. Logging ────────────────────────────────────────────────────────────
    log_path = paths['log']
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

    ADAPTIVE_MODEL_PATH = os.path.join(base_dir, f"gradmatch_{budget}_adaptive_model.pth")
    INDICES_SAVE_PATH = paths['indices']
    WEIGHTS_SAVE_PATH = os.path.join(base_dir, f"gradmatch_{budget}_weights.pt")
    ADAPTIVE_METRICS_PATH = os.path.join(base_dir, f"gradmatch_{budget}_adaptive_metrics.json")

    # ── 3. Loaders (training set only; no validation split) ──────────────────
    trainloader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=(device == 'cuda'), num_workers=args.num_workers,
    )
    testloader = DataLoader(
        cfg['testset'], batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=(device == 'cuda'), num_workers=args.num_workers,
    )

    # ── 4. Model ──────────────────────────────────────────────────────────────
    model = common.build_model(cfg, NUM_CLASSES, device)

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
        'valid'          : False,   # never match against held-out data
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
        f"linear_layer={args.linear_layer}"
    )

    if device == 'cuda':
        torch.cuda.synchronize()
    selection_t0 = time.perf_counter()
    dataloader = GradMatchDataLoader(
        trainloader, trainloader, dss_args, logger,  # CORDS requires a 'val' loader slot; unused because valid=False
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
    if device == 'cuda':
        torch.cuda.synchronize()
    selection_seconds = time.perf_counter() - selection_t0

    torch.save(model.state_dict(), ADAPTIVE_MODEL_PATH)
    logger.info(f"Saved adaptive-training model weights  → {ADAPTIVE_MODEL_PATH}")

    idxes  = torch.tensor(dataloader.subset_indices)
    gammas = torch.tensor(dataloader.subset_weights)
    torch.save(idxes,  INDICES_SAVE_PATH)
    torch.save(gammas, WEIGHTS_SAVE_PATH)
    logger.info(f"Saved subset indices → {INDICES_SAVE_PATH}")
    logger.info(f"Saved subset weights → {WEIGHTS_SAVE_PATH}")

    common.atomic_json_dump(accuracy_history, ADAPTIVE_METRICS_PATH)
    common.atomic_json_dump(
        {
            'method': 'gradmatch',
            'selection_seconds': selection_seconds,
            'phases_seconds': {'extra_training_and_selection': selection_seconds},
            'reference_epochs_used': 0,
            'adaptive_training_epochs': NUM_EPOCHS,
            'device': torch.cuda.get_device_name() if device == 'cuda' else 'cpu',
        },
        paths['timing'])
    logger.info(f"Saved adaptive accuracy log → {ADAPTIVE_METRICS_PATH}")

    # The adaptive model has seen a sequence of changing weighted subsets, so
    # its accuracy is not the quality of the final saved subset.  Evaluate the
    # final indices with the same fresh, unweighted retrain used by every other
    # method; this is the metrics file consumed by the comparison scripts.
    common.finish(cfg, trainset, idxes, device, args, logger, paths)
    logger.info(f"All artefacts saved to: {base_dir}")


if __name__ == "__main__":
    main()
