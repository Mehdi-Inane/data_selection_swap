"""
GradMatch subset selection on CIFAR-10 using the CORDS library.

Selection type : PerBatch  (global OMP — not per-class)
Returns        : indices of the top-k selected points (pool-relative and
                 global CIFAR-10 positions) plus their OMP weights (gammas).

Usage
-----
  python gradmatch_cifar10.py

Key outputs (saved as .npy)
  gradmatch_pool_idxs.npy     – indices within the training pool (0 … N_POOL-1)
  gradmatch_cifar10_idxs.npy  – same indices mapped back to original CIFAR-10
  gradmatch_gammas.npy        – OMP weights for each selected point
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import shutil
import os


# ─── sklearn compatibility patch ─────────────────────────────────────────────
# CORDS was written against scikit-learn < 1.2 which had `load_boston`.
# Newer sklearn removed it, but its __getattr__ raises ImportError (not
# AttributeError), so hasattr() and try/except AttributeError both fail.
# The only reliable fix is to inject a stub directly into the module __dict__
# *before* any CORDS module is imported — __dict__ is checked before
# __getattr__, so sklearn's error path is never reached.
import sklearn.datasets as _skd

def _load_boston_stub(*args, **kwargs):
    raise RuntimeError(
        "load_boston was removed in scikit-learn >= 1.2. "
        "This stub should never be called for CIFAR-10 / GradMatch."
    )

_skd.__dict__["load_boston"] = _load_boston_stub   # bypass __getattr__
# ─────────────────────────────────────────────────────────────────────────────

from cords.selectionstrategies.SL import GradMatchStrategy

# ─── User-facing config ──────────────────────────────────────────────────────

# Point this to wherever CIFAR-10 is already stored on your cluster.
# torchvision will not re-download if the data already exists there.
DATA_DIR = "/network/datasets/cifar10"   # <-- adjust

FRAC = 0.3


BUDGET          = int(FRAC * 50000)      # k: how many points to select
BATCH_SIZE      = 128        # batch size for both loaders
VAL_FRAC        = 0.1        # fraction of train split held out as validation
ETA             = 0.01       # step-size for the one-step gradient approximation
LINEAR_LAYER    = True       # True  → use last-fc weights+biases gradients
                             # False → use last-fc biases only (faster, less accurate)
VALID           = False      # False → match subset grad against full train grad
                             # True  → match against validation grad (needs labelled val)
V1              = True       # True  → newer, more accurate OMP solver
LAM             = 0.0        # OMP regularisation (0 = none)
EPS             = 1e-4       # OMP convergence tolerance
NUM_CLASSES     = 10
SELECTION_TYPE  = "PerBatch" # global selection (not per-class)
NUM_WORKERS     = 4
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
logger.info(f"Running on device: {DEVICE}")

# ─── Transforms ──────────────────────────────────────────────────────────────

# Standard CIFAR-10 normalisation constants
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

# Light augmentation for the training pool; no augmentation for val/selection
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ─── Dataset split ───────────────────────────────────────────────────────────

# Load the full CIFAR-10 training split (50 000 examples).
# Set download=False if the data is already present; True to auto-fetch.


source_tar = "/network/datasets/cifar10/cifar-10-python.tar.gz"
data_dir = os.environ.get("SLURM_TMPDIR", "./data") # Fallback to ./data if not on a Slurm node
target_tar = os.path.join(data_dir, "cifar-10-python.tar.gz")

# 2. Copy the compressed file to the node's local storage (only if it hasn't been copied yet)
if not os.path.exists(target_tar):
    print(f"Copying dataset from network to {data_dir}...")
    os.makedirs(data_dir, exist_ok=True)
    shutil.copy(source_tar, target_tar)

# 3. Load the dataset
# IMPORTANT: download=True is required here. 
# PyTorch will see the tar.gz file already exists in data_dir, skip the internet download, and just extract it.
full_dataset = torchvision.datasets.CIFAR10(
    root=data_dir,
    train=True,
    download=True, 
    transform=train_transform
)

n_total  = len(full_dataset)           # 50 000
n_val    = int(n_total * VAL_FRAC)    # 5 000
n_pool   = n_total - n_val            # 45 000  ← the pool we select from

pool_idx = list(range(n_pool))
val_idx  = list(range(n_pool, n_total))

train_subset = Subset(full_dataset, pool_idx)

# Validation set uses the eval transform (no random augmentation).
val_dataset = torchvision.datasets.CIFAR10(
    root=data_dir, train=True, download=False, transform=eval_transform
)
val_subset = Subset(val_dataset, val_idx)

logger.info(f"Pool size : {n_pool}  |  Val size : {n_val}  |  Budget : {BUDGET}")

# ─── Dataloaders ─────────────────────────────────────────────────────────────

# IMPORTANT: shuffle=False on trainloader so that the integer indices returned
# by GradMatch.select() map consistently back to positions in pool_idx.
trainloader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,          # must be False for index consistency
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
)
valloader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
)

# ─── CORDS-compatible ResNet-18 wrapper ──────────────────────────────────────
# CORDS's DataSelectionStrategy requires two things the stock torchvision
# ResNet does NOT have:
#
#   1. model.get_embedding_dim()
#        → returns the width of the penultimate (pre-fc) layer (512 for R18)
#
#   2. model(x, last=True, freeze=True)
#        → when last=True  : returns (logits, embedding)
#          when last=False : returns logits only
#        → when freeze=True: backbone runs under torch.no_grad(); only the
#          final fc layer stays in the autograd graph
#
# The wrapper below adds both without touching the pretrained weights.

class CORDSResNet18(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        base = torchvision.models.resnet18()
        emb_dim = base.fc.in_features          # 512

        # Everything up to (but not including) the final fc layer
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool,
        )
        self.fc = nn.Linear(emb_dim, num_classes)
        self._emb_dim = emb_dim

    # ── required by CORDS ────────────────────────────────────────────────────
    def get_embedding_dim(self) -> int:
        return self._emb_dim

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor,
                last: bool = False,
                freeze: bool = False) -> torch.Tensor:
        """
        last   – if True, return (logits, embedding) instead of just logits
        freeze – if True, run the backbone under no_grad (only fc is live)
        """
        if freeze:
            with torch.no_grad():
                emb = self.backbone(x).flatten(1)
        else:
            emb = self.backbone(x).flatten(1)

        out = self.fc(emb)

        if last:
            return out, emb
        return out


model = CORDSResNet18(num_classes=NUM_CLASSES).to(DEVICE)

loss_fn = nn.CrossEntropyLoss()

# ─── GradMatch strategy ──────────────────────────────────────────────────────

strategy = GradMatchStrategy(
    trainloader    = trainloader,
    valloader      = valloader,
    model          = model,
    loss           = loss_fn,
    eta            = ETA,
    device         = DEVICE,
    num_classes    = NUM_CLASSES,
    linear_layer   = LINEAR_LAYER,
    selection_type = SELECTION_TYPE,   # 'PerBatch' → global OMP
    logger         = logger,
    valid          = VALID,
    v1             = V1,
    lam            = LAM,
    eps            = EPS,
)

# ─── Run selection ───────────────────────────────────────────────────────────

logger.info("Starting GradMatch selection …")

# model_params captures the current weight state at which gradients are computed.
model_params = model.state_dict()

# select() returns:
#   pool_idxs  – list of ints, positions within trainloader's dataset (train_subset)
#   gammas     – OMP weights tensor, one weight per selected point
pool_idxs, gammas = strategy.select(budget=BUDGET, model_params=model_params)

pool_idxs = list(pool_idxs)  # make sure it's a plain Python list

# Map pool-relative indices → original CIFAR-10 training-split indices
cifar10_idxs = [pool_idx[i] for i in pool_idxs]

logger.info(f"Selection complete. Selected {len(pool_idxs)} points.")

# ─── Inspect results ─────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print(f"  GradMatch selected  {len(pool_idxs)}  points  (budget={BUDGET})")
print("=" * 55)
print(f"  Pool-relative idxs  (first 20): {pool_idxs[:20]}")
print(f"  CIFAR-10      idxs  (first 20): {cifar10_idxs[:20]}")
gammas_np = gammas.cpu().numpy() if isinstance(gammas, torch.Tensor) else np.array(gammas)
print(f"  Gammas              (first 20): {gammas_np[:20].round(4)}")
print("=" * 55 + "\n")

# ─── Save ────────────────────────────────────────────────────────────────────

np.save(f"gradmatch_pool_idxs_budget_{BUDGET}.npy",    np.array(pool_idxs))
np.save(f"gradmatch_cifar10_idxs_budget_{BUDGET}.npy", np.array(cifar10_idxs))
np.save(f"gradmatch_gammas_budget_{BUDGET}.npy",        gammas_np)

logger.info(f"Saved: gradmatch_pool_idxs_budget_{BUDGET}.npy | gradmatch_cifar10_idxs_budget_{BUDGET}.npy | gradmatch_gammas__budget_{BUDGET}.npy")

# ─── Optional: build a subset DataLoader for downstream training ─────────────

# Uncomment if you want to immediately train on the selected subset.
#
# selected_subset = Subset(full_dataset, cifar10_idxs)
# selected_loader = DataLoader(
#     selected_subset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=NUM_WORKERS,
# )