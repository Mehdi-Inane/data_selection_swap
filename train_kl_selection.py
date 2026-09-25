"""
train_kl_selection.py
KL-Faithful subset selection via Gradient-Ranked Single-Swap Descent (CIFAR-100 & ImageNet).

Memory reduction strategy
──────────────────────────
The last-layer gradient for sample i is:

    g_i  =  vec( grad_logit_i  φ_i^T )   (outer product, flattened)

so the kernel used by the algorithm factorises exactly as

    K = G G^T  =  (L L^T) ⊙ (Φ Φ^T)          (Hadamard product)

where  L ∈ R^{n×C}  are logit-gradient vectors and  Φ ∈ R^{n×F}  are
penultimate-layer features.  We store (L, Φ) instead of G and rewrite
compute_h without ever materialising G:

    Dataset     G / ckpt   (L,Φ) / ckpt   reduction
    CIFAR-100     9.2 GB       110 MB         83×   (exact)
    ImageNet     ~2.4 TB       7.0 GB        340×   (exact)

For ImageNet with many checkpoints, an optional sparse-Rademacher JL
(Achlioptas 2003) can be applied independently to L and Φ, reducing
each to k dims (e.g. k=64 → 600 MB / ckpt for ImageNet, approximate).
"""

import argparse
import glob
import json
import logging
import math
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

import selection_common as common


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
# Dataset configuration
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
# Sparse-Rademacher JL projection matrix (Achlioptas 2003)
# ─────────────────────────────────────────────────────────────────────────────

def build_sparse_jl(dim_in: int, k: int, seed: int) -> torch.Tensor:
    """
    Build sparse Rademacher JL matrix  R ∈ R^{dim_in × k}.

    Each entry is  ±sqrt(3/k)  with prob 1/6 each, 0 with prob 2/3.
    Guarantees  E[R_ij²] = 1/k  =>  E[||Rx||²] = ||x||²  (unbiased norm).
    """
    rng   = torch.Generator().manual_seed(seed)
    scale = math.sqrt(3.0 / k)
    nonzero = torch.bernoulli(torch.full((dim_in, k), 1.0 / 3.0), generator=rng).bool()
    signs   = (torch.randint(0, 2, (dim_in, k), generator=rng) * 2 - 1).float()
    R = torch.zeros(dim_in, k)
    R[nonzero] = signs[nonzero] * scale
    return R


# ─────────────────────────────────────────────────────────────────────────────
# Factored feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_factored_features(
    model,
    dataloader,
    device,
    R_L=None,
    R_Phi=None,
):
    """Extract Kronecker factors of the last-layer gradient for all samples."""
    model.eval()
    L_buf, Phi_buf = [], []

    features_dict = {}
    def _hook(_, inp, __):
        features_dict['phi'] = inp[0]

    handle = model.linear.register_forward_hook(_hook)

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            probs         = torch.softmax(outputs, dim=1)
            one_hot       = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)
            grad_logits   = probs - one_hot             # [B, C]
            phi           = features_dict['phi']        # [B, F]

            if R_L is not None:
                grad_logits = grad_logits @ R_L        # [B, k]
            if R_Phi is not None:
                phi = phi @ R_Phi                      # [B, k]

            L_buf.append(grad_logits.cpu())
            Phi_buf.append(phi.cpu())

    handle.remove()
    return torch.cat(L_buf, dim=0), torch.cat(Phi_buf, dim=0)


def load_trajectory_factored_features(model, dataloader, checkpoint_paths, device,
                                       R_L=None, R_Phi=None):
    """Load T checkpoints in order; extract (L_t, Φ_t) for each."""
    LP_list = []
    for ckpt in checkpoint_paths:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        L, Phi = extract_factored_features(model, dataloader, device, R_L=R_L, R_Phi=R_Phi)
        LP_list.append((L, Phi))
    return LP_list


# ─────────────────────────────────────────────────────────────────────────────
# Factored kernel–vector product
# ─────────────────────────────────────────────────────────────────────────────

def _factored_Kv(LP_list, v, device):
    n   = LP_list[0][0].shape[0]
    vv  = v.squeeze()
    p   = torch.zeros(n, 1, device=device)
    for L, Phi in LP_list:
        L   = L.to(device)
        Phi = Phi.to(device)
        Z   = (L * vv.unsqueeze(1)).T @ Phi
        LZ  = L @ Z
        p  += (LZ * Phi).sum(dim=1, keepdim=True)
    return p


def _build_c_sum(LP_list, n, device):
    c = torch.zeros(n, 1, device=device)
    for L, Phi in LP_list:
        L   = L.to(device)
        Phi = Phi.to(device)
        Z   = L.T @ Phi
        LZ  = L @ Z
        c  += (LZ * Phi).sum(dim=1, keepdim=True)
    return c


def compute_h(LP_list, alpha_vec, m, n, c_sum, device):
    p0_alpha = _factored_Kv(LP_list, alpha_vec, device)
    return (2.0 / m ** 2) * p0_alpha - (2.0 / (m * n)) * c_sum


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 1: Gradient-Ranked Single-Swap Descent
# ─────────────────────────────────────────────────────────────────────────────

def gradient_ranked_single_swap(LP_list, m, max_iters=5000, log_freq=5, device='cuda'):
    n     = LP_list[0][0].shape[0]
    c_sum = _build_c_sum(LP_list, n, device)

    perm  = torch.randperm(n, device=device)
    alpha = torch.zeros(n, 1, device=device)
    alpha[perm[:m]] = 1.0
    h     = compute_h(LP_list, alpha, m, n, c_sum, device)

    objective_history = []

    for t in range(max_iters):

        if t % log_freq == 0:
            mask  = alpha.squeeze() == 1.0
            term1 = 0.5 * h[mask].sum()
            term2 = (1.0 / (m * n)) * c_sum[mask].sum()
            objective_history.append({"iteration": t, "objective": (term1 - term2).item()})

        # Candidate swap
        h_in          = h.clone(); h_in[alpha == 0]  = -float('inf')
        i_t           = torch.argmax(h_in).item()
        h_out         = h.clone(); h_out[alpha == 1] =  float('inf')
        j_t           = torch.argmin(h_out).item()

        if h[i_t].item() - h[j_t].item() <= 0:
            break

        alpha_tilde         = alpha.clone()
        alpha_tilde[i_t]    = 0.0
        alpha_tilde[j_t]    = 1.0
        h_tilde             = compute_h(LP_list, alpha_tilde, m, n, c_sum, device)

        delta_t = 0.5 * (h_tilde[j_t] - h_tilde[i_t] + h[j_t] - h[i_t]).item()
        if delta_t < 0:
            alpha, h = alpha_tilde, h_tilde
        else:
            break

    mask  = alpha.squeeze() == 1.0
    term1 = 0.5 * h[mask].sum()
    term2 = (1.0 / (m * n)) * c_sum[mask].sum()
    objective_history.append({"iteration": t, "objective": (term1 - term2).item()})

    print(f"Total iterations: {t}")
    return torch.where(alpha.squeeze() == 1.0)[0].cpu(), objective_history, t


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fraction',       default=0.3,         type=float)
    p.add_argument('--seed', '-seed',   default=42,          type=int)
    p.add_argument('--checkpoint_dir', required=True,       type=str)
    p.add_argument('--dataset',        default='cifar100',  choices=['cifar100', 'imagenet'])
    p.add_argument('--data_dir',       required=True,       type=str,
                   help='Root of the staged dataset (e.g. $SLURM_TMPDIR/cifar100_data)')
    p.add_argument('--proj_dim',       default=0,           type=int,
                   help='Sparse-JL target dim per factor (0 = exact, no JL)')
    p.add_argument('--max_iters',      default=5000,        type=int,
                   help='Maximum iterations for single-swap descent')
    p.add_argument('--batch_size',     default=128,         type=int,
                   help='Batch size for dataloaders')
    p.add_argument('--lr',             default=0.01,        type=float,
                   help='Learning rate for retraining')
    p.add_argument('--num_workers',    default=4,           type=int,
                   help='Data loader workers')
    p.add_argument('--download',       action='store_true', default=False,
                   help='Download dataset if not present')
    p.add_argument('--scoring_transform', default='train', choices=['train', 'test'],
                   help="Transform used when extracting features ('train' = previous behaviour)")
    p.add_argument('--selection_only', action='store_true', default=False,
                   help='Stop after saving indices + timing (skip retraining)')
    args = p.parse_args()

    set_seed(args.seed)

    # ── 1. Dataset ───────────────────────────────────────────────────────────
    cfg         = get_dataset_config(args.dataset, args.data_dir, download=args.download)
    NUM_CLASSES = cfg['num_classes']
    NUM_EPOCHS  = cfg['num_epochs']
    FEAT_DIM    = 512           # ResNet18 penultimate layer width

    n_val   = int(0.1 * len(cfg['full_train']))
    n_train = len(cfg['full_train']) - n_val
    budget  = int(args.fraction * n_train)

    scratch_path = os.environ.get("SCRATCH", "/home/mila/a/ahmedm/scratch")
    base_dir     = f"{scratch_path}/gradmatch_swap/{args.dataset}/{budget}/seed_{args.seed}"
    os.makedirs(base_dir, exist_ok=True)

    # ── 2. Logging ───────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(base_dir, f"kl_faithful_{budget}_training.log")),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Dataset={args.dataset}  budget={budget}  seed={args.seed}  "
                f"device={'cuda' if torch.cuda.is_available() else 'cpu'}")

    BATCH_SIZE = args.batch_size
    LR         = args.lr
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'

    MODEL_SAVE_PATH   = os.path.join(base_dir, f"kl_faithful_{budget}_model.pth")
    INDICES_SAVE_PATH = os.path.join(base_dir, f"kl_faithful_{budget}_indices.pt")
    METRICS_SAVE_PATH = os.path.join(base_dir, f"kl_faithful_{budget}_metrics.json")
    OBJ_SAVE_PATH     = os.path.join(base_dir, f"kl_faithful_{budget}_objective.json")
    TIMING_SAVE_PATH  = os.path.join(base_dir, f"kl_faithful_{budget}_timing.json")

    # ── 3. Data loaders ──────────────────────────────────────────────────────
    trainset, _ = random_split(
        cfg['full_train'], [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    eval_loader = DataLoader(common.scoring_subset(cfg, trainset, args.scoring_transform),
                             batch_size=BATCH_SIZE, shuffle=False,
                             pin_memory=True, num_workers=args.num_workers)
    testloader  = DataLoader(cfg['testset'], batch_size=BATCH_SIZE, shuffle=False,
                             pin_memory=True, num_workers=args.num_workers)

    # ── 4. Reference model ───────────────────────────────────────────────────
    ref_model = ResNet18(num_classes=NUM_CLASSES)
    if cfg['cifar_style']:
        ref_model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        ref_model.maxpool = nn.Identity()
    ref_model = ref_model.to(device)

    # ── 5. JL matrices ───────────────────────────────────────────────────────
    k = args.proj_dim
    if k > 0:
        mem_jl  = n_train * 2 * k * 4 / 1e9
        mem_fac = n_train * (NUM_CLASSES + FEAT_DIM) * 4 / 1e9
        mem_g   = n_train * NUM_CLASSES * FEAT_DIM * 4 / 1e9
        logger.info(
            f"Sparse-Rademacher JL: C={NUM_CLASSES}→{k}, F={FEAT_DIM}→{k}. "
            f"Per-ckpt RAM: G={mem_g:.1f} GB → factored={mem_fac:.1f} GB → JL={mem_jl:.2f} GB"
        )
        R_L   = build_sparse_jl(NUM_CLASSES, k, seed=args.seed    ).to(device)
        R_Phi = build_sparse_jl(FEAT_DIM,    k, seed=args.seed + 1).to(device)
    else:
        mem_fac = n_train * (NUM_CLASSES + FEAT_DIM) * 4 / 1e9
        mem_g   = n_train * NUM_CLASSES * FEAT_DIM * 4 / 1e9
        logger.info(
            f"Exact Kronecker-factored kernel. "
            f"Per-ckpt RAM: G={mem_g:.1f} GB → (L,Φ)={mem_fac:.2f} GB"
        )
        R_L = R_Phi = None

    # ── 6. Trajectory extraction ─────────────────────────────────────────────
    if not os.path.isdir(args.checkpoint_dir):
        raise ValueError(f"Checkpoint directory not found: {args.checkpoint_dir}")
    ckpt_paths = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.pth")))
    if not ckpt_paths:
        raise ValueError(f"No .pth checkpoints in {args.checkpoint_dir}")

    timer = common.SelectionTimer(device)
    logger.info(f"Found {len(ckpt_paths)} checkpoints — extracting factored features ...")
    with timer.phase('scoring'):
        LP_list = load_trajectory_factored_features(
            ref_model, eval_loader, ckpt_paths, device, R_L=R_L, R_Phi=R_Phi,
        )

    # ── 7. Algorithm 1 ───────────────────────────────────────────────────────
    logger.info(f"Running single-swap descent  (budget={budget}/{n_train}) ...")
    with timer.phase('selection'):
        selected_indices, objective_history, total_iters = gradient_ranked_single_swap(
            LP_list, budget, max_iters=args.max_iters, log_freq=5, device=device,
        )
    logger.info(f"Converged in {total_iters} iterations, "
                f"{len(selected_indices)} samples selected.")

    common.save_timing(TIMING_SAVE_PATH, timer, logger, method='kl_faithful',
                       reference_epochs_used=common.reference_epochs_used(ckpt_paths),
                       swap_iterations=total_iters)
    torch.save(selected_indices, INDICES_SAVE_PATH)
    with open(OBJ_SAVE_PATH, 'w') as fh:
        json.dump(objective_history, fh, indent=4)
    if args.selection_only:
        logger.info("--selection_only set: skipping retraining.")
        return

    # ── 8. Retrain on selected subset ────────────────────────────────────────
    trainloader = DataLoader(
        Subset(trainset, selected_indices),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=args.num_workers,
    )

    model = ResNet18(num_classes=NUM_CLASSES)
    if cfg['cifar_style']:
        model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    logger.info("Retraining on KL-Faithful subset ...")
    accuracy_history = {}

    for epoch in range(1, NUM_EPOCHS + 1):
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

        if epoch % 50 == 0 or epoch == NUM_EPOCHS:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in testloader:
                    x, y     = x.to(device), y.to(device)
                    correct  += model(x).argmax(1).eq(y).sum().item()
                    total    += y.size(0)
            acc = 100.0 * correct / total
            accuracy_history[epoch] = acc
            logger.info(f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
                        f"Test Acc: {acc:.2f}% | "
                        f"Train Loss: {running_loss / len(trainloader):.4f}")

    # ── 9. Save ───────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    with open(METRICS_SAVE_PATH, 'w') as fh:
        json.dump(accuracy_history, fh, indent=4)
    logger.info(f"Saved all artefacts to {base_dir}")


if __name__ == "__main__":
    main()