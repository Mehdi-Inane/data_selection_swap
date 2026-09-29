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

Checkpoint subsampling
──────────────────────
--num_checkpoints K (default 10) restricts the trajectory to K
checkpoints from --checkpoint_dir. --checkpoint_selection spread
(default) spaces them over the whole run; prefix takes the first K
filenames, which for train_full_data.py's save cadence means epochs
1-10. Pass --num_checkpoints 0 to use every checkpoint found.

Objective weighting (--optimizer)
─────────────────────────────────
The kernel is P_ij = sum_k w_k * g_i^k^T M_k g_j^k, and --optimizer
picks (w_k, M_k) via reference_trajectory.py, which owns the derivations
and is shared with train_kl_selection_full.py:

    uniform      w_k = 1,                  M_k = I
    sgd          w_k = sum eta_s,          M_k = I
    adam         w_k = sum eta_s a_s^2,    M_k = D_k^2   (KL weighting)
    adam_update  w_k = sum eta_s^2 a_s^2,  M_k = D_k^2   (update mismatch)

w_k >= 0 and M_k >= 0 keep P positive semi-definite, so every swap
identity below still holds for the weighted objective. An arbitrary
coordinate-wise M does not survive independent JL projections of L and
Phi, so the adam variants require --proj_dim 0.

Swap rule (--rule)
──────────────────
    ranked  Algorithm 3. Test the top-ranked pair only, stop on the first
            rejection. Cheap, but "stopped" does not mean "optimal".
    best    Algorithm 2. The exact best swap each iteration, found by
            enumerating pairs in increasing h_j - h_i and stopping once
            the next linear term cannot beat the best exact change
            already found. Exact without ever storing an n x n kernel.

Reproducing the pre-existing runs: --optimizer uniform --rule ranked
--checkpoint_selection prefix --no_include_bias --scoring_transform train.
"""

import argparse
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

import d_alpha
import reference_trajectory as ref
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
    X,
    Y,
    batch_size,
    R_L=None,
    R_Phi=None,
    include_bias=True,
):
    """
    Extract Kronecker factors of the last-layer gradient for all samples.

    X and Y are the *materialised* pool, not a dataloader. Under the
    augmenting 'train' transform a dataloader redraws the augmentation on
    every pass, so each checkpoint would see a different view of the same
    example and P would sum inner products between gradients at different
    inputs. Fixing the view once makes P the object it is meant to be and
    makes the run reproducible.

    With include_bias, a constant 1 is appended to every Phi_i. The
    gradient w.r.t. linear.bias is exactly the logit gradient, so
    <g_i, g_j> then picks up the extra <L_i, L_j> * 1 term it was missing
    — the classifier bias was previously dropped from the kernel. The
    matching column of M (the bias entries of D_k^2) is appended by
    reference_trajectory.adam_metric_factored.
    """
    model.eval()
    L_buf, Phi_buf = [], []

    features_dict = {}
    def _hook(_, inp, __):
        features_dict['phi'] = inp[0]

    handle = model.linear.register_forward_hook(_hook)

    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            inputs, targets = X[start:start + batch_size], Y[start:start + batch_size]
            outputs = model(inputs)

            probs         = torch.softmax(outputs, dim=1)
            one_hot       = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)
            grad_logits   = probs - one_hot             # [B, C]
            phi           = features_dict['phi']        # [B, F]

            if R_L is not None:
                grad_logits = grad_logits @ R_L        # [B, k]
            if R_Phi is not None:
                phi = phi @ R_Phi                      # [B, k]
            if include_bias:
                phi = torch.cat([phi, torch.ones_like(phi[:, :1])], dim=1)

            L_buf.append(grad_logits.cpu())
            Phi_buf.append(phi.cpu())

    handle.remove()
    return torch.cat(L_buf, dim=0), torch.cat(Phi_buf, dim=0)


def build_weighted_trajectory(model, X, Y, batch_size, checkpoints, device, use_metric,
                              num_classes, feat_dim, R_L=None, R_Phi=None,
                              include_bias=True, adam_eps=1e-8, logger=None):
    """
    [(L_k, Phi_k, M_k, w_k)], all resident on `device`.

    Keeping the factors on the GPU for the whole of selection is what makes
    the swap loop cheap: a kernel column is then O(n(C+F)) arithmetic with
    no host transfer, against O(nCF) plus a 1.1 GB copy for a full matvec.

    M_k is None for the Euclidean objectives. For adam it is D_k^2 shaped
    [C, F(+1)], rescaled by one factor shared across all checkpoints:
    (sqrt(vhat)+eps)^-2 spans many orders of magnitude, and a single global
    scale tames it without touching the relative weighting between
    checkpoints, so neither the argmin nor any swap decision changes.
    """
    factors, metrics = [], []
    for c in checkpoints:
        model.load_state_dict(ref.load_weights(c.path))
        L, Phi = extract_factored_features(model, X, Y, batch_size, R_L=R_L,
                                           R_Phi=R_Phi, include_bias=include_bias)
        factors.append((L.to(device), Phi.to(device)))
        metrics.append(
            ref.adam_metric_factored(c, num_classes, feat_dim, eps=adam_eps, power=2,
                                     include_bias=include_bias)
            if use_metric else None)

    if use_metric:
        scale = max(float(M.mean()) for M in metrics)
        metrics = [(M / scale).float().to(device) for M in metrics]
        if logger:
            logger.info(f"Metric M_k = D_k^2 rescaled by 1/{scale:.4g} "
                        f"(global, objective is in those units)")

    return [(L, Phi, M, c.weight)
            for (L, Phi), M, c in zip(factors, metrics, checkpoints)]


# ─────────────────────────────────────────────────────────────────────────────
# Weighted factored kernel–vector product
#   (P a)_i = sum_k w_k * sum_{c,f} M_k[c,f] L_k[i,c] Phi_k[i,f] Z_k[c,f]
#   with Z_k = L_k^T diag(a) Phi_k, i.e. two [n,C]x[C,F] contractions per
#   checkpoint and never an n x n or n x (C*F) object.
# ─────────────────────────────────────────────────────────────────────────────

def weighted_kernel_mv(traj, a):
    out = torch.zeros_like(a)
    for L, Phi, M, w in traj:
        Z = L.T @ (Phi * a[:, None])            # [C, F]
        if M is not None:
            Z = M * Z
        out += w * ((L @ Z) * Phi).sum(dim=1)
    return out


def compute_h(traj, alpha, m, n, c_sum):
    """h = grad f = (2/m^2) P alpha - (2/(mn)) P 1."""
    return (2.0 / m ** 2) * weighted_kernel_mv(traj, alpha) - (2.0 / (m * n)) * c_sum


def objective(h, c_sum, alpha, m, n):
    """f(alpha) = (1/m^2) a'Pa + q'a, rewritten as (1/2) h'a - (1/(mn)) c'a."""
    mask = alpha == 1.0
    return float((0.5 * h[mask].sum() - (1.0 / (m * n)) * c_sum[mask].sum()).item())


def kernel_diagonal(traj):
    """d_i = P_ii = sum_k w_k sum_{c,f} M[c,f] L[i,c]^2 Phi[i,f]^2."""
    d = torch.zeros(traj[0][0].shape[0], device=traj[0][0].device)
    for L, Phi, M, w in traj:
        if M is None:
            d += w * (L * L).sum(dim=1) * (Phi * Phi).sum(dim=1)
        else:
            d += w * (((L * L) @ M) * (Phi * Phi)).sum(dim=1)
    return d


def kernel_pairs(traj, i_idx, j_idx):
    """
    P_ij for a batch of pairs, without touching the rest of the kernel.

    Without M the Kronecker structure separates completely,
    P_ij = <L_i,L_j><Phi_i,Phi_j>, at O(C+F) per pair; with M it is
    (L_i*L_j)' M (Phi_i*Phi_j) at O(CF). Either way nothing n x n is built,
    which is what makes the exact best-swap screening affordable.
    """
    out = torch.zeros(i_idx.numel(), device=traj[0][0].device)
    for L, Phi, M, w in traj:
        ll = L[i_idx] * L[j_idx]          # [B, C]
        pp = Phi[i_idx] * Phi[j_idx]      # [B, F]
        if M is None:
            out += w * ll.sum(dim=1) * pp.sum(dim=1)
        else:
            out += w * ((ll @ M) * pp).sum(dim=1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 1: Gradient-Ranked Single-Swap Descent
# ─────────────────────────────────────────────────────────────────────────────

def gradient_ranked_single_swap(traj, m, max_iters=5000, log_freq=5, device='cuda',
                                 seed=None):
    """
    Algorithm 3 of the manuscript, unchanged apart from the kernel it calls.

    Rank exchanges by the gradient h, test only the best-ranked candidate,
    and accept it only if its *exact* change in the objective is negative.
    The exact change of a one-for-one exchange delta = e_j - e_i is
    (1/2) delta'(h + h_tilde) for any quadratic, which is why one extra
    kernel-vector product suffices and the pair's own P_ij is never needed.
    """
    n = traj[0][0].shape[0]
    c_sum = weighted_kernel_mv(traj, torch.ones(n, device=device))

    g = torch.Generator(device='cpu')
    if seed is not None:
        g.manual_seed(seed)
    perm = torch.randperm(n, generator=g).to(device)
    alpha = torch.zeros(n, device=device)
    alpha[perm[:m]] = 1.0
    h = compute_h(traj, alpha, m, n, c_sum)

    objective_history = []
    t, gamma_t, delta_t = 0, float('nan'), None
    reason = 'max_iters'

    for t in range(max_iters):

        if t % log_freq == 0:
            objective_history.append({"iteration": t,
                                      "objective": objective(h, c_sum, alpha, m, n)})

        # Candidate: the selected index with the largest h, the unselected
        # index with the smallest h. This pair minimises the linearised
        # change h_j - h_i.
        h_in = h.clone(); h_in[alpha == 0] = -float('inf')
        i_t = int(torch.argmax(h_in).item())
        h_out = h.clone(); h_out[alpha == 1] = float('inf')
        j_t = int(torch.argmin(h_out).item())

        gamma_t = h[i_t].item() - h[j_t].item()
        if gamma_t <= 0:
            reason = 'gamma<=0'
            break

        alpha_tilde = alpha.clone()
        alpha_tilde[i_t] = 0.0
        alpha_tilde[j_t] = 1.0
        h_tilde = compute_h(traj, alpha_tilde, m, n, c_sum)

        delta_t = 0.5 * (h_tilde[j_t] - h_tilde[i_t] + h[j_t] - h[i_t]).item()
        if delta_t < 0:
            alpha, h = alpha_tilde, h_tilde
        else:
            # Algorithm 3 stops here, but only the *top-ranked* pair was
            # tested: a lower-ranked pair may still improve f. Use
            # --rule best for the exact answer.
            reason = 'top-ranked candidate rejected (Algorithm 3 stopping rule)'
            break

    objective_history.append({"iteration": t,
                              "objective": objective(h, c_sum, alpha, m, n)})
    diag = {"rule": "ranked", "iterations": t, "swaps": t, "reason": reason,
            "final_gamma": gamma_t, "final_change": delta_t,
            "objective": objective_history[-1]["objective"]}
    return torch.where(alpha == 1.0)[0].cpu(), objective_history, diag


def best_swap(traj, m, max_iters=5000, log_freq=5, device='cuda', seed=None,
              max_pairs=0, batch_pairs=8192, logger=None):
    """
    Algorithm 2: the exact best single swap, found without storing P.

    Algorithm 3 tests only its top-ranked pair, so rejecting that pair
    does not mean no improving swap exists. The exact change is

        Delta_ij = h_j - h_i + (1/m^2) sum_k w_k ||g_j^k - g_i^k||^2_{M_k}
                 = h_j - h_i + (1/m^2) (d_i + d_j - 2 P_ij),

    a linear term plus a *non-negative* curvature term. So enumerate
    pairs in increasing h_j - h_i (selection_common.ranked_pairs), keep
    the best exact Delta seen, and stop as soon as the next unexamined
    linear term is >= that best: no pair beyond it can beat it, because
    curvature only adds. What remains is an exact search, and when it
    reports no improving swap, there genuinely is none.

    The screening is exact but not cheap in the worst case — m(n-m) is
    425 million pairs at the CIFAR-100 default — so --max_pairs caps the
    work per iteration, at the cost of the certificate.
    """
    n = traj[0][0].shape[0]
    ones = torch.ones(n, device=device)
    c_sum = weighted_kernel_mv(traj, ones)
    d = kernel_diagonal(traj)

    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    perm = torch.randperm(n, generator=g).to(device)
    alpha = torch.zeros(n, device=device)
    alpha[perm[:m]] = 1.0
    h = compute_h(traj, alpha, m, n, c_sum)

    objective_history = []
    reason, t = 'max_iters', 0
    examined_total = 0

    for t in range(max_iters):
        if t % log_freq == 0:
            objective_history.append({"iteration": t,
                                      "objective": objective(h, c_sum, alpha, m, n)})

        sel = torch.where(alpha == 1.0)[0]
        uns = torch.where(alpha == 0.0)[0]
        drop = sel[torch.argsort(h[sel], descending=True)]
        add = uns[torch.argsort(h[uns])]
        pairs = common.ranked_pairs(drop.cpu().numpy(), h[drop].cpu().numpy(),
                                    add.cpu().numpy(), h[add].cpu().numpy())

        best, best_pair, examined, exhausted = 0.0, None, 0, True
        buf_lin, buf_i, buf_j = [], [], []

        def flush():
            nonlocal best, best_pair
            if not buf_i:
                return
            ii = torch.tensor(buf_i, device=device)
            jj = torch.tensor(buf_j, device=device)
            delta = (torch.tensor(buf_lin, device=device)
                     + (d[ii] + d[jj] - 2.0 * kernel_pairs(traj, ii, jj)) / m ** 2)
            k = int(torch.argmin(delta))
            if float(delta[k]) < best:
                best, best_pair = float(delta[k]), (buf_i[k], buf_j[k])
            buf_lin.clear(); buf_i.clear(); buf_j.clear()

        for lin, i, j in pairs:
            # Curvature is >= 0, so every unexamined pair has
            # Delta >= lin >= this lin. Once that cannot beat `best`,
            # the search is provably done.
            if lin >= best:
                flush()
                if lin >= best:
                    break
            buf_lin.append(lin); buf_i.append(i); buf_j.append(j)
            examined += 1
            if len(buf_i) >= batch_pairs:
                flush()
            if max_pairs and examined >= max_pairs:
                flush()
                exhausted = False
                break
        flush()
        examined_total += examined

        if best_pair is None:
            reason = ('no improving swap exists (exact)' if exhausted
                      else f'no improving swap in the {examined} best-ranked pairs '
                           f'(--max_pairs, not certified)')
            break

        i_t, j_t = best_pair
        alpha[i_t] = 0.0
        alpha[j_t] = 1.0
        h = compute_h(traj, alpha, m, n, c_sum)
        if logger and (t % log_freq == 0):
            logger.info(f"iter {t}: best exact change {best:.6e} "
                        f"({examined} pairs screened of {len(sel) * len(uns)})")

    objective_history.append({"iteration": t,
                              "objective": objective(h, c_sum, alpha, m, n)})
    diag = {"rule": "best", "iterations": t, "swaps": t, "reason": reason,
            "pairs_screened": examined_total,
            "objective": objective_history[-1]["objective"]}
    return torch.where(alpha == 1.0)[0].cpu(), objective_history, diag


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
    p.add_argument('--num_checkpoints', default=10,         type=int,
                   help='How many checkpoints from --checkpoint_dir feed the kernel. '
                        '0 = every checkpoint found.')
    p.add_argument('--checkpoint_selection', default='spread', choices=['spread', 'prefix'],
                   help="'spread' spaces the K checkpoints over the whole run; 'prefix' "
                        "takes the first K filenames (epochs 1-10 for train_full_data.py), "
                        "which is the previous behaviour.")
    p.add_argument('--checkpoint_epochs', default=None, type=int, nargs='+',
                   help='Explicit checkpoint epochs; overrides the two options above.')
    p.add_argument('--optimizer',      default='uniform',
                   choices=list(ref.OPTIMIZERS),
                   help='Objective weighting (w_k, M_k). See the module docstring. '
                        'NOTE the default optimises neither the learning-rate weighting '
                        'nor the Adam preconditioner; it exists to reproduce old runs.')
    p.add_argument('--metric',         default='auto', choices=list(ref.METRICS),
                   help='Override M_k independently of the step weight. Needed for a '
                        'short prefix trajectory, where eta is flat and --optimizer '
                        'uniform --metric adam is the only way to get equal checkpoint '
                        'weights together with the Adam preconditioner.')
    p.add_argument('--momentum_response', default='persistent',
                   choices=list(ref.MOMENTUM_RESPONSES),
                   help="a_s in w_k. 'persistent' (a_s = 1) is the exact response to a "
                        'discrepancy that lasts, which is what a subset produces; '
                        "'isolated' is (1-beta1)/(1-beta1^s).")
    p.add_argument('--rule',           default='ranked', choices=['ranked', 'best'],
                   help="'ranked' is Algorithm 3: test the top-ranked pair, stop on the "
                        "first rejection. 'best' is Algorithm 2: the exact best swap "
                        'each iteration, via heap screening on the linear term, so '
                        '"no improving swap" actually means there is none.')
    p.add_argument('--max_pairs',      default=0,           type=int,
                   help='--rule best: cap on pairs screened per iteration (0 = no cap, '
                        'exact). The cap bounds the worst case, m(n-m) = 425M pairs at '
                        'the CIFAR-100 default, but forfeits the certificate.')
    p.add_argument('--batch_pairs',    default=8192,        type=int,
                   help='--rule best: pairs evaluated per batched kernel call.')
    p.add_argument('--ref_lr',         default=0.01,        type=float,
                   help="Base LR of the *reference* trajectory, used to rebuild eta_s for "
                        "--optimizer sgd on a train_full_data.py directory. Ignored when "
                        "the directory has a trajectory.json manifest.")
    p.add_argument('--adam_eps',       default=1e-8,        type=float,
                   help='eps inside D_k = (sqrt(vhat_k) + eps)^-1.')
    p.add_argument('--beta1',          default=0.9,         type=float)
    p.add_argument('--beta2',          default=0.999,       type=float)
    p.add_argument('--include_bias',   action='store_true', default=True,
                   help='Include linear.bias in the kernel (default on).')
    p.add_argument('--no_include_bias', dest='include_bias', action='store_false',
                   help='Drop linear.bias, reproducing the previous kernel.')
    p.add_argument('--max_iters',      default=5000,        type=int,
                   help='Maximum iterations for single-swap descent')
    p.add_argument('--batch_size',     default=128,         type=int,
                   help='Batch size for dataloaders')
    p.add_argument('--lr',             default=0.01,        type=float,
                   help='Learning rate for retraining')
    p.add_argument('--retrain_optimizer', default=None, choices=['sgd', 'adam', 'adamw'],
                   help='Optimizer for the retrain-on-subset stage. Default: follow '
                        '--optimizer, so the subset is not scored against one optimizer '
                        'and retrained with another.')
    p.add_argument('--init_ckpt',      default=None,        type=str,
                   help='Shared retraining initialisation. Created on first use, then '
                        'reused, so two selectors are compared from the same weights.')
    p.add_argument('--num_workers',    default=4,           type=int,
                   help='Data loader workers')
    p.add_argument('--download',       action='store_true', default=False,
                   help='Download dataset if not present')
    p.add_argument('--scoring_transform', default='test', choices=['train', 'test'],
                   help="View the pool is scored through. Default 'test' (clean) so that "
                        "this and train_kl_selection_full.py score identical inputs; "
                        "'train' reproduces the old runs. Either way the view is "
                        'materialised once and shared by every checkpoint.')
    p.add_argument('--selection_only', action='store_true', default=False,
                   help='Stop after saving indices + timing (skip retraining)')
    args = p.parse_args()

    set_seed(args.seed)

    use_metric = ref.resolve_metric(args.optimizer, args.metric)
    if use_metric and args.proj_dim > 0:
        raise ValueError(
            f"M_k = D_k^2 needs --proj_dim 0. The sparse-JL step projects L and Phi "
            f"independently, which preserves <L_i,L_j><Phi_i,Phi_j> in expectation but "
            f"not the coordinate-wise metric "
            f"sum_{{c,f}} M[c,f] L[i,c]L[j,c]Phi[i,f]Phi[j,f]."
        )
    if args.retrain_optimizer is None:
        args.retrain_optimizer = 'adam' if use_metric else 'sgd'

    # ── 0. Dataset ───────────────────────────────────────────────────────────
    cfg         = get_dataset_config(args.dataset, args.data_dir, download=args.download)
    NUM_CLASSES = cfg['num_classes']
    NUM_EPOCHS  = cfg['num_epochs']
    FEAT_DIM    = 512           # ResNet18 penultimate layer width

    n_val   = int(0.1 * len(cfg['full_train']))
    n_train = len(cfg['full_train']) - n_val
    budget  = int(args.fraction * n_train)

    # ── 1. Resolve checkpoint trajectory ─────────────────────────────────────
    #      Before any file naming below, so the saved names can record how
    #      many checkpoints were actually used, not just what was requested.
    if not os.path.isdir(args.checkpoint_dir):
        raise ValueError(f"Checkpoint directory not found: {args.checkpoint_dir}")
    checkpoints, manifest, ckpt_description = ref.resolve(
        args.checkpoint_dir, optimizer=args.optimizer,
        num_checkpoints=args.num_checkpoints, selection=args.checkpoint_selection,
        epochs=args.checkpoint_epochs, n_train=n_train, batch_size=args.batch_size,
        base_lr=args.ref_lr, num_epochs=NUM_EPOCHS, beta1=args.beta1,
        momentum=args.momentum_response, metric=args.metric)
    num_ckpts_used = len(checkpoints)

    scratch_path = os.environ.get("SCRATCH", "/home/mila/a/ahmedm/scratch")
    base_dir     = f"{scratch_path}/gradmatch_swap/{args.dataset}/{budget}/seed_{args.seed}"
    os.makedirs(base_dir, exist_ok=True)

    # ── 2. Logging ───────────────────────────────────────────────────────────
    # file_prefix carries the checkpoint count (K{num_ckpts_used}) so that
    # sweeps over --num_checkpoints for the same (dataset, budget, seed) land
    # in the same base_dir without ever overwriting each other's files. The
    # objective tag keeps the three --optimizer variants apart too.
    obj_tag = '' if args.optimizer == 'uniform' else f"_{args.optimizer}"
    file_prefix = f"kl_faithful{obj_tag}_{budget}_K{num_ckpts_used}"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(base_dir, f"{file_prefix}_training.log")),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Dataset={args.dataset}  budget={budget}  seed={args.seed}  "
                f"device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info(f"Trajectory: {ckpt_description}"
                + ("" if manifest else "  (legacy dir, eta_s reconstructed from the "
                                        "cosine schedule)"))
    logger.info("\n" + ref.describe(checkpoints))
    logger.info(f"rule={args.rule}  metric={'D_k^2' if use_metric else 'I'}  "
                f"include_bias={args.include_bias}  "
                f"scoring_transform={args.scoring_transform}  "
                f"retrain_optimizer={args.retrain_optimizer}  init_ckpt={args.init_ckpt}")
    if args.optimizer == 'uniform' and not use_metric:
        logger.warning("--optimizer uniform --metric none: w_k = 1 and M_k = I, so this "
                       "run uses neither the learning-rate weighting nor the Adam "
                       "preconditioner. Pass --optimizer sgd or --metric adam.")
    last_epoch = checkpoints[-1].epoch
    if last_epoch < 0.25 * NUM_EPOCHS:
        logger.warning(
            f"Trajectory stops at epoch {last_epoch} of {NUM_EPOCHS}. The objective "
            f"describes only that prefix and says nothing about the rest of training; "
            f"the subset is chosen to match early-training gradients and is being "
            f"retrained for the full {NUM_EPOCHS}.")

    BATCH_SIZE = args.batch_size
    LR         = args.lr
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'

    MODEL_SAVE_PATH   = os.path.join(base_dir, f"{file_prefix}_model.pth")
    INDICES_SAVE_PATH = os.path.join(base_dir, f"{file_prefix}_indices.pt")
    METRICS_SAVE_PATH = os.path.join(base_dir, f"{file_prefix}_metrics.json")
    OBJ_SAVE_PATH     = os.path.join(base_dir, f"{file_prefix}_objective.json")
    TIMING_SAVE_PATH  = os.path.join(base_dir, f"{file_prefix}_timing.json")

    # ── 3. Scoring pool ──────────────────────────────────────────────────────
    #      Materialised once: a dataloader under the 'train' transform
    #      redraws the augmentation on every pass, so each checkpoint would
    #      see a different view of the same example and P would mix
    #      gradients taken at different inputs.
    trainset, _ = random_split(
        cfg['full_train'], [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    X, Y = d_alpha.materialise(common.scoring_subset(cfg, trainset, args.scoring_transform),
                               device, batch_size=BATCH_SIZE, num_workers=args.num_workers)
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
    timer = common.SelectionTimer(device)
    with timer.phase('scoring'):
        traj = build_weighted_trajectory(
            ref_model, X, Y, BATCH_SIZE, checkpoints, device, use_metric,
            NUM_CLASSES, FEAT_DIM, R_L=R_L, R_Phi=R_Phi,
            include_bias=args.include_bias, adam_eps=args.adam_eps, logger=logger,
        )

    # ── 7. Swap descent ──────────────────────────────────────────────────────
    logger.info(f"Running --rule {args.rule} swap descent "
                f"(budget={budget}/{n_train}) ...")
    with timer.phase('selection'):
        if args.rule == 'ranked':
            selected_indices, objective_history, diag = gradient_ranked_single_swap(
                traj, budget, max_iters=args.max_iters, log_freq=5, device=device,
                seed=args.seed,
            )
        else:
            selected_indices, objective_history, diag = best_swap(
                traj, budget, max_iters=args.max_iters, log_freq=5, device=device,
                seed=args.seed, max_pairs=args.max_pairs,
                batch_pairs=args.batch_pairs, logger=logger,
            )
    logger.info(f"Stopped after {diag['iterations']} iterations ({diag['reason']}), "
                f"{len(selected_indices)} samples selected. "
                f"f: {objective_history[0]['objective']:.6e} -> "
                f"{objective_history[-1]['objective']:.6e}")
    if diag['reason'] == 'max_iters':
        logger.warning("--max_iters reached, so the search had not settled; the subset "
                       "is the best found, not a local optimum.")

    common.save_timing(TIMING_SAVE_PATH, timer, logger, method='kl_faithful',
                       reference_epochs_used=[c.epoch for c in checkpoints],
                       swap_iterations=diag['iterations'],
                       objective_optimizer=args.optimizer,
                       objective_metric='D_k^2' if use_metric else 'I',
                       momentum_response=args.momentum_response,
                       scoring_transform=args.scoring_transform,
                       include_bias=args.include_bias,
                       checkpoint_selection=args.checkpoint_selection,
                       checkpoint_steps=[c.step for c in checkpoints],
                       checkpoint_weights=[c.weight for c in checkpoints],
                       num_checkpoints_used=num_ckpts_used,
                       **{k_: v for k_, v in diag.items() if k_ != 'iterations'})
    torch.save(selected_indices, INDICES_SAVE_PATH)
    with open(OBJ_SAVE_PATH, 'w') as fh:
        json.dump(objective_history, fh, indent=4)
    if args.selection_only:
        logger.info("--selection_only set: skipping retraining.")
        return

    # ── 8. Retrain on selected subset ────────────────────────────────────────
    del traj, X, Y
    if device == 'cuda':
        torch.cuda.empty_cache()

    trainloader = DataLoader(
        Subset(trainset, selected_indices),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=args.num_workers,
    )

    model = ResNet18(num_classes=NUM_CLASSES)
    if cfg['cifar_style']:
        model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model = model.to(device)
    logger.info(common.load_or_create_init(model, args.init_ckpt, logger))

    criterion = nn.CrossEntropyLoss()
    optimizer = common.build_optimizer(model.parameters(), args.retrain_optimizer, LR,
                                       betas=(args.beta1, args.beta2), eps=args.adam_eps)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    logger.info(f"Retraining on KL-Faithful subset with {args.retrain_optimizer} ...")
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