"""KL-Faithful selection via gradient-ranked single-swap descent.

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
filenames, which for the full-data trainer's save cadence means epochs
1-10. Pass --num_checkpoints 0 to use every checkpoint found.

Objective weighting (--optimizer)
─────────────────────────────────
The kernel is P_ij = sum_k w_k * g_i^k^T M_k g_j^k, and --optimizer
picks (w_k, M_k) via ``trajectory``, which owns the derivations
and is shared with ``selection.kl_full``:

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
import logging
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split

from .. import common
from .. import objective as d_alpha
from .. import trajectory as ref


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    """Backward-compatible alias for :func:`data_selection_swap.common.set_seed`."""
    common.set_seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset configuration
# ─────────────────────────────────────────────────────────────────────────────

def get_dataset_config(dataset: str, data_dir: str, download: bool = False) -> dict:
    """Backward-compatible alias for the shared dataset configuration."""
    return common.get_dataset_config(dataset, data_dir, download=download)


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
    ``trajectory.adam_metric_factored``.
    """
    model.eval()
    L_buf, Phi_buf = [], []

    features_dict = {}
    def _hook(_, inp, __):
        features_dict['phi'] = inp[0]

    handle = common.classifier_layer(model).register_forward_hook(_hook)

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

    Keeping the factors on the GPU avoids host transfer and the n-by-(CF)
    materialisation. A generic kernel matvec is still O(nCF) arithmetic;
    factorisation is primarily the decisive memory reduction, not an
    O(n(C+F)) arithmetic shortcut.

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
# Numerically stable centered objective used by Algorithms 2 and 3

_FP64_CHUNK_SIZE = 2048


def _validate_factored_swap_problem(traj, m, max_iters, log_freq=1,
                                    max_pairs=0, batch_pairs=1):
    """Validate the PSD assumptions used by both manuscript algorithms."""
    if not traj:
        raise ValueError("The trajectory must contain at least one checkpoint")
    n = int(traj[0][0].shape[0])
    if not 1 <= m < n:
        raise ValueError(f"m must satisfy 1 <= m < n (got m={m}, n={n})")
    if max_iters < 0:
        raise ValueError("max_iters must be non-negative")
    if log_freq <= 0:
        raise ValueError("log_freq must be positive")
    if max_pairs < 0:
        raise ValueError("max_pairs must be non-negative")
    if batch_pairs <= 0:
        raise ValueError("batch_pairs must be positive")

    device = traj[0][0].device
    positive_weight = False
    for k, (L, Phi, M, w) in enumerate(traj):
        if L.ndim != 2 or Phi.ndim != 2 or L.shape[0] != n or Phi.shape[0] != n:
            raise ValueError(f"Invalid factor shapes at checkpoint {k}: "
                             f"L={tuple(L.shape)}, Phi={tuple(Phi.shape)}")
        if L.device != device or Phi.device != device:
            raise ValueError("All trajectory factors must be on the same device")
        weight = float(w)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Checkpoint weights must be finite and non-negative; got {w}")
        positive_weight |= weight > 0
        if M is not None:
            if M.shape != (L.shape[1], Phi.shape[1]):
                raise ValueError(f"Metric shape {tuple(M.shape)} does not match "
                                 f"({L.shape[1]}, {Phi.shape[1]}) at checkpoint {k}")
            if M.device != device:
                raise ValueError("Metrics and trajectory factors must share a device")
            if not bool(torch.isfinite(M).all()) or bool((M < 0).any()):
                raise ValueError("Exact screening requires a finite, non-negative metric")
    if not positive_weight:
        raise ValueError("At least one checkpoint weight must be positive")
    return n, device


def _outer_sum_fp64(L, Phi, index=None, chunk_size=_FP64_CHUNK_SIZE):
    """Return ``sum_i outer(L_i, Phi_i)`` with float64 accumulation."""
    if index is None:
        index = torch.arange(L.shape[0], device=L.device)
    else:
        index = torch.as_tensor(index, dtype=torch.long, device=L.device).reshape(-1)
    out = torch.zeros((L.shape[1], Phi.shape[1]), dtype=torch.float64, device=L.device)
    for start in range(0, index.numel(), chunk_size):
        idx = index[start:start + chunk_size]
        out.add_(L[idx].double().T @ Phi[idx].double())
    return out


def _centered_state(traj, selected, m):
    """Build Z_k = sum_{i in I} g_i^k - (m/n) sum_i g_i^k in float64."""
    n = int(traj[0][0].shape[0])
    centered = []
    constant = 0.0
    for L, Phi, M, w in traj:
        total = _outer_sum_fp64(L, Phi)
        subset = _outer_sum_fp64(L, Phi, selected)
        z = subset - (float(m) / n) * total
        metric = None if M is None else M.double()
        mean = total / n
        mean_sq = mean * mean
        constant += float(w) * float(
            (mean_sq if metric is None else mean_sq * metric).sum().item())
        centered.append(z)
    return centered, constant


def _centered_D(traj, centered, m):
    """Evaluate the manuscript's non-negative D(alpha)."""
    total = 0.0
    for (_, _, M, w), z in zip(traj, centered):
        metric = None if M is None else M.double()
        sq = z * z
        total += float(w) * float(
            (sq if metric is None else sq * metric).sum().item()) / (m ** 2)
    return total


def _centered_gradient(traj, centered, m, chunk_size=_FP64_CHUNK_SIZE):
    """Compute h_i=(2/m^2)<g_i,MZ> without float32 cancellation."""
    n = int(traj[0][0].shape[0])
    h = torch.zeros(n, dtype=torch.float64, device=traj[0][0].device)
    scale = 2.0 / (m ** 2)
    for (L, Phi, M, w), z in zip(traj, centered):
        tangent = z if M is None else M.double() * z
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            Lb, Pb = L[start:stop].double(), Phi[start:stop].double()
            h[start:stop].add_(scale * float(w) * ((Lb @ tangent) * Pb).sum(dim=1))
    return h


def _pair_changes_fp64(traj, centered, m, i_idx, j_idx, linear_terms=None):
    """Exact centered float64 objective changes for exchanges ``i -> j``."""
    factor_device = traj[0][0].device
    ii = torch.as_tensor(i_idx, dtype=torch.long, device=factor_device).reshape(-1)
    jj = torch.as_tensor(j_idx, dtype=torch.long, device=factor_device).reshape(-1)
    if ii.numel() != jj.numel():
        raise ValueError("i_idx and j_idx must have the same length")
    if linear_terms is None:
        changes = torch.zeros(ii.numel(), dtype=torch.float64, device=factor_device)
    else:
        changes = torch.as_tensor(
            linear_terms, dtype=torch.float64, device=factor_device).reshape(-1).clone()
        if changes.numel() != ii.numel():
            raise ValueError("linear_terms and pair indices must have the same length")

    for (L, Phi, M, w), z in zip(traj, centered):
        li, lj = L[ii].double(), L[jj].double()
        pi, pj = Phi[ii].double(), Phi[jj].double()
        metric = None if M is None else M.double()
        if linear_terms is None:
            tangent = z if metric is None else metric * z
            inner_i = ((li @ tangent) * pi).sum(dim=1)
            inner_j = ((lj @ tangent) * pj).sum(dim=1)

        if metric is None:
            diag_i = (li * li).sum(dim=1) * (pi * pi).sum(dim=1)
            diag_j = (lj * lj).sum(dim=1) * (pj * pj).sum(dim=1)
            cross = (li * lj).sum(dim=1) * (pi * pj).sum(dim=1)
        else:
            diag_i = (((li * li) @ metric) * (pi * pi)).sum(dim=1)
            diag_j = (((lj * lj) @ metric) * (pj * pj)).sum(dim=1)
            cross = (((li * lj) @ metric) * (pi * pj)).sum(dim=1)

        # Preserve the PSD lower bound if the norm is a few ulps negative.
        curvature = (diag_i + diag_j - 2.0 * cross).clamp_min_(0.0)
        changes.add_(float(w) * curvature / (m ** 2))
        if linear_terms is None:
            changes.add_(float(w) * 2.0 * (inner_j - inner_i) / (m ** 2))
    return changes


def _apply_centered_swap(traj, centered, i, j):
    for (L, Phi, _, _), z in zip(traj, centered):
        z.add_(torch.outer(L[j].double(), Phi[j].double())
               - torch.outer(L[i].double(), Phi[i].double()))


def _state_record(rule, swaps, pair_searches, D, constant, gamma,
                  lower_bound, certificate):
    return {"rule": rule, "iteration": swaps, "swaps": swaps,
            "pair_searches": pair_searches, "objective": D - constant,
            "D": D, "gamma": gamma, "lower_bound": lower_bound,
            "certificate": certificate}


# Algorithms 3 and 2 from the manuscript
# ─────────────────────────────────────────────────────────────────────────────

def gradient_ranked_single_swap(traj, m, max_iters=5000, log_freq=5, device='cuda',
                                 seed=None):
    """Faithful Algorithm 3 with centered float64 decisions and certificate.

    At each state only the pair with maximum selected h_i and minimum
    unselected h_j is tested.  A non-negative exact change therefore stops
    Algorithm 3 but does *not* claim one-swap local optimality.  The returned
    history maintains the manuscript's best lower bound and monotone
    history-aware excess certificate.
    """
    n, factor_device = _validate_factored_swap_problem(
        traj, m, max_iters, log_freq=log_freq)
    g = torch.Generator(device='cpu')
    if seed is not None:
        g.manual_seed(seed)
    selected = torch.zeros(n, dtype=torch.bool, device=factor_device)
    selected[torch.randperm(n, generator=g)[:m].to(factor_device)] = True
    centered, constant = _centered_state(traj, torch.where(selected)[0], m)

    history = []
    swaps = pair_searches = states_evaluated = 0
    reason, final_change = 'max_iters', None
    last_accepted_change = None
    final_gamma = float('nan')
    best_lower_bound = -float('inf')
    final_D = final_objective = final_certificate = float('nan')
    r = min(m, n - m)

    while True:
        D = _centered_D(traj, centered, m)
        h = _centered_gradient(traj, centered, m)
        states_evaluated += 1
        sel, uns = torch.where(selected)[0], torch.where(~selected)[0]
        i_t = int(sel[torch.argmax(h[sel])].item())
        j_t = int(uns[torch.argmin(h[uns])].item())
        final_gamma = float((h[i_t] - h[j_t]).item())
        best_lower_bound = max(best_lower_bound,
                               D - r * max(final_gamma, 0.0))
        final_certificate = max(0.0, D - best_lower_bound)
        final_D, final_objective = D, D - constant
        record = _state_record('ranked', swaps, pair_searches, D, constant,
                               final_gamma, best_lower_bound, final_certificate)
        if swaps % log_freq == 0:
            history.append(record)
        final_change = None

        if final_gamma <= 0.0:
            reason = 'gamma<=0 (globally optimal certificate)'
            break
        if pair_searches >= max_iters:
            reason = 'max_iters'
            break

        final_change = float(
            _pair_changes_fp64(traj, centered, m, [i_t], [j_t])[0].item())
        pair_searches += 1
        if final_change >= 0.0:
            reason = 'top-ranked candidate rejected (Algorithm 3 stopping rule)'
            break

        _apply_centered_swap(traj, centered, i_t, j_t)
        selected[i_t], selected[j_t] = False, True
        swaps += 1
        last_accepted_change = final_change

    final_record = _state_record('ranked', swaps, pair_searches, final_D, constant,
                                 final_gamma, best_lower_bound, final_certificate)
    final_record.update({"final": True, "change": final_change, "reason": reason})
    if not history or history[-1].get("iteration") != swaps:
        history.append(final_record)
    else:
        history[-1].update(final_record)

    diag = {"algorithm": 3, "rule": "ranked", "iterations": swaps,
            "pair_searches": pair_searches, "states_evaluated": states_evaluated,
            "swaps": swaps, "reason": reason, "final_gamma": final_gamma,
            "final_change": final_change,
            "last_accepted_change": last_accepted_change, "D": final_D,
            "objective": final_objective, "lower_bound": best_lower_bound,
            "certificate": final_certificate, "certificate_valid": True,
            "globally_optimal": final_gamma <= 0.0,
            "one_swap_local_optimum": final_gamma <= 0.0,
            "decision_dtype": "float64", "centered_acceptance": True}
    return torch.where(selected)[0].cpu(), history, diag


def best_swap(traj, m, max_iters=5000, log_freq=5, device='cuda', seed=None,
              max_pairs=0, batch_pairs=8192, logger=None):
    """Faithful Algorithm 2; ``max_pairs=0`` performs the exact pair search.

    Feasible pairs are visited by increasing linear term h_j-h_i.  Their
    curvature is a non-negative squared norm, so once the next linear term
    cannot beat the best exact change already seen, the argmin is certified
    without forming P.  A positive ``max_pairs`` deliberately truncates this
    proof and is labelled non-exact in the diagnostics.
    """
    n, factor_device = _validate_factored_swap_problem(
        traj, m, max_iters, log_freq=log_freq,
        max_pairs=max_pairs, batch_pairs=batch_pairs)
    g = torch.Generator(device='cpu')
    if seed is not None:
        g.manual_seed(seed)
    selected = torch.zeros(n, dtype=torch.bool, device=factor_device)
    selected[torch.randperm(n, generator=g)[:m].to(factor_device)] = True
    centered, constant = _centered_state(traj, torch.where(selected)[0], m)

    history = []
    swaps = pair_searches = states_evaluated = examined_total = 0
    reason, final_change = 'max_iters', None
    last_accepted_change = None
    best_lower_bound = -float('inf')
    final_D = final_objective = final_gamma = final_certificate = float('nan')
    all_pair_searches_exact = True
    one_swap_local_optimum = False
    r = min(m, n - m)

    while True:
        D = _centered_D(traj, centered, m)
        h = _centered_gradient(traj, centered, m)
        states_evaluated += 1
        sel, uns = torch.where(selected)[0], torch.where(~selected)[0]
        drop = sel[torch.argsort(h[sel], descending=True)]
        add = uns[torch.argsort(h[uns])]
        final_gamma = float((h[drop[0]] - h[add[0]]).item())
        best_lower_bound = max(best_lower_bound,
                               D - r * max(final_gamma, 0.0))
        final_certificate = max(0.0, D - best_lower_bound)
        final_D, final_objective = D, D - constant
        record = _state_record('best', swaps, pair_searches, D, constant,
                               final_gamma, best_lower_bound, final_certificate)
        if swaps % log_freq == 0:
            history.append(record)
        final_change = None

        if swaps >= max_iters:
            reason = 'max_iters'
            break

        pairs = common.ranked_pairs(drop.cpu().numpy(), h[drop].cpu().numpy(),
                                    add.cpu().numpy(), h[add].cpu().numpy())
        best, best_pair, examined = float('inf'), None, 0
        search_complete = False
        total_pairs = int(sel.numel() * uns.numel())
        buf_lin, buf_i, buf_j = [], [], []

        def flush():
            nonlocal best, best_pair
            if not buf_i:
                return
            delta = _pair_changes_fp64(
                traj, centered, m, buf_i, buf_j, linear_terms=buf_lin)
            k = int(torch.argmin(delta))
            if float(delta[k]) < best:
                best, best_pair = float(delta[k]), (buf_i[k], buf_j[k])
            buf_lin.clear()
            buf_i.clear()
            buf_j.clear()

        for lin, i, j in pairs:
            if lin >= best:
                flush()
                if lin >= best:
                    search_complete = True
                    break
            buf_lin.append(lin)
            buf_i.append(i)
            buf_j.append(j)
            examined += 1
            if len(buf_i) >= batch_pairs:
                flush()
            if max_pairs and examined >= max_pairs:
                flush()
                search_complete = examined == total_pairs
                break
        else:
            search_complete = True
        flush()
        pair_searches += 1
        examined_total += examined
        all_pair_searches_exact &= search_complete
        final_change = None if best_pair is None else best

        if best_pair is None or best >= 0.0:
            if search_complete:
                reason = 'no improving swap exists (exact one-swap local optimum)'
                one_swap_local_optimum = True
            else:
                reason = (f'no improving swap in the {examined} best-ranked pairs '
                          f'(--max_pairs, not a local-optimality certificate)')
            break

        i_t, j_t = best_pair
        _apply_centered_swap(traj, centered, i_t, j_t)
        selected[i_t], selected[j_t] = False, True
        swaps += 1
        last_accepted_change = best
        if logger and ((swaps - 1) % log_freq == 0):
            qualifier = 'exact' if search_complete else 'truncated'
            logger.info(f"swap {swaps}: best {qualifier} change {best:.6e} "
                        f"({examined} pairs screened of {total_pairs})")

    globally_optimal = final_gamma <= 0.0
    one_swap_local_optimum |= globally_optimal
    final_record = _state_record('best', swaps, pair_searches, final_D, constant,
                                 final_gamma, best_lower_bound, final_certificate)
    final_record.update({"final": True, "change": final_change, "reason": reason})
    if not history or history[-1].get("iteration") != swaps:
        history.append(final_record)
    else:
        history[-1].update(final_record)

    diag = {"algorithm": 2, "rule": "best", "iterations": swaps,
            "pair_searches": pair_searches, "states_evaluated": states_evaluated,
            "swaps": swaps, "reason": reason, "pairs_screened": examined_total,
            "final_change": final_change,
            "last_accepted_change": last_accepted_change,
            "final_gamma": final_gamma,
            "D": final_D, "objective": final_objective,
            "lower_bound": best_lower_bound, "certificate": final_certificate,
            "certificate_valid": True, "globally_optimal": globally_optimal,
            "one_swap_local_optimum": one_swap_local_optimum,
            "all_pair_searches_exact": all_pair_searches_exact,
            "algorithm_2_exact": max_pairs == 0 and all_pair_searches_exact,
            "decision_dtype": "float64", "centered_acceptance": True}
    return torch.where(selected)[0].cpu(), history, diag


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fraction',       default=0.3,         type=float)
    p.add_argument('--seed', '-seed',   default=42,          type=int)
    p.add_argument('--checkpoint_dir', required=True,       type=str)
    p.add_argument('--dataset',        default='cifar100',  choices=list(common.DATASETS))
    p.add_argument('--architecture',   default=None, choices=list(common.ARCHITECTURES),
                   help='Network architecture. Defaults to resnet18, or lenet for MNIST.')
    p.add_argument('--data_dir',       required=True,       type=str,
                   help='Root of the staged dataset (e.g. $SLURM_TMPDIR/cifar100_data)')
    p.add_argument('--proj_dim',       default=0,           type=int,
                   help='Sparse-JL target dim per factor (0 = exact, no JL)')
    p.add_argument('--num_checkpoints', default=10,         type=int,
                   help='How many checkpoints from --checkpoint_dir feed the kernel. '
                        '0 = every checkpoint found.')
    p.add_argument('--checkpoint_selection', default='spread', choices=['spread', 'prefix'],
                   help="'spread' spaces the K checkpoints over the whole run; 'prefix' "
                        "takes the first K filenames (normally epochs 1-10), "
                        "which is the previous behaviour.")
    p.add_argument('--checkpoint_weighting', default='point',
                   choices=list(ref.CHECKPOINT_WEIGHTINGS),
                   help="'point' uses the saved checkpoint learning rate eta_k, matching "
                        "D_eta and D_Adam directly. 'window' sums learning rates over "
                        "the optimizer steps represented by each checkpoint.")
    p.add_argument('--checkpoint_epochs', default=None, type=int, nargs='+',
                   help='Explicit checkpoint epochs; overrides the two options above.')
    p.add_argument('--optimizer',      default='uniform',
                   choices=list(ref.OPTIMIZERS),
                   help='Objective weighting (w_k, M_k). See the module docstring. '
                        'NOTE the default optimises neither the learning-rate weighting '
                        'nor the Adam preconditioner; it exists to reproduce old runs.')
    p.add_argument('--metric',         default='auto', choices=list(ref.METRICS),
                   help='Override M_k independently of the step weight, allowing '
                        '--optimizer uniform --metric adam to isolate Adam coordinate '
                        'scaling from learning-rate weighting.')
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
                        "--optimizer sgd on a full-data checkpoint directory. Ignored when "
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
                        "this and selection.kl_full score identical inputs; "
                        "'train' reproduces the old runs. Either way the view is "
                        'materialised once and shared by every checkpoint.')
    p.add_argument('--selection_only', action='store_true', default=False,
                   help='Stop after saving indices + timing (skip retraining)')
    args = p.parse_args()

    common.set_seed(args.seed)

    use_metric = ref.resolve_metric(args.optimizer, args.metric)
    if use_metric and args.proj_dim > 0:
        raise ValueError(
            f"M_k = D_k^2 needs --proj_dim 0. The sparse-JL step projects L and Phi "
            f"independently, which preserves <L_i,L_j><Phi_i,Phi_j> in expectation but "
            f"not the coordinate-wise metric "
            f"sum_{{c,f}} M[c,f] L[i,c]L[j,c]Phi[i,f]Phi[j,f]."
        )
    if args.retrain_optimizer is None:
        args.retrain_optimizer = (
            'adam' if use_metric or args.optimizer in ('adam', 'adam_update') else 'sgd')

    # ── 0. Dataset ───────────────────────────────────────────────────────────
    cfg         = common.get_dataset_config(
        args.dataset, args.data_dir, download=args.download,
        architecture=args.architecture)
    NUM_CLASSES = cfg['num_classes']
    NUM_EPOCHS  = cfg['num_epochs']

    n_val   = int(0.1 * len(cfg['full_train']))
    n_train = len(cfg['full_train']) - n_val
    budget  = int(args.fraction * n_train)
    if not 1 <= budget < n_train:
        raise ValueError(
            f"--fraction={args.fraction!r} gives budget={budget}; require "
            f"1 <= budget < n_train={n_train}."
        )
    trainset, _ = random_split(
        cfg['full_train'], [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    pool_indices_sha256 = common.ordered_indices_fingerprint(trainset.indices)

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
        momentum=args.momentum_response, metric=args.metric,
        checkpoint_weighting=args.checkpoint_weighting,
        expected_dataset=args.dataset, expected_seed=args.seed,
        expected_architecture=cfg['architecture'],
        expected_pool_indices_sha256=pool_indices_sha256)
    num_ckpts_used = len(checkpoints)

    scratch_path = os.environ.get("SCRATCH", "/home/mila/a/ahmedm/scratch")
    base_dir     = f"{scratch_path}/gradmatch_swap/{args.dataset}/{budget}/seed_{args.seed}"
    os.makedirs(base_dir, exist_ok=True)

    # ── 2. Logging ───────────────────────────────────────────────────────────
    # file_prefix carries the checkpoint count (K{num_ckpts_used}) so that
    # sweeps over --num_checkpoints for the same (dataset, budget, seed) land
    # in the same base_dir without ever overwriting each other's files. The
    # The inferred method name keeps objective/algorithm ablations from
    # silently overwriting one another while retaining the historical name
    # for the default uniform/ranked/exact configuration.
    method = f"kl_faithful_{cfg['architecture']}" + ref.objective_tag(
        args.optimizer, args.metric, args.checkpoint_weighting,
        args.momentum_response)
    if args.rule != 'ranked':
        method += f'_{args.rule}'
    if args.proj_dim > 0:
        method += f'_proj{args.proj_dim}'
    if not args.include_bias:
        method += '_nobias'
    if args.scoring_transform != 'test':
        method += f'_{args.scoring_transform}view'
    if manifest is not None:
        trajectory_identity = {
            'manifest_run_id': manifest.get('run_id'),
            'manifest_fingerprint': common.config_fingerprint(manifest),
        }
    else:
        trajectory_identity = {
            'legacy_directory': os.path.realpath(args.checkpoint_dir),
            'files': [
                {
                    'path': os.path.realpath(c.path),
                    'size': os.stat(c.path).st_size,
                    'mtime_ns': os.stat(c.path).st_mtime_ns,
                }
                for c in checkpoints
            ],
        }
    run_config = {
        'schema_version': 2,
        'selector': 'kl_factored',
        'arguments': {**vars(args),
                      'checkpoint_dir': os.path.realpath(args.checkpoint_dir),
                      'data_dir': os.path.realpath(args.data_dir),
                      'init_ckpt': (os.path.realpath(args.init_ckpt)
                                    if args.init_ckpt else None)},
        'resolved': {
            'dataset': args.dataset, 'architecture': cfg['architecture'],
            'n_train': n_train, 'budget': budget,
            'pool_indices_sha256': pool_indices_sha256,
            'trajectory': trajectory_identity,
            'checkpoints': [
                {'path': os.path.realpath(c.path), 'epoch': c.epoch,
                 'step': c.step, 'lr': c.lr, 'weight': c.weight,
                 'window': list(c.window)}
                for c in checkpoints
            ],
        },
    }
    run_id = common.config_fingerprint(run_config)
    file_prefix = f"{method}_{budget}_K{num_ckpts_used}_cfg{run_id}"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(base_dir, f"{file_prefix}_training.log")),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Dataset={args.dataset}  architecture={cfg['architecture']}  "
                f"budget={budget}  seed={args.seed}  "
                f"device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info(f"Immutable run fingerprint: {run_id}")
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
    RUN_SAVE_PATH     = os.path.join(base_dir, f"{file_prefix}_run.json")
    if args.selection_only:
        stale = [path for path in (MODEL_SAVE_PATH, METRICS_SAVE_PATH)
                 if os.path.exists(path)]
        if stale:
            raise FileExistsError(
                "Refusing --selection_only because retraining artifacts already exist "
                f"for this exact run: {stale}. Remove them explicitly or change the "
                "configuration.")
    common.atomic_json_dump(
        {'schema_version': 1, 'run_id': run_id, 'status': 'configured',
         'config': run_config}, RUN_SAVE_PATH)

    # ── 3. Scoring pool ──────────────────────────────────────────────────────
    #      Materialised once: a dataloader under the 'train' transform
    #      redraws the augmentation on every pass, so each checkpoint would
    #      see a different view of the same example and P would mix
    #      gradients taken at different inputs.
    X, Y = d_alpha.materialise(common.scoring_subset(cfg, trainset, args.scoring_transform),
                               device, batch_size=BATCH_SIZE, num_workers=args.num_workers)
    testloader  = DataLoader(cfg['testset'], batch_size=BATCH_SIZE, shuffle=False,
                             pin_memory=True, num_workers=args.num_workers)

    # ── 4. Reference model ───────────────────────────────────────────────────
    ref_model = common.build_model(
        cfg, NUM_CLASSES, device, architecture=cfg['architecture'])
    FEAT_DIM = common.feature_dim(ref_model)
    uses_batchnorm = any(
        isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        for module in ref_model.modules())
    if uses_batchnorm:
        logger.warning(
            "The model contains BatchNorm. Selection scores per-example gradients "
            "with frozen running statistics (model.eval()); this is a reproducible "
            "surrogate, not the train-mode BatchNorm gradient field that produced "
            "the reference trajectory.")

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
    logger.info(f"Stopped after {diag['swaps']} accepted swaps and "
                f"{diag['pair_searches']} pair searches ({diag['reason']}), "
                f"{len(selected_indices)} samples selected. "
                f"f: {objective_history[0]['objective']:.6e} -> "
                f"{objective_history[-1]['objective']:.6e}")
    if diag['reason'] == 'max_iters':
        logger.warning("--max_iters reached, so the search had not settled; the subset "
                       "is the best found, not a local optimum.")

    common.save_timing(TIMING_SAVE_PATH, timer, logger, method=method,
                       run_id=run_id,
                       reference_epochs_used=max(c.epoch for c in checkpoints),
                       swap_iterations=diag['iterations'],
                       objective_optimizer=args.optimizer,
                       objective_metric='D_k^2' if use_metric else 'I',
                       momentum_response=args.momentum_response,
                       scoring_transform=args.scoring_transform,
                       batchnorm_scoring_mode=('eval_frozen' if uses_batchnorm else 'none'),
                       train_mode_gradient_correspondence=not uses_batchnorm,
                       include_bias=args.include_bias,
                       checkpoint_selection=args.checkpoint_selection,
                       checkpoint_weighting=args.checkpoint_weighting,
                       checkpoint_steps=[c.step for c in checkpoints],
                       checkpoint_weights=[c.weight for c in checkpoints],
                       num_checkpoints_used=num_ckpts_used,
                       **{k_: v for k_, v in diag.items() if k_ != 'iterations'})
    torch.save(selected_indices, INDICES_SAVE_PATH)
    common.atomic_json_dump(objective_history, OBJ_SAVE_PATH)
    if args.selection_only:
        common.atomic_json_dump(
            {'schema_version': 1, 'run_id': run_id,
             'status': 'selection_complete', 'config': run_config,
             'artifacts': {'indices': INDICES_SAVE_PATH,
                           'objective': OBJ_SAVE_PATH,
                           'timing': TIMING_SAVE_PATH}},
            RUN_SAVE_PATH)
        logger.info("--selection_only set: skipping retraining.")
        return

    # ── 8. Retrain on selected subset ────────────────────────────────────────
    del traj, X, Y
    if device == 'cuda':
        torch.cuda.empty_cache()

    train_generator = torch.Generator().manual_seed(args.seed + 10_000)
    trainloader = DataLoader(
        Subset(trainset, selected_indices),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=args.num_workers,
        generator=train_generator,
    )

    common.set_seed(args.seed)
    model = common.build_model(
        cfg, NUM_CLASSES, device, architecture=cfg['architecture'])
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
    common.atomic_json_dump(accuracy_history, METRICS_SAVE_PATH)
    common.atomic_json_dump(
        {'schema_version': 1, 'run_id': run_id, 'status': 'complete',
         'config': run_config,
         'artifacts': {'indices': INDICES_SAVE_PATH, 'objective': OBJ_SAVE_PATH,
                       'timing': TIMING_SAVE_PATH, 'model': MODEL_SAVE_PATH,
                       'metrics': METRICS_SAVE_PATH}},
        RUN_SAVE_PATH)
    logger.info(f"Saved all artefacts to {base_dir}")


if __name__ == "__main__":
    main()
