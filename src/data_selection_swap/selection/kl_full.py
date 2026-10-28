"""
Algorithm 3 (gradient-ranked single-swap descent) over *all* trainable
parameters.

Why a second script
───────────────────
The factored selector restricts the gradient to the last layer, where
g_i = vec(l_i phi_i^T) factorises and the kernel column P[:,i] costs
O(n(C+F)). Nothing like that holds for the full network: convolutional
per-sample gradients do not factorise, and storing G_t for ResNet-18 on
CIFAR-100 would need 50000 x 11.2M x 4 B = 2.2 TB per checkpoint. P is not
available here, in any form, ever.

It is not needed. Write u_I^k = sum_{i in I} g_i^k. Then

    D(alpha) = sum_k w_k || u_I^k/m - gbar_k ||^2_{M_k}                (1)

which is the manuscript's trajectory-averaged gradient discrepancy, and
which differs from f(alpha) = (1/m^2) alpha'P alpha + q'alpha only by the
constant sum_k w_k ||gbar_k||^2. So D and f have identical swap
decisions, and D is the non-negative one, so it is what gets reported.

Everything the algorithm needs follows from (1) with *aggregate*
gradients only:

  ranking    h_i = dD/dalpha_i = (2/m) sum_k w_k <g_i^k, M_k v_k>,
             v_k = u_I^k/m - gbar_k. Fixing the tangent M_k v_k turns
             all n of these into one forward-mode JVP sweep per
             checkpoint — one forward pass over the pool, no per-sample
             gradient ever materialised. h runs through the model's
             float32 ops, so it is used only to *rank* candidates.

  acceptance for any delta in {-1,0,1}^n with 1'delta = 0,

                 Delta = sum_k w_k [ (2/m) <v_k, M_k u_delta^k>
                                     + (1/m^2) ||u_delta^k||^2_{M_k} ]

             where u_delta^k is an aggregate gradient over the handful of
             swapped points. This is delta'h + (1/m^2) delta'P delta
             regrouped so that every term is an inner product of float64
             aggregates: near a converged subset the two contributions
             nearly cancel, and reusing the float32 h here would let its
             rounding error decide whether a swap is accepted. It is
             exact, not a linearisation, since D is quadratic in alpha.
             The same u_delta^k then updates u_I^k, so the accept test
             costs nothing extra.

Cost per iteration is therefore one forward pass over the pool per
checkpoint: about 1960 batches at n = 50000, batch 256 and K = 10. The
2000 single swaps alg3.sh asks for would take hours, so --max_iters
defaults to 200 — which may well stop before the search settles, and is
why the stopping reason, the final gradient gap and the best rejected
exact change are all logged and saved rather than left implicit.
--swaps_per_sweep amortises the sweep over a block of exchanges, backing
off when the block's exact change is not negative; --candidates relaxes
Algorithm 3's stopping rule. At --swaps_per_sweep 1 --candidates 1 the
loop is literally Algorithm 3.

Objective weighting (--optimizer) is shared with ``selection.kl``
and documented in ``trajectory``: uniform, sgd, adam (the KL
weighting, w_k = sum eta_s a_s^2 with M_k = D_k^2) and adam_update
(the squared update mismatch, w_k = sum eta_s^2 a_s^2).

Count sketch (--proj_dim)
─────────────────────────
--proj_dim p > 0 replaces every stored aggregate with a count sketch
R g ∈ R^p (one random bucket and one random sign per coordinate, seed
--proj_seed). R is linear and has an exact adjoint, so D, the swap
test and the JVP all run in the sketched coordinates. The tangent is
Rᵀ(R v); with an Adam metric it is sqrt(M) ⊙ Rᵀ(R (sqrt(M) ⊙ v)), and
M is read for that checkpoint and dropped rather than kept for all K.
Acceptance is exact for the sketched objective ||R v||^2, which is what
the JVP ranks. p = 0 is the exact path above. Sketched runs keep
checkpoint weights on CPU and move one checkpoint onto the pool's
device for each backward or JVP.

This is Algorithm 3 over full-network gradients, not the manuscript's
exact best-swap search. The exact search needs the per-pair curvature
||g_j - g_i||^2_M, which is unavailable here for the same reason P is;
``selection.kl --rule best`` does it on the last-layer kernel.
"""

import argparse
import glob
import json
import logging
import math
import os

import torch
import torch.nn.functional as F

from .. import common
from .. import objective as d_alpha
from .. import trajectory as ref
from ..paths import default_results_root, storage_variant, tree


# ─────────────────────────────────────────────────────────────────────────────
# Functional plumbing
#   Parameters are carried as dicts rather than loaded into the module,
#   so the JVP sweep and the aggregate backwards run against exactly the
#   same weights and buffers, and no checkpoint is re-read from disk
#   inside the swap loop.
# ─────────────────────────────────────────────────────────────────────────────

def split_state(model, state_dict, device):
    """(params, buffers) for functional_call, params in module order."""
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    params, buffers = {}, {}
    for n, v in state_dict.items():
        (params if n in names else buffers)[n] = v.to(device)
    missing = names - set(params)
    if missing:
        raise KeyError(f"Checkpoint is missing trainable parameters: {sorted(missing)}")
    return params, buffers


def flat_dim(params):
    return sum(v.numel() for v in params.values())


def flatten(d, order):
    return torch.cat([d[n].reshape(-1) for n in order])


def unflatten(flat, template, order):
    out, off = {}, 0
    for n in order:
        v = template[n]
        out[n] = flat[off:off + v.numel()].view_as(v)
        off += v.numel()
    return out


class CountSketch:
    """
    Implicit Johnson-Lindenstrauss map R: R^d -> R^p.

    One bucket and one sign per coordinate (a count sketch). The map is
    stored as an index vector, never as a d-by-p matrix. It is linear, so
    R(sum_i g_i) = sum_i R g_i, and the adjoint satisfies
    <R x, y> = <x, R^T y> exactly.
    """

    def __init__(self, dim, proj_dim, seed):
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if proj_dim <= 0:
            raise ValueError(f"proj_dim must be positive, got {proj_dim}")
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(seed))
        self.dim = int(dim)
        self.proj_dim = int(proj_dim)
        self.seed = int(seed)
        self.buckets = torch.randint(0, self.proj_dim, (self.dim,),
                                     generator=generator, dtype=torch.int64)
        self.signs = (torch.randint(0, 2, (self.dim,), generator=generator,
                                    dtype=torch.int64) * 2 - 1).to(torch.int8)

    def apply(self, flat):
        """R @ flat, for a single vector of length d."""
        if flat.numel() != self.dim:
            raise ValueError(f"expected {self.dim} entries, got {flat.numel()}")
        signed = flat * self.signs.to(device=flat.device, dtype=flat.dtype)
        out = torch.zeros(self.proj_dim, dtype=flat.dtype, device=flat.device)
        return out.scatter_add_(0, self.buckets.to(flat.device), signed)

    def apply_batch(self, flats):
        """R applied to each row. ``flats`` is ``[B, d]`` and the result is ``[B, p]``."""
        if flats.dim() != 2 or flats.shape[1] != self.dim:
            raise ValueError(f"expected [B, {self.dim}], got {tuple(flats.shape)}")
        signs = self.signs.to(device=flats.device, dtype=flats.dtype)
        buckets = self.buckets.to(flats.device).expand(flats.shape[0], -1)
        signed = flats * signs
        out = flats.new_zeros(flats.shape[0], self.proj_dim)
        return out.scatter_add_(1, buckets, signed)

    def adjoint(self, sketched):
        """R^T @ sketched, a vector of length d."""
        if sketched.numel() != self.proj_dim:
            raise ValueError(
                f"expected {self.proj_dim} sketch entries, got {sketched.numel()}")
        buckets = self.buckets.to(sketched.device)
        signs = self.signs.to(device=sketched.device, dtype=sketched.dtype)
        return signs * sketched[buckets]


def grad_sum(model, params, buffers, X, Y, index, batch_size, order):
    """sum_{i in index} g_i, flattened, in float64."""
    def loss(p, xb, yb):
        logits = torch.func.functional_call(model, {**p, **buffers}, (xb,))
        return F.cross_entropy(logits, yb, reduction='sum')

    total = torch.zeros(flat_dim(params), dtype=torch.float64, device=X.device)
    for start in range(0, index.numel(), batch_size):
        sel = index[start:start + batch_size]
        g = torch.func.grad(loss)(params, X[sel], Y[sel])
        total += flatten(g, order).double()
    return total


def directional_derivatives(model, params, buffers, tangent, X, Y, batch_size, out):
    """out[i] += <g_i, tangent> for every i in the pool, one forward pass per batch."""
    n = X.shape[0]
    for start in range(0, n, batch_size):
        xb, yb = X[start:start + batch_size], Y[start:start + batch_size]

        def f(p):
            logits = torch.func.functional_call(model, {**p, **buffers}, (xb,))
            return F.cross_entropy(logits, yb, reduction='none')

        _, jv = torch.func.jvp(f, (params,), (tangent,))
        out[start:start + batch_size] = jv.double()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory state
# ─────────────────────────────────────────────────────────────────────────────

class Trajectory:
    """
    Per-checkpoint (params, buffers, w_k, M_k, gbar_k) plus the running
    aggregate u_I^k. Weights stay resident: 11.2M params x 4 B x 10
    checkpoints is under half a GB, against a disk read per iteration.
    The fp64 vectors are the real cost, ~90 MB each; v_k = u_I^k/m -
    gbar_k is a difference of nearly equal quantities, so fp64 is not
    optional.

    With a count sketch, those fp64 vectors live in R^p instead of R^d
    and the weights stay on CPU. One checkpoint is staged onto the pool
    device for the backward or JVP that needs it. An Adam metric is
    applied as sqrt(M) before the sketch and is not kept resident.
    """

    def __init__(self, model, checkpoints, use_metric, X, Y, device,
                 batch_size, adam_eps=1e-8, logger=None, metric_scale=None,
                 proj_dim=0, proj_seed=0):
        self.model = model.eval()
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.order = [n for n, p in model.named_parameters() if p.requires_grad]
        self.weights = [c.weight for c in checkpoints]
        self.checkpoints = list(checkpoints)
        self.named = list(model.named_parameters())
        self.adam_eps = adam_eps
        self.use_metric = use_metric
        self.proj_dim = int(proj_dim)
        self.proj_seed = int(proj_seed) if self.proj_dim else None
        if self.proj_dim < 0:
            raise ValueError(f"proj_dim must be >= 0, got {self.proj_dim}")
        param_dim = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.sketch = (CountSketch(param_dim, self.proj_dim, int(proj_seed))
                       if self.proj_dim else None)
        # Preconditioning is baked into the sketch, so M itself is not stored.
        self.preconditioned = bool(use_metric and self.sketch is not None)
        self.weight_device = 'cpu' if self.sketch is not None else device
        self.states, self.metrics, self.gbar = [], [], []
        self.u = None
        self.metric_scale = 1.0

        if self.preconditioned:
            # One scale shared by every checkpoint, matching the exact path,
            # computed without retaining the diagonals.
            means = [float(ref.adam_metric_flat(c, self.named, eps=adam_eps,
                                                power=2).mean())
                     for c in checkpoints]
            self.metric_scale = (metric_scale if metric_scale is not None
                                 else max(means))
            if logger:
                logger.info(f"Metric M_k = D_k^2 rescaled by 1/{self.metric_scale:.4g} "
                            f"(global, streamed into the sketch, D is in those units)")

        if logger and self.sketch is not None:
            logger.info(f"Count sketch R: R^{param_dim} -> R^{self.proj_dim} "
                        f"(seed {int(proj_seed)}); storing R g, weights on CPU.")

        pool = torch.arange(X.shape[0], device=device)
        for c in checkpoints:
            params, buffers = split_state(
                model, ref.load_weights(c.path), self.weight_device)
            self.states.append((params, buffers))
            if use_metric and self.sketch is None:
                self.metrics.append(
                    ref.adam_metric_flat(c, self.named, eps=adam_eps, power=2).to(device))
            else:
                self.metrics.append(None)
            if logger:
                logger.info(f"  checkpoint epoch {c.epoch:3d} (step {c.step}): "
                            f"mean gradient over the pool ...")
            self.gbar.append(self._reduce(params, buffers, pool, c) / X.shape[0])

        if use_metric and self.sketch is None:
            # (sqrt(vhat)+eps)^-2 spans many orders of magnitude; one scale
            # shared by every checkpoint tames it without disturbing the
            # relative weighting, so no swap decision changes.
            self.metric_scale = (metric_scale if metric_scale is not None
                                 else max(float(M.mean()) for M in self.metrics))
            self.metrics = [M / self.metric_scale for M in self.metrics]
            if logger:
                logger.info(f"Metric M_k = D_k^2 rescaled by 1/{self.metric_scale:.4g} "
                            f"(global, D is in those units)")

    def __len__(self):
        return len(self.states)

    def _stage(self, params, buffers):
        """One checkpoint on the pool device. Already-resident weights are not copied."""
        device = self.X.device
        if next(iter(params.values())).device == device:
            return params, buffers
        return ({n: v.to(device) for n, v in params.items()},
                {n: v.to(device) for n, v in buffers.items()})

    def _sqrt_scaled_metric(self, checkpoint):
        """sqrt(M_k / scale), loaded for this checkpoint and not retained."""
        metric = ref.adam_metric_flat(
            checkpoint, self.named, eps=self.adam_eps, power=2)
        return (metric / self.metric_scale).sqrt()

    def _reduce(self, params, buffers, index, checkpoint):
        """
        sum_{i in index} g_i^k, flattened, in float64.

        With a sketch, the full aggregate is temporary: an Adam metric
        multiplies it as sqrt(M), then R is applied and the R^d vector
        is dropped. Without a sketch this is the exact aggregate.
        """
        params, buffers = self._stage(params, buffers)
        total = grad_sum(self.model, params, buffers, self.X, self.Y, index,
                         self.batch_size, self.order)
        if self.sketch is None:
            return total
        if self.preconditioned:
            total = total * self._sqrt_scaled_metric(checkpoint).to(
                device=total.device, dtype=total.dtype)
        return self.sketch.apply(total)

    def _ranking_direction(self, checkpoint, metric, residual):
        """
        Flat tangent whose inner product with g_i is the sketched (or exact)
        coordinate of dD/dv. Exact path: M v. Sketched path: R^T (R v),
        or sqrt(M) ⊙ R^T (R (sqrt(M) ⊙ v)) when the metric was folded in.
        """
        if self.sketch is None:
            return residual if metric is None else metric * residual
        direction = self.sketch.adjoint(residual)
        if self.preconditioned:
            direction = direction * self._sqrt_scaled_metric(checkpoint).to(
                device=direction.device, dtype=direction.dtype)
        return direction

    def set_subset(self, index):
        """Recompute u_I^k from scratch."""
        self.u = [self._reduce(p, b, index, c)
                  for (p, b), c in zip(self.states, self.checkpoints)]

    def aggregate(self, index):
        """sum_{i in index} g_i^k for every checkpoint."""
        return [self._reduce(p, b, index, c)
                for (p, b), c in zip(self.states, self.checkpoints)]

    def residuals(self, m):
        """v_k = u_I^k/m - gbar_k, one at a time: each is ~90 MB in float64."""
        for u, g in zip(self.u, self.gbar):
            yield u / m - g

    def D(self, m):
        """sum_k w_k ||u_I^k/m - gbar_k||^2_{M_k} >= 0."""
        return sum(w * d_alpha.weighted_sq_norm(v, M)
                   for w, v, M in zip(self.weights, self.residuals(m), self.metrics))

    def exact_change(self, deltas, m):
        """
        Delta = sum_k w_k [ (2/m)<v_k, M_k u_delta^k> + (1/m^2)||u_delta^k||^2_M ].

        Algebraically delta'h + (1/m^2) delta'P delta, but evaluated
        entirely in float64 from the aggregates rather than by summing
        entries of the float32 h. Near convergence the linear and
        curvature terms very nearly cancel, and that difference is
        exactly what decides acceptance. A count sketch stores R(sqrt(M) ⊙ u)
        with no separate M, so the same formula is the exact change of
        the sketched objective.
        """
        total = 0.0
        for w, M, v, du in zip(self.weights, self.metrics, self.residuals(m), deltas):
            Mdu = du if M is None else M * du
            total += w * ((2.0 / m) * float((v * Mdu).sum())
                          + float((du * Mdu).sum()) / m ** 2)
        return total

    def gradient(self, m):
        """
        h_i = (2/m) sum_k w_k <g_i^k, M_k v_k>, one JVP sweep per checkpoint.

        The tangent is normalised before the sweep and the scale restored
        after: v_k is a residual that gets small as the subset improves,
        and the forward pass runs in the model's float32. Under a count
        sketch the tangent is the adjoint of the sketched residual, so
        <g_i, R^T R v> = <R g_i, R v> and h is the exact derivative of
        the sketched objective.
        """
        h = torch.zeros(self.X.shape[0], dtype=torch.float64, device=self.X.device)
        scratch = torch.empty_like(h)
        for checkpoint, (params, buffers), w, M, v in zip(
                self.checkpoints, self.states, self.weights, self.metrics,
                self.residuals(m)):
            params, buffers = self._stage(params, buffers)
            t = self._ranking_direction(checkpoint, M, v)
            norm = float(t.norm())
            if norm == 0.0:
                continue
            tangent = unflatten((t / norm).to(next(iter(params.values())).dtype),
                                params, self.order)
            directional_derivatives(self.model, params, buffers, tangent,
                                    self.X, self.Y, self.batch_size, scratch)
            h += (2.0 / m) * w * norm * scratch
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 3
# ─────────────────────────────────────────────────────────────────────────────

def _starting_index(n, m, seed, init_index, device):
    """Sorted pool positions of the subset Algorithm 3 starts from.

    ``init_index`` is that subset. Without it the start is a seeded random
    draw of size ``m``.
    """
    if init_index is None:
        generator = torch.Generator(device='cpu')
        if seed is not None:
            generator.manual_seed(seed)
        return torch.randperm(n, generator=generator)[:m].to(device).sort().values
    index = torch.as_tensor(init_index).detach().reshape(-1).cpu()
    if index.dtype.is_floating_point or index.dtype == torch.bool:
        raise ValueError(
            f"init_index must be integer pool positions, got dtype {index.dtype}")
    index = index.to(torch.long)
    if index.numel() != m:
        raise ValueError(
            f"init_index has {index.numel()} entries; the budget is {m}")
    if int(index.unique().numel()) != m:
        raise ValueError("init_index contains duplicate positions")
    if m and (int(index.min()) < 0 or int(index.max()) >= n):
        raise ValueError(
            f"init_index positions must lie in 0..{n - 1}, "
            f"got {int(index.min())}..{int(index.max())}")
    return index.to(device).sort().values


def gradient_ranked_single_swap(traj, m, max_iters=200, swaps_per_sweep=1,
                                candidates=1, refresh_every=0, logger=None, seed=None,
                                init_index=None):
    """
    Rank exchanges by h, then accept only on a negative *exact* change.

    ``init_index`` starts the search from that subset instead of a random one.
    With ``max_iters=0`` the returned subset is that index, sorted.

    swaps_per_sweep > 1 exchanges the s worst-ranked selected points for
    the s best-ranked unselected ones in a single step. The gradient
    ranking is only a linearisation, so a block can fail where each of
    its swaps would have succeeded; the block size is halved until the
    exact change is negative.

    --candidates c relaxes Algorithm 3's stopping rule, which declares
    convergence as soon as its single top-ranked pair is rejected even
    though a lower-ranked pair may still improve D. With c > 1 the top c
    pairs are tested before giving up. This is a partial fix: it is not
    the exact best-swap search, because the curvature term
    (1/m^2)||g_j - g_i||^2_M is not available per pair without per-sample
    gradients. ``selection.kl --rule best`` does the exact version
    on the last-layer kernel.

    Returns the stopping reason alongside the subset, since "converged"
    and "ran out of sweeps" mean very different things here.
    """
    n = int(traj.X.shape[0])
    if not 1 <= m < n:
        raise ValueError(f"m must satisfy 1 <= m < n (got m={m}, n={n})")
    if max_iters < 0:
        raise ValueError("max_iters must be non-negative")
    if swaps_per_sweep <= 0:
        raise ValueError("swaps_per_sweep must be positive")
    if candidates <= 0:
        raise ValueError("candidates must be positive")
    if refresh_every < 0:
        raise ValueError("refresh_every must be non-negative")

    index = _starting_index(n, m, seed, init_index, traj.X.device)
    selected = torch.zeros(n, dtype=torch.bool, device=traj.X.device)
    selected[index] = True
    traj.set_subset(index)

    initial_D = traj.D(m)
    history = [{"iteration": 0, "D": initial_D, "swaps": 0}]
    swaps = accepted_updates = sweeps = attempted_exchanges = 0
    gamma = float('nan')
    reason = 'max_iters'
    last_change = last_rejected = best_rejected = None
    best_lower_bound = -float('inf')
    final_certificate = None
    r = min(m, n - m)
    cand = min(candidates, m * (n - m))

    def attempt(drop, add):
        """Exact float64 change for this exchange; applies it if negative."""
        nonlocal attempted_exchanges
        attempted_exchanges += 1
        deltas = [a - d for a, d in zip(traj.aggregate(add), traj.aggregate(drop))]
        change = traj.exact_change(deltas, m)
        if change < 0:
            for u, du in zip(traj.u, deltas):
                u += du
            selected[drop] = False
            selected[add] = True
        return change

    for sweep in range(1, max_iters + 1):
        sweeps = sweep
        D_before = traj.D(m)
        h = traj.gradient(m)
        sel_idx = torch.where(selected)[0]
        uns_idx = torch.where(~selected)[0]
        drop_order = sel_idx[torch.argsort(h[sel_idx], descending=True)]
        add_order = uns_idx[torch.argsort(h[uns_idx])]

        # gamma_t: the best-ranked pair's linearised gain. Non-positive
        # means no exchange even looks like a descent direction, which is
        # Algorithm 3's optimality test on the linear term alone.
        gamma = float(h[drop_order[0]] - h[add_order[0]])
        best_lower_bound = max(
            best_lower_bound, D_before - r * max(gamma, 0.0))
        final_certificate = max(0.0, D_before - best_lower_bound)
        history[-1].update({"sweep": sweep, "gamma": gamma,
                            "lower_bound": best_lower_bound,
                            "certificate_estimate": final_certificate})
        if gamma <= 0:
            reason = 'gamma<=0'
            break

        accepted, change, last_rejected = False, None, None

        # Block exchange first: it is the only way to amortise the sweep
        # over more than one swap. Halve on failure — the ranking is a
        # linearisation, so a block can fail where its members succeed.
        block = min(swaps_per_sweep, m, n - m)
        while block > 1 and not accepted:
            change = attempt(drop_order[:block], add_order[:block])
            if change < 0:
                swaps += block
                accepted_updates += 1
                accepted = True
            else:
                last_rejected = (change if last_rejected is None
                                 else min(last_rejected, change))
                best_rejected = (change if best_rejected is None
                                 else min(best_rejected, change))
                block //= 2

        # Then single pairs in increasing linear term. Algorithm 3 stops
        # at the first rejection (candidates=1); c > 1 keeps looking,
        # since a rejected top pair does not mean every pair is bad.
        if not accepted:
            pairs = common.ranked_pairs(
                drop_order[:cand].cpu().numpy(), h[drop_order[:cand]].cpu().numpy(),
                add_order[:cand].cpu().numpy(), h[add_order[:cand]].cpu().numpy())
            for c, (_, i, j) in enumerate(pairs):
                if c >= cand:
                    break
                change = attempt(torch.tensor([i], device=h.device),
                                 torch.tensor([j], device=h.device))
                if change < 0:
                    swaps += 1
                    accepted_updates += 1
                    accepted = True
                    break
                last_rejected = (change if last_rejected is None
                                 else min(last_rejected, change))
                best_rejected = (change if best_rejected is None
                                 else min(best_rejected, change))

        last_change = change
        if not accepted:
            reason = (f'top-{cand} candidates all rejected' if cand > 1
                      else 'top-ranked candidate rejected (Algorithm 3 stopping rule)')
            break

        if refresh_every and accepted_updates % refresh_every == 0:
            traj.set_subset(torch.where(selected)[0])

        D_after = traj.D(m)
        final_certificate = max(0.0, D_after - best_lower_bound)
        history.append({"iteration": accepted_updates, "sweep": sweep,
                        "D": D_after, "swaps": swaps, "gamma": gamma,
                        "change": change, "lower_bound": best_lower_bound,
                        "certificate_estimate": final_certificate})
        if logger and (sweep % 10 == 0 or sweep == 1):
            logger.info(f"sweep {sweep}: D={D_after:.6e}  swaps={swaps}  "
                        f"gamma={gamma:.3e}  change={change:.3e}")

    index = torch.where(selected)[0]
    # u_I drifts as it accumulates differences; settle the reported value.
    traj.set_subset(index)
    final_D = traj.D(m)
    if math.isfinite(best_lower_bound):
        final_certificate = max(0.0, final_D - best_lower_bound)
    final_state_gradient_evaluated = reason != 'max_iters'
    terminal_gamma = gamma if final_state_gradient_evaluated else None
    proj_dim = int(getattr(traj, "proj_dim", 0) or 0)
    diag = {"algorithm": 3, "rule": "ranked_full_network", "reason": reason,
            "sweeps": sweeps, "attempted_exchanges": attempted_exchanges,
            "accepted_updates": accepted_updates, "swaps": swaps,
            "final_gamma": terminal_gamma, "last_evaluated_gamma": gamma,
            "final_state_gradient_evaluated": final_state_gradient_evaluated,
            "final_change": last_change,
            "last_rejected_change": last_rejected,
            "best_rejected_change": best_rejected, "D": final_D,
            "lower_bound_estimate": (best_lower_bound
                                     if math.isfinite(best_lower_bound) else None),
            "certificate_estimate": final_certificate,
            "certificate_valid": False,
            "certificate_note": (
                "Gradient ranks use float32 model JVPs; the bound is diagnostic."
                + (" D is the count-sketch objective, exact in that space."
                   if proj_dim else "")),
            "algorithm_3_exact": swaps_per_sweep == 1 and candidates == 1,
            "decision_dtype": "float64", "centered_acceptance": True,
            "proj_dim": proj_dim,
            "proj_seed": getattr(traj, "proj_seed", None),
            "projection": "count_sketch" if proj_dim else "none"}
    history.append({"iteration": accepted_updates, "final": True, **diag})
    if logger:
        gamma_text = "not evaluated" if terminal_gamma is None else f"{terminal_gamma:.3e}"
        logger.info(f"stopped after {sweeps} sweeps: {reason}. final gamma={gamma_text}, "
                    f"best rejected exact change="
                    + ("n/a" if best_rejected is None else f"{best_rejected:.3e}"))
        if reason == 'max_iters':
            logger.warning("--max_iters reached, so the search had not settled; "
                           "the subset is the best found, not a local optimum.")
    return index.cpu(), history, diag


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _retrain_only(cfg, trainset, device, args, base_dir, method, budget, K):
    """Recover the retraining stage of a run interrupted after selection.

    A sweep task can die (wall-clock kill, node failure, scratch outage)
    after common.finish has saved *_indices.pt but before *_metrics.json
    exists. retrain_on_subset resets the RNG immediately before model
    construction and uses a dedicated loader generator, so retraining from
    the saved indices reproduces the accuracy the interrupted run would
    have reported; only the selection stage is redundant work, and the
    recovered artifacts must land under the interrupted run's cfg hash.
    """
    pattern = os.path.join(base_dir, f"{method}_{budget}_K{K}_cfg*_indices.pt")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"--retrain_only found no saved selection matching {pattern}. The "
            "selection never finished; rerun without --retrain_only instead.")
    if len(matches) > 1:
        raise RuntimeError(
            "--retrain_only is ambiguous: several runs of this cell have saved "
            f"selections: {matches}. Resolve by hand.")
    prefix = matches[0][:-len('_indices.pt')]
    paths = dict(base_dir=base_dir,
                 model=f"{prefix}_model.pth",
                 indices=f"{prefix}_indices.pt",
                 metrics=f"{prefix}_metrics.json",
                 timing=f"{prefix}_timing.json",
                 objective=f"{prefix}_objective.json",
                 run=f"{prefix}_run.json")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(f"{prefix}_training.log"),
                  logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"--retrain_only: reusing saved selection {matches[0]}")

    if os.path.exists(paths['metrics']):
        logger.info(f"Retraining already complete: {paths['metrics']}")
        return

    recorded = {}
    if os.path.exists(paths['run']):
        with open(paths['run']) as fh:
            recorded = json.load(fh)
        if recorded.get('status') == 'complete':
            raise RuntimeError(
                f"{paths['run']} reports status 'complete' but {paths['metrics']} "
                "is missing; investigate before retraining.")
        arguments = (recorded.get('config') or {}).get('arguments') or {}

        def _same(key, recorded_value):
            current = getattr(args, key, None)
            if key == 'init_ckpt':
                current = os.path.realpath(current) if current else None
            return recorded_value == current
        drift = {key: {'interrupted_run': recorded_value,
                       'this command': getattr(args, key, None)}
                 for key, recorded_value in arguments.items()
                 if key in ('seed', 'batch_size', 'lr', 'retrain_optimizer',
                            'init_ckpt') and not _same(key, recorded_value)}
        if drift:
            raise RuntimeError(
                "--retrain_only recipe differs from the interrupted run "
                f"(interrupted vs this command): {drift}. Rerun must reproduce "
                "the original retraining arguments exactly.")
    else:
        logger.warning(f"No {paths['run']} to check the original recipe "
                       "against; retraining with the arguments given.")

    selected = common.torch_load(paths['indices'], map_location='cpu')
    if len(selected) != budget:
        raise RuntimeError(
            f"{paths['indices']} holds {len(selected)} indices, "
            f"expected budget={budget}")
    common.finish(cfg, trainset, selected, device, args, logger, paths)

    artifacts = {'indices': paths['indices'], 'objective': paths['objective'],
                 'timing': paths['timing'],
                 'model': paths['model'], 'metrics': paths['metrics']}
    common.atomic_json_dump(
        {'schema_version': recorded.get('schema_version', 1),
         'run_id': recorded.get('run_id'), 'status': 'complete',
         'config': recorded.get('config'),
         'retrained_from_saved_selection': True,
         'artifacts': artifacts}, paths['run'])


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fraction',       default=0.3,         type=float)
    p.add_argument('--seed', '-seed',  default=42,          type=int)
    p.add_argument('--checkpoint_dir', required=True,       type=str)
    p.add_argument('--allow_manifest_mismatch', action='store_true',
                   help='Use a reference trajectory from another split. The '
                        'manuscript scores every selection seed against the '
                        'seed-42 trajectory (Appendix G.1). Without this flag '
                        'the trajectory seed, pool, dataset, and architecture '
                        'must match this run.')
    p.add_argument('--dataset',        default='cifar100',  choices=list(common.DATASETS))
    p.add_argument('--architecture',   default=None, choices=list(common.ARCHITECTURES),
                   help='Network architecture. Defaults to resnet18, or lenet for MNIST.')
    p.add_argument('--data_dir',       required=True,       type=str)
    p.add_argument('--num_checkpoints', default=10,         type=int,
                   help='0 = every checkpoint found.')
    p.add_argument('--checkpoint_selection', default='spread', choices=['spread', 'prefix'],
                   help="'spread' spaces the K checkpoints over the whole run.")
    p.add_argument('--checkpoint_weighting', default='point',
                   choices=list(ref.CHECKPOINT_WEIGHTINGS),
                   help="'point' uses the saved checkpoint learning rate eta_k, matching "
                        "D_eta and D_Adam directly. 'window' integrates over the "
                        "optimizer steps represented by each checkpoint.")
    p.add_argument('--checkpoint_epochs', default=None, type=int, nargs='+',
                   help='Explicit checkpoint epochs; overrides the two options above.')
    p.add_argument('--optimizer',      default='uniform', choices=list(ref.OPTIMIZERS),
                   help='Objective weighting (w_k, M_k). See trajectory.py. '
                        'NOTE the default optimises neither Adam weights nor its '
                        'preconditioner; it is kept only to reproduce earlier runs.')
    p.add_argument('--metric',         default='auto', choices=list(ref.METRICS),
                   help='Override M_k independently of the step weight, allowing '
                        '--optimizer uniform --metric adam to isolate Adam coordinate '
                        'scaling from learning-rate weighting.')
    p.add_argument('--momentum_response', default='persistent',
                   choices=list(ref.MOMENTUM_RESPONSES),
                   help="a_s in w_k. 'persistent' (a_s = 1) is the exact response to a "
                        'discrepancy that lasts, which is what a subset produces; '
                        "'isolated' is (1-beta1)/(1-beta1^s).")
    p.add_argument('--ref_lr',         default=0.01,        type=float,
                   help='Base LR of the reference run, for rebuilding eta_s on a legacy dir.')
    p.add_argument('--adam_eps',       default=1e-8,        type=float)
    p.add_argument('--beta1',          default=0.9,         type=float)
    p.add_argument('--beta2',          default=0.999,       type=float)
    p.add_argument('--max_iters',      default=200,         type=int,
                   help='Maximum sweeps. Each sweep is one forward pass over the pool '
                        'per checkpoint (~1760 batches at CIFAR-100 scale with K=10), '
                        'so this is the wall-clock knob — and 200 may well stop before '
                        'the search settles. The stopping reason is logged and saved.')
    p.add_argument('--swaps_per_sweep', default=1,          type=int,
                   help='Exchanges attempted per gradient sweep. 1 is Algorithm 3 '
                        'exactly; larger values amortise the sweep and back off '
                        'automatically when the block fails its exact test.')
    p.add_argument('--candidates',     default=1,           type=int,
                   help='Single pairs tested per sweep before declaring convergence. '
                        "1 is Algorithm 3's stopping rule, which gives up as soon as its "
                        'top-ranked pair is rejected even though a lower-ranked one may '
                        'still improve D.')
    p.add_argument('--refresh_every',  default=0,           type=int,
                   help='Recompute u_I from scratch every N accepted sweeps (0 = never).')
    p.add_argument('--batch_size',     default=128,         type=int)
    p.add_argument('--score_batch_size', default=256,       type=int,
                   help='Batch size for the JVP and aggregate-gradient sweeps.')
    p.add_argument('--lr',             default=0.01,        type=float,
                   help='Learning rate for retraining')
    p.add_argument('--retrain_optimizer', default=None, choices=['sgd', 'adam', 'adamw'],
                   help='Default: follow --optimizer.')
    p.add_argument('--init_ckpt',      default=None,        type=str,
                   help='Shared retraining initialisation. Created on first use, then '
                        'reused, so two selectors are compared from the same weights.')
    p.add_argument('--num_workers',    default=4,           type=int)
    p.add_argument('--download',       action='store_true', default=False)
    p.add_argument('--scoring_transform', default='test', choices=['train', 'test'],
                   help="View the pool is scored through. Default 'test' (clean) so that "
                        'this and the last-layer selector score identical inputs.')
    p.add_argument('--selection_only', action='store_true', default=False)
    p.add_argument('--retrain_only', action='store_true', default=False,
                   help='Skip selection entirely and only run the retraining stage, '
                        'reusing the *_indices.pt already saved in the run directory '
                        '(recovery for runs interrupted after saving their selection). '
                        'Excluded from the run fingerprint so the recovered '
                        'artifacts keep the interrupted run\'s cfg hash.')
    p.add_argument('--proj_dim',       default=0,           type=int,
                   help='Count-sketch dimension for the full-network aggregates. '
                        '0 keeps the exact R^d vectors. 4096 is a practical size '
                        'for ResNet-18 with every CIFAR checkpoint: gbar and u_I '
                        'live in R^p, the JVP tangent is the sketch adjoint, and '
                        'acceptance is exact for the sketched objective.')
    p.add_argument('--proj_seed',      default=0,           type=int,
                   help='Seed of the count sketch. Independent of --seed so the '
                        'embedding is shared across data seeds.')
    p.add_argument('--init_indices',   default=None,        type=str,
                   help='Pool-relative indices to start from, as saved in '
                        '*_indices.pt. Length must equal the budget.')
    p.add_argument('--init_method',    default=None,        type=str,
                   help='Baseline that produced --init_indices. Required with it, '
                        'and written into the filename as _from_<method> so the '
                        'warm start does not share a cold-start K cell.')
    args = p.parse_args()
    if args.retrain_only and args.selection_only:
        p.error('--retrain_only reruns the retraining stage; combining it with '
                '--selection_only (which skips retraining) is contradictory.')

    common.set_seed(args.seed)
    use_metric = ref.resolve_metric(args.optimizer, args.metric)
    if args.retrain_optimizer is None:
        args.retrain_optimizer = (
            'adam' if use_metric or args.optimizer in ('adam', 'adam_update') else 'sgd')

    cfg         = common.get_dataset_config(
        args.dataset, args.data_dir, download=args.download,
        architecture=args.architecture)
    NUM_CLASSES = cfg['num_classes']
    NUM_EPOCHS  = cfg['num_epochs']
    device      = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainset = common.make_train_set(cfg)
    n_train  = len(trainset)
    budget  = int(args.fraction * n_train)
    if not 1 <= budget < n_train:
        raise ValueError(
            f"--fraction={args.fraction!r} gives budget={budget}; require "
            f"1 <= budget < n_train={n_train}."
        )
    pool_indices_sha256 = common.ordered_indices_fingerprint(trainset.indices)

    init_index = None
    init_indices_sha256 = None
    if args.init_indices or args.init_method:
        if not args.init_indices or not args.init_method:
            raise ValueError("--init_indices and --init_method must be set together")
        if not args.init_method.isidentifier():
            raise ValueError(
                "--init_method must be a single filename token, "
                f"got {args.init_method!r}")
        if not os.path.isfile(args.init_indices):
            raise ValueError(f"Initial indices not found: {args.init_indices}")
        loaded = common.torch_load(args.init_indices, map_location='cpu')
        init_index = _starting_index(n_train, budget, None, loaded, 'cpu')
        init_indices_sha256 = common.ordered_indices_fingerprint(init_index.tolist())

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
        expected_pool_indices_sha256=pool_indices_sha256,
        allow_manifest_mismatch=args.allow_manifest_mismatch)
    K = len(checkpoints)

    scratch_path = default_results_root()
    variant = storage_variant(args.optimizer, args.retrain_optimizer)
    base_dir = os.path.join(
        tree(scratch_path, 'gradmatch_swap', variant),
        args.dataset, cfg['architecture'], str(budget), f"seed_{args.seed}",
    )
    os.makedirs(base_dir, exist_ok=True)

    method = f"kl_faithful_full_{cfg['architecture']}" + ref.objective_tag(
        args.optimizer, args.metric, args.checkpoint_weighting,
        args.momentum_response)
    if args.scoring_transform != 'test':
        method += f'_{args.scoring_transform}view'
    if args.proj_dim:
        method += f'_jl{args.proj_dim}'
    if args.init_method:
        method += f'_from_{args.init_method}'
    if args.retrain_only:
        _retrain_only(cfg, trainset, device, args, base_dir, method, budget, K)
        return
    if manifest is not None:
        trajectory_identity = {
            'manifest_run_id': manifest.get('run_id'),
            'manifest_fingerprint': common.config_fingerprint(manifest),
        }
    else:
        trajectory_identity = {
            'legacy_directory': os.path.realpath(args.checkpoint_dir),
            'files': [
                {'path': os.path.realpath(c.path),
                 'size': os.stat(c.path).st_size,
                 'mtime_ns': os.stat(c.path).st_mtime_ns}
                for c in checkpoints
            ],
        }
    run_config = {
        'schema_version': 2,
        'selector': 'kl_full_network',
        # --retrain_only is a recovery-mode flag, not part of the experiment:
        # it must not shift the fingerprint, or recovered artifacts would miss
        # the interrupted run's cfg hash.
        'arguments': {**{key: value for key, value in vars(args).items()
                         if key != 'retrain_only'},
                      'checkpoint_dir': os.path.realpath(args.checkpoint_dir),
                      'data_dir': os.path.realpath(args.data_dir),
                      'init_ckpt': (os.path.realpath(args.init_ckpt)
                                    if args.init_ckpt else None),
                      'init_indices': (os.path.realpath(args.init_indices)
                                       if args.init_indices else None)},
        'resolved': {
            'dataset': args.dataset, 'architecture': cfg['architecture'],
            'n_train': n_train, 'budget': budget,
            'pool_indices_sha256': pool_indices_sha256,
            'init_method': args.init_method,
            'init_indices_sha256': init_indices_sha256,
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
    file_prefix = f"{method}_{budget}_K{K}_cfg{run_id}"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(os.path.join(base_dir, f"{file_prefix}_training.log")),
                  logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Dataset={args.dataset}  architecture={cfg['architecture']}  "
                f"budget={budget}/{n_train}  seed={args.seed}  "
                f"device={device}")
    logger.info(f"Immutable run fingerprint: {run_id}")
    logger.info(f"Trajectory: {ckpt_description}"
                + ("" if manifest else "  (legacy dir, eta_s reconstructed from the "
                                        "cosine schedule)"))
    logger.info("\n" + ref.describe(checkpoints))
    logger.info(f"metric={'D_k^2' if use_metric else 'I'}  "
                f"scoring_transform={args.scoring_transform}  "
                f"retrain_optimizer={args.retrain_optimizer}  "
                f"init_ckpt={args.init_ckpt}  "
                f"init_method={args.init_method}")
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

    # Same keys as common.get_save_paths, so common.finish can be reused,
    # but laid out like selection.kl's so the two Algorithm 3
    # variants land side by side and analysis.comparison
    # discovers them the same way.
    paths = dict(base_dir=base_dir,
                 model=os.path.join(base_dir, f"{file_prefix}_model.pth"),
                 indices=os.path.join(base_dir, f"{file_prefix}_indices.pt"),
                 metrics=os.path.join(base_dir, f"{file_prefix}_metrics.json"),
                 timing=os.path.join(base_dir, f"{file_prefix}_timing.json"),
                 objective=os.path.join(base_dir, f"{file_prefix}_objective.json"),
                 run=os.path.join(base_dir, f"{file_prefix}_run.json"))
    if args.selection_only:
        stale = [path for path in (paths['model'], paths['metrics'])
                 if os.path.exists(path)]
        if stale:
            raise FileExistsError(
                "Refusing --selection_only because retraining artifacts already exist "
                f"for this exact run: {stale}. Remove them explicitly or change the "
                "configuration.")
    common.atomic_json_dump(
        {'schema_version': 1, 'run_id': run_id, 'status': 'configured',
         'config': run_config}, paths['run'])

    scoring_set = common.scoring_subset(cfg, trainset, args.scoring_transform)

    model = common.build_model(
        cfg, NUM_CLASSES, device, architecture=cfg['architecture'])
    uses_batchnorm = any(
        isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        for module in model.modules())
    if uses_batchnorm:
        logger.warning(
            "The model contains BatchNorm. Selection scores per-example gradients "
            "with frozen running statistics (model.eval()); this is a reproducible "
            "surrogate, not the train-mode BatchNorm gradient field that produced "
            "the reference trajectory.")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Every sweep touches the whole pool many times over, so it is
    # materialised on the device once rather than reloaded per iteration.
    X, Y = d_alpha.materialise(scoring_set, device, batch_size=args.batch_size,
                               num_workers=args.num_workers)
    if args.proj_dim:
        logger.info(f"Pool on device: {tuple(X.shape)}  ({X.numel() * 4 / 1e9:.2f} GB).  "
                    f"{n_params/1e6:.1f}M trainable parameters, count sketch "
                    f"p={args.proj_dim} (seed {args.proj_seed}); aggregates are "
                    f"{2 * K * args.proj_dim * 8 / 1e6:.1f} MB and checkpoint "
                    f"weights stay on CPU.")
    else:
        logger.info(f"Pool on device: {tuple(X.shape)}  ({X.numel() * 4 / 1e9:.2f} GB).  "
                    f"{n_params/1e6:.1f}M trainable parameters, "
                    f"{3 * K * n_params * 8 / 1e9:.2f} GB of float64 trajectory state.")

    timer = common.SelectionTimer(device)
    with timer.phase('scoring'):
        traj = Trajectory(model, checkpoints, use_metric, X, Y, device,
                          args.score_batch_size, adam_eps=args.adam_eps, logger=logger,
                          proj_dim=args.proj_dim, proj_seed=args.proj_seed)

    logger.info(f"Running gradient-ranked single-swap descent over all parameters "
                f"(swaps_per_sweep={args.swaps_per_sweep}, "
                f"candidates={args.candidates}) ...")
    with timer.phase('selection'):
        selected_indices, history, diag = gradient_ranked_single_swap(
            traj, budget, max_iters=args.max_iters,
            swaps_per_sweep=args.swaps_per_sweep, candidates=args.candidates,
            refresh_every=args.refresh_every, logger=logger, seed=args.seed,
            init_index=init_index)
    logger.info(f"{diag['sweeps']} sweeps, {diag['swaps']} accepted swaps, "
                f"{len(selected_indices)} samples selected. "
                f"D: {history[0]['D']:.6e} -> {diag['D']:.6e}")

    common.save_timing(paths['timing'], timer, logger, method=method,
                       run_id=run_id,
                       reference_epochs_used=max(c.epoch for c in checkpoints),
                       swap_iterations=diag['sweeps'], accepted_swaps=diag['swaps'],
                       accepted_updates=diag['accepted_updates'],
                       attempted_exchanges=diag['attempted_exchanges'],
                       stopping_reason=diag['reason'],
                       final_gamma=diag['final_gamma'],
                       final_change=diag['final_change'],
                       best_rejected_change=diag['best_rejected_change'],
                       final_D=diag['D'], metric_scale=traj.metric_scale,
                       lower_bound_estimate=diag['lower_bound_estimate'],
                       certificate_estimate=diag['certificate_estimate'],
                       certificate_valid=diag['certificate_valid'],
                       algorithm_3_exact=diag['algorithm_3_exact'],
                       swaps_per_sweep=args.swaps_per_sweep,
                       candidates=args.candidates,
                       objective_optimizer=args.optimizer,
                       objective_metric='D_k^2' if use_metric else 'I',
                       momentum_response=args.momentum_response,
                       scoring_transform=args.scoring_transform,
                       batchnorm_scoring_mode=('eval_frozen' if uses_batchnorm else 'none'),
                       train_mode_gradient_correspondence=not uses_batchnorm,
                       checkpoint_selection=args.checkpoint_selection,
                       checkpoint_weighting=args.checkpoint_weighting,
                       checkpoint_steps=[c.step for c in checkpoints],
                       checkpoint_weights=[c.weight for c in checkpoints],
                       num_checkpoints_used=K,
                       proj_dim=args.proj_dim,
                       proj_seed=args.proj_seed if args.proj_dim else None,
                       projection='count_sketch' if args.proj_dim else 'none')
    common.atomic_json_dump(history, paths['objective'])

    # The pool, the float64 trajectory state and K sets of weights are all
    # still resident; retraining needs none of them.
    del traj
    del X, Y
    if device == 'cuda':
        torch.cuda.empty_cache()

    common.finish(cfg, trainset, selected_indices, device, args, logger, paths)
    status = 'selection_complete' if args.selection_only else 'complete'
    artifacts = {'indices': paths['indices'], 'objective': paths['objective'],
                 'timing': paths['timing']}
    if not args.selection_only:
        artifacts.update({'model': paths['model'], 'metrics': paths['metrics']})
    common.atomic_json_dump(
        {'schema_version': 1, 'run_id': run_id, 'status': status,
         'config': run_config, 'artifacts': artifacts}, paths['run'])


if __name__ == '__main__':
    main()
