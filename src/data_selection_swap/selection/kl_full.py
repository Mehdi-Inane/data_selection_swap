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

Second-order rule (--swap_rule second_order)
────────────────────────────────────────────
The rule is in ``selection.second_order``; its two engines are here.
Mode A (--proj_dim p > 0): ``build_cached_trajectory`` caches every sketched
per-example gradient once, and a ``CachedTrajectory`` sweep is a matvec with
no network pass. Mode B (--proj_dim 0): ``Trajectory`` keeps the JVP sweep
above for the scores, and ``Trajectory.diagonal`` / ``Trajectory.swap_block``
add exact per-example curvature, from one pass over the pool and from the 2k
shortlisted images per sweep. --sketch_cache runs Algorithm 3 on Mode A.
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
from .. import subsets
from .. import trajectory as ref
from ..paths import default_results_root, storage_variant, tree
from . import second_order


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
        self._tables = {}

    def _tables_on(self, device):
        """(buckets, signs) on ``device``, copied there once."""
        key = str(device)
        if key not in self._tables:
            self._tables[key] = (self.buckets.to(device), self.signs.to(device))
        return self._tables[key]

    def accumulate_batch(self, out, flats, offset):
        """
        out[b] += R[:, offset:offset+c] @ flats[b] for every row of ``flats``.

        ``flats`` [B, c] holds coordinates offset..offset+c-1, e.g. one
        parameter tensor's per-example gradients, so a [B, d] matrix is never
        formed. Accumulates in ``out``'s dtype.
        """
        stop = offset + flats.shape[1]
        buckets, signs = self._tables_on(out.device)
        signed = flats.to(out.dtype) * signs[offset:stop].to(out.dtype)
        out.scatter_add_(1, buckets[offset:stop].expand(flats.shape[0], -1), signed)
        return out

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


def disable_inplace_activations(model):
    """Turn off ``inplace`` on activation modules before vmap(grad).

    Numerically identical; it only stops in-place writes into activations
    that per-example backward passes may still need.
    """
    for module in model.modules():
        if isinstance(getattr(module, 'inplace', None), bool):
            module.inplace = False
    return model


def per_sample_grads(model, params, buffers, X, Y):
    """{name: [B, *shape]} gradients of each example's own cross-entropy."""
    def loss_one(p, x, y):
        logits = torch.func.functional_call(model, {**p, **buffers}, (x.unsqueeze(0),))
        return F.cross_entropy(logits, y.unsqueeze(0))
    return torch.func.vmap(torch.func.grad(loss_one), in_dims=(None, 0, 0))(params, X, Y)


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
                 proj_dim=0, proj_seed=0, per_sample_batch_size=64):
        self.model = disable_inplace_activations(model.eval())
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.per_sample_batch_size = int(per_sample_batch_size)
        self.order = [n for n, p in model.named_parameters() if p.requires_grad]
        # Where each parameter sits in the flat gradient layout.
        self.slices, start = {}, 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.slices[name] = slice(start, start + p.numel())
                start += p.numel()
        self._q = None
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

    # ── Mode B of the second-order swap: exact in R^d, memory O(d + n) ──────

    def per_sample_blocks(self, params, buffers, rows):
        """(coordinates, per-example gradients [len(rows), numel]) per parameter.

        The gradient of one example at a time is never materialised as a
        single flat vector; callers work through the parameters in turn.
        """
        grads = per_sample_grads(self.model, params, buffers, self.X[rows], self.Y[rows])
        for name in self.order:
            yield self.slices[name], grads[name].flatten(start_dim=1)

    def diagonal(self):
        """
        q_r = C_rr = sum_k w_k ||g_r^k - gbar_k||^2_{M_k} for every example.

        One vmap(grad) pass over the pool per checkpoint, run once; only the n
        float64 scalars are kept. gbar_k is known beforehand, so each example
        is centred directly rather than through ||g||^2 - 2<g, gbar> + ||gbar||^2.
        """
        if self._q is None:
            if self.sketch is not None:
                raise RuntimeError("diagonal() is the exact Mode B path; with a count "
                                   "sketch use build_cached_trajectory (Mode A).")
            n = int(self.X.shape[0])
            q = torch.zeros(n, dtype=torch.float64, device=self.X.device)
            for (params, buffers), w, M, gbar in zip(
                    self.states, self.weights, self.metrics, self.gbar):
                params, buffers = self._stage(params, buffers)
                for start in range(0, n, self.per_sample_batch_size):
                    rows = slice(start, start + self.per_sample_batch_size)
                    for cols, g in self.per_sample_blocks(params, buffers, rows):
                        z = g.double() - gbar[cols]
                        sq = z * z if M is None else z * z * M[cols]
                        q[rows] += w * sq.sum(dim=1)
            self._q = q
        return self._q

    def swap_block(self, drop, add, m):
        """
        Delta[a, b] = D(I - drop[a] + add[b]) - D(I), from per-example
        gradients of these |drop| + |add| examples only, in float64.

        Their linear scores s_r = (2/m) sum_k w_k <zeta_r, M_k v_k>
        (v_k = u_I^k/m - gbar_k) and their centred Gram matrix give every
        term of Delta = d_i + a_j - 2 C_ij / m^2, recomputed here rather than
        taken from the float32 JVP scores that chose the shortlist.
        """
        if self.sketch is not None:
            raise RuntimeError("swap_block() is the exact Mode B path; with a count "
                               "sketch use build_cached_trajectory (Mode A).")
        index = torch.cat([drop, add])
        total, kd = index.numel(), drop.numel()
        gram = torch.zeros(total, total, dtype=torch.float64, device=self.X.device)
        s = torch.zeros(total, dtype=torch.float64, device=self.X.device)
        for (params, buffers), w, M, gbar, v in zip(
                self.states, self.weights, self.metrics, self.gbar, self.residuals(m)):
            params, buffers = self._stage(params, buffers)
            for cols, g in self.per_sample_blocks(params, buffers, index):
                z = g.double() - gbar[cols]
                mz = z if M is None else z * M[cols]
                gram += w * (mz @ z.T)
                s += (2.0 / m) * w * (mz @ v[cols])
        q, inv = torch.diagonal(gram), 1.0 / m ** 2
        return ((q[:kd] * inv - s[:kd])[:, None] + (q[kd:] * inv + s[kd:])[None, :]
                - 2.0 * inv * gram[:kd, kd:])


class CachedTrajectory:
    """
    Mode A of the second-order swap: every z_r = (sqrt(w_k) R P (g_r^k -
    gbar_k))_k cached once as a row of Z [n, K p], so that

        D(I) = || sum_{i in I} z_i ||^2 / m^2

    and every swap quantity is a product with Z. No sweep makes a network
    pass. Z is stored in float32 (205 MB at n = 50000, p = 1024, K = 1) after
    centring in float64; every decision is computed in float64 from the
    stored rows, so it is exact for the objective those rows define.

    Implements the trajectory protocol of ``gradient_ranked_single_swap``
    and ``second_order.second_order_swap``.
    """

    def __init__(self, Z, proj_dim=0, proj_seed=None, chunk_rows=8192):
        self.Z = Z
        self.X = Z.new_empty((Z.shape[0], 0))  # the protocol reads n and the device
        self.u = None
        self.proj_dim, self.proj_seed = int(proj_dim), proj_seed
        self.chunk_rows = int(chunk_rows)
        self.q = torch.cat([(rows * rows).sum(dim=1) for rows in self._rows()])

    @classmethod
    def from_gradients(cls, gradients, weights, sketch=None, dtype=torch.float32):
        """Cache from explicit per-example gradients, one [n, d] matrix per checkpoint.

        Each matrix is centred over its rows in float64, sketched if given,
        scaled by sqrt(w), and only then cast to ``dtype``.
        """
        if len(gradients) != len(weights):
            raise ValueError("one weight per gradient matrix")
        blocks = []
        for G, w in zip(gradients, weights):
            G = torch.as_tensor(G).double()
            centred = G - G.mean(dim=0, keepdim=True)
            if sketch is not None:
                centred = sketch.apply_batch(centred)
            blocks.append(math.sqrt(float(w)) * centred)
        return cls(torch.cat(blocks, dim=1).to(dtype),
                   proj_dim=0 if sketch is None else sketch.proj_dim,
                   proj_seed=getattr(sketch, 'seed', None))

    def _rows(self, index=None):
        """Rows of Z (all, or ``index``) in float64, ``chunk_rows`` at a time."""
        if index is not None:
            index = torch.as_tensor(index, device=self.Z.device)
        count = self.Z.shape[0] if index is None else index.numel()
        for start in range(0, count, self.chunk_rows):
            block = slice(start, start + self.chunk_rows)
            yield (self.Z[block] if index is None else self.Z[index[block]]).double()

    def _sum(self, index):
        return sum(rows.sum(dim=0) for rows in self._rows(index))

    def set_subset(self, index):
        self.u = [self._sum(index)]

    def aggregate(self, index):
        return [self._sum(index)]

    def D(self, m):
        return float(self.u[0] @ self.u[0]) / m ** 2

    def exact_change(self, deltas, m):
        du = deltas[0]
        return float((2.0 * self.u[0] + du) @ du) / m ** 2

    def gradient(self, m):
        """s_r = (2/m^2) <z_r, u> for every example, in float64."""
        return torch.cat([rows @ self.u[0] for rows in self._rows()]) * (2.0 / m ** 2)

    def diagonal(self):
        return self.q

    def swap_block(self, drop, add, m):
        """Delta[a, b] = D(I - drop[a] + add[b]) - D(I), exact in float64."""
        Zd, Za = self.Z[drop].double(), self.Z[add].double()
        inv = 1.0 / m ** 2
        d = self.q[drop] * inv - 2.0 * inv * (Zd @ self.u[0])
        a = self.q[add] * inv + 2.0 * inv * (Za @ self.u[0])
        return d[:, None] + a[None, :] - 2.0 * inv * (Zd @ Za.T)


def build_cached_trajectory(traj, dtype=torch.float32, logger=None):
    """
    Mode A cache from a ``Trajectory``: one vmap(grad) pass over the pool per
    checkpoint.

    Each parameter's per-example gradients are multiplied by sqrt(M) (Adam
    metric only) and scattered into the sketch in float64; the sketched mean
    gradient is subtracted and the row scaled by sqrt(w_k), still in float64,
    and only then cast to ``dtype``. Without a sketch the rows are the exact
    gradients, which only fits small networks (the unit tests).
    """
    n, device = int(traj.X.shape[0]), traj.X.device
    sketch = traj.sketch
    width = sketch.proj_dim if sketch is not None else sum(
        s.stop - s.start for s in traj.slices.values())
    Z = torch.empty(n, len(traj) * width, dtype=dtype, device=device)
    for k, (checkpoint, (params, buffers), w, M, gbar) in enumerate(zip(
            traj.checkpoints, traj.states, traj.weights, traj.metrics, traj.gbar)):
        params, buffers = traj._stage(params, buffers)
        if traj.preconditioned:     # sketched Adam metric: gbar is R(sqrt(M) gbar)
            pre, center = traj._sqrt_scaled_metric(checkpoint).to(device), gbar
        elif M is not None:         # exact Adam metric
            pre = M.sqrt()
            center = pre * gbar
        else:
            pre, center = None, gbar
        columns = slice(k * width, (k + 1) * width)
        for start in range(0, n, traj.per_sample_batch_size):
            rows = slice(start, min(start + traj.per_sample_batch_size, n))
            acc = torch.zeros(rows.stop - rows.start, width, dtype=torch.float64,
                              device=device)
            for cols, g in traj.per_sample_blocks(params, buffers, rows):
                g = g.double() if pre is None else g.double() * pre[cols]
                if sketch is None:
                    acc[:, cols] += g
                else:
                    sketch.accumulate_batch(acc, g, cols.start)
            Z[rows, columns] = (math.sqrt(w) * (acc - center)).to(dtype)
        if logger:
            logger.info(f"  cached per-example gradients at epoch {checkpoint.epoch}: "
                        f"{n} x {width} ({dtype}).")
    return CachedTrajectory(Z, proj_dim=traj.proj_dim, proj_seed=traj.proj_seed)


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 3
# ─────────────────────────────────────────────────────────────────────────────

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

    index = subsets.initial_subset(n, m, seed=seed, init_index=init_index).to(traj.X.device)
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
                exchanged = (drop_order[:block].tolist(), add_order[:block].tolist())
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
                    exchanged = ([i], [j])
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
                        "certificate_estimate": final_certificate,
                        "dropped": exchanged[0], "added": exchanged[1]})
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
                            'init_ckpt', 'retrain_schedule')
                 and not _same(key, recorded_value)}
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


SWAP_RULE_ARGUMENTS = ('swap_rule', 'shortlist_k', 'class_balanced', 'class_shortlist_k',
                       'accept_tol', 'sketch_cache', 'per_sample_batch_size')
# Outcome fields one swap rule reports and the other does not; copied into
# the timing record when present.
RULE_OUTCOMES = ('accepted_updates', 'attempted_exchanges', 'final_gamma', 'final_change',
                 'best_rejected_change', 'lower_bound_estimate', 'certificate_estimate',
                 'certificate_valid', 'algorithm_3_exact', 'one_swap_local_optimum',
                 'final_best_change', 'aggregate_drift_max_abs')


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
    common.add_retrain_schedule_argument(p)
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
    p.add_argument('--swap_rule',      default='ranked',    choices=['ranked', 'second_order'],
                   help="'ranked' is Algorithm 1: the top linear-ranked pair, stop at the "
                        "first rejection. 'second_order' shortlists the --shortlist_k "
                        'drops and adds with the smallest exact single-point change '
                        '(linear term plus diagonal curvature) and takes the best exact '
                        'exchange in that k x k block. With --proj_dim > 0 it runs on a '
                        'cached count sketch with no network pass during swaps (Mode A); '
                        'with --proj_dim 0 it is exact in R^d (Mode B).')
    p.add_argument('--shortlist_k',    default=32,          type=int,
                   help='second_order: drops and adds shortlisted per sweep.')
    p.add_argument('--class_balanced', action='store_true', default=False,
                   help='second_order: start from a draw stratified at '
                        'common.per_class_budget quotas and only exchange examples of '
                        'the same class, so every class keeps its quota.')
    p.add_argument('--class_shortlist_k', default=16,       type=int,
                   help='--class_balanced: per-class shortlist size.')
    p.add_argument('--accept_tol',     default=1e-12,       type=float,
                   help='second_order: accept an exchange only if its exact float64 '
                        'change is below -accept_tol.')
    p.add_argument('--sketch_cache',   action='store_true', default=False,
                   help='ranked rule with --proj_dim > 0: run Algorithm 1 on the Mode A '
                        'cache instead of a JVP sweep per iteration. (second_order always '
                        'uses the cache when --proj_dim > 0.)')
    p.add_argument('--per_sample_batch_size', default=64,   type=int,
                   help='Examples per vmap(grad) call when building the Mode A cache or '
                        'the Mode B curvature.')
    args = p.parse_args()
    if args.retrain_only and args.selection_only:
        p.error('--retrain_only reruns the retraining stage; combining it with '
                '--selection_only (which skips retraining) is contradictory.')
    if args.swap_rule == 'second_order':
        if (args.swaps_per_sweep != 1 or args.candidates != 1 or args.refresh_every
                or args.sketch_cache):
            p.error('--swaps_per_sweep, --candidates, --refresh_every and --sketch_cache '
                    'are options of the ranked rule.')
    elif args.class_balanced:
        p.error('--class_balanced requires --swap_rule second_order.')
    if args.sketch_cache and args.proj_dim <= 0:
        p.error('--sketch_cache caches count-sketched gradients; it needs --proj_dim > 0.')
    if (min(args.shortlist_k, args.class_shortlist_k, args.per_sample_batch_size) < 1
            or not args.accept_tol >= 0):
        p.error('--shortlist_k, --class_shortlist_k and --per_sample_batch_size must be '
                'positive and --accept_tol non-negative.')
    cached = args.proj_dim > 0 and (args.swap_rule == 'second_order' or args.sketch_cache)
    if cached:
        swap_engine = 'cached_sketch'
    elif args.swap_rule == 'second_order':
        swap_engine = 'exact_full_network'
    else:
        swap_engine = 'jvp_sketch' if args.proj_dim else 'jvp_exact'

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
        init_index = subsets.initial_subset(n_train, budget, init_index=loaded)
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
    if args.swap_rule == 'second_order':
        method += f'_so{args.shortlist_k}'
        if args.class_balanced:
            method += f'_cb{args.class_shortlist_k}'
    elif args.sketch_cache:
        method += '_cache'
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
        # the interrupted run's cfg hash. The swap-rule options are left out
        # of a plain Algorithm 1 run for the same reason: they postdate it, and
        # adding them would give every existing configuration a new hash, a
        # second cfg in its result cell, and an explicit failure in
        # analysis.comparison. --retrain_schedule is left out at its default
        # for the same reason.
        'arguments': {**{key: value for key, value in vars(args).items()
                         if key != 'retrain_only'
                         and not (key == 'retrain_schedule' and value == 'epochs')
                         and not (key in SWAP_RULE_ARGUMENTS
                                  and args.swap_rule == 'ranked'
                                  and not args.sketch_cache)},
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
                 run=os.path.join(base_dir, f"{file_prefix}_run.json"),
                 init_indices=os.path.join(base_dir, f"{file_prefix}_init_indices.pt"))
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

    labels = Y if args.class_balanced else None
    start = subsets.initial_subset(n_train, budget, seed=args.seed, init_index=init_index,
                                   labels=labels, num_classes=NUM_CLASSES)
    # With the dropped/added positions of each accepted update in the
    # objective history, this rebuilds the subset after any number of swaps
    # (subsets.replay_swaps).
    torch.save(start, paths['init_indices'])

    timer = common.SelectionTimer(device)
    with timer.phase('scoring'):
        traj = Trajectory(model, checkpoints, use_metric, X, Y, device,
                          args.score_batch_size, adam_eps=args.adam_eps, logger=logger,
                          proj_dim=args.proj_dim, proj_seed=args.proj_seed,
                          per_sample_batch_size=args.per_sample_batch_size)
        if cached:
            logger.info("Caching sketched per-example gradients (Mode A) ...")
            engine = build_cached_trajectory(traj, logger=logger)
        else:
            engine = traj
            if args.swap_rule == 'second_order':
                logger.info("Per-example curvature over the pool (Mode B) ...")
                engine.diagonal()

    # Model calls made by the swap search itself; Mode A makes none.
    forward_calls = [0]

    def count_forward(*_):
        forward_calls[0] += 1

    hook = model.register_forward_hook(count_forward)
    try:
        with timer.phase('selection'):
            if args.swap_rule == 'second_order':
                logger.info(f"Second-order k x k swap on the {swap_engine} engine "
                            f"(k={args.shortlist_k}"
                            + (f", class-balanced k_c={args.class_shortlist_k}"
                               if args.class_balanced else "") + ") ...")
                selected_indices, history, diag = second_order.second_order_swap(
                    engine, budget, k=args.shortlist_k, max_iters=args.max_iters,
                    accept_tol=args.accept_tol, labels=labels, num_classes=NUM_CLASSES,
                    class_k=args.class_shortlist_k, init_index=start, logger=logger)
            else:
                logger.info(f"Gradient-ranked single-swap descent on the {swap_engine} "
                            f"engine (swaps_per_sweep={args.swaps_per_sweep}, "
                            f"candidates={args.candidates}) ...")
                selected_indices, history, diag = gradient_ranked_single_swap(
                    engine, budget, max_iters=args.max_iters,
                    swaps_per_sweep=args.swaps_per_sweep, candidates=args.candidates,
                    refresh_every=args.refresh_every, logger=logger, seed=args.seed,
                    init_index=start)
    finally:
        hook.remove()
    initial_D, final_D = history[0]['D'], diag['D']
    overlap = len(set(start.tolist()) & set(selected_indices.tolist()))
    logger.info(f"{diag['sweeps']} sweeps, {diag['swaps']} accepted swaps, "
                f"{len(selected_indices)} samples selected. D: {initial_D:.6e} -> "
                f"{final_D:.6e} ({forward_calls[0]} model calls during the search)")

    common.save_timing(paths['timing'], timer, logger, checkpoint_dir=args.checkpoint_dir,
                       method=method, run_id=run_id,
                       reference_epochs_used=max(c.epoch for c in checkpoints),
                       swap_rule=args.swap_rule, swap_engine=swap_engine,
                       shortlist_k=(args.shortlist_k
                                    if args.swap_rule == 'second_order' else None),
                       class_balanced=args.class_balanced,
                       # Swap diagnostics under the implementation guide's names.
                       initial_G_T=initial_D, final_G_T=final_D,
                       relative_G_T_drop_pct=(100.0 * (initial_D - final_D) / initial_D
                                              if initial_D > 0 else 0.0),
                       accepted_swaps=diag['swaps'], sweeps_run=diag['sweeps'],
                       stopping_reason=diag['reason'],
                       jaccard_with_init=overlap / (2 * budget - overlap),
                       selection_forward_calls=forward_calls[0],
                       # Names the existing analysis reads.
                       swap_iterations=diag['sweeps'], final_D=final_D,
                       metric_scale=traj.metric_scale,
                       **{key: diag[key] for key in RULE_OUTCOMES if key in diag},
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

    # The pool, the float64 trajectory state, the cache and K sets of
    # weights are all still resident; retraining needs none of them.
    del engine, traj
    del X, Y
    if device == 'cuda':
        torch.cuda.empty_cache()

    common.finish(cfg, trainset, selected_indices, device, args, logger, paths)
    status = 'selection_complete' if args.selection_only else 'complete'
    artifacts = {'indices': paths['indices'], 'objective': paths['objective'],
                 'timing': paths['timing'], 'init_indices': paths['init_indices']}
    if not args.selection_only:
        artifacts.update({'model': paths['model'], 'metrics': paths['metrics']})
    common.atomic_json_dump(
        {'schema_version': 1, 'run_id': run_id, 'status': status,
         'config': run_config, 'artifacts': artifacts}, paths['run'])


if __name__ == '__main__':
    main()
