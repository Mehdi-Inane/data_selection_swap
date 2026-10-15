"""Exact all-parameter evaluation of the faithfulness objective D(alpha).

Why this exists
───────────────
``analysis.comparison`` already reports a D_alpha column, but it
is built from the last-layer Kronecker-factored kernel
K = (L L^T) ⊙ (Phi Phi^T), so it only sees the gradient w.r.t.
`linear.weight`. The manuscript's objective is over the gradient of every
trainable parameter; this module computes that one, exactly.

The identity that makes it cheap
────────────────────────────────
For a subset I of size m, write

    u_I^t  = sum_{i in I}    grad l(theta_t, Z_i)
    gbar_t = (1/n) sum_{i in pool} grad l(theta_t, Z_i)

Then

    D(alpha) = sum_t w_t || u_I^t / m  -  gbar_t ||^2_{M_t} .

Both terms are *aggregate* gradients, i.e. ordinary backward passes with
reduction='sum'. Evaluating D therefore needs no per-sample gradients, no
n x n Gram matrix, and no random projection — just T * (m/n)
epoch-equivalents of backward. At CIFAR-100 scale (n=50000, T=10,
m=15000) that is a few tens of seconds per subset, versus 2 TB per
checkpoint to store the gradients P would be built from.

Conventions
───────────
* Gradients are of the pure cross-entropy loss. Weight decay is optimizer
  state, not part of grad l, so no optimizer is involved here.
* Scoring runs under model.eval(), so BatchNorm uses running statistics
  and a sample's gradient does not depend on the rest of its batch.
* The pool is materialised once as a device tensor. That removes the
  dataloader from the inner loop and freezes the scoring view, so D is
  reproducible run to run even under the augmenting 'train' transform.
* Weights w_t are normalised to sum to 1, so D is a weighted *mean* over
  the observed checkpoints rather than a sum. This is what makes D
  comparable across different --num_checkpoints; without it the K-ablation
  measures its own normalisation.
* Aggregates accumulate in float64. v_t is a difference of two similar
  vectors, and the cancellation is where precision actually matters.

The weights w_t and the metric M
────────────────────────────────
Both come from ``trajectory.resolve``, which is shared with the two
selection scripts so that what is optimised and what is scored cannot
drift apart. `metrics` is an optional per-checkpoint flat tensor laid out
like flat_grads() below, or None for Euclidean (M = I).
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import common
from . import trajectory as ref


# ─────────────────────────────────────────────────────────────────────────────
# Parameter bookkeeping
#   One fixed ordering of the trainable parameters defines the layout of
#   every flat vector here (aggregate gradients and metrics alike). BN
#   running statistics are buffers, not parameters, so they are correctly
#   absent: they are not trained by gradient descent.
# ─────────────────────────────────────────────────────────────────────────────
def trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def flat_dim(params) -> int:
    return sum(p.numel() for p in params)


def flat_grads(params) -> torch.Tensor:
    return torch.cat([p.grad.reshape(-1) for p in params])


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate gradients
# ─────────────────────────────────────────────────────────────────────────────
def flat_grad_sum(model, X, Y, index, batch_size, params=None):
    """
    sum_{i in index} grad l(theta, Z_i), as a float64 flat vector, plus the
    number of samples summed. reduction='sum' means no per-batch averaging
    to undo, and float64 accumulation keeps the sum exact enough that the
    u_I/m - gbar cancellation downstream is meaningful.
    """
    params = trainable_params(model) if params is None else params
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')

    total = torch.zeros(flat_dim(params), dtype=torch.float64, device=X.device)
    for start in range(0, index.numel(), batch_size):
        sel = index[start:start + batch_size]
        model.zero_grad(set_to_none=True)
        criterion(model(X[sel]), Y[sel]).backward()
        total += flat_grads(params).double()
    model.zero_grad(set_to_none=True)
    return total, int(index.numel())


def weighted_sq_norm(v: torch.Tensor, metric=None) -> float:
    sq = v * v
    return float((sq if metric is None else sq * metric).sum().item())


# ─────────────────────────────────────────────────────────────────────────────
# Materialising the scoring pool
#   Pulled onto the device once and reused. Fixing the view here is what
#   makes D reproducible: under scoring_transform='train' the augmentation
#   is drawn once and then shared by every checkpoint and every subset,
#   instead of being redrawn on each pass.
# ─────────────────────────────────────────────────────────────────────────────
def materialise(pool, device, batch_size=256, num_workers=4):
    n = len(pool)
    sample, _ = pool[0]
    nbytes = n * sample.numel() * 4
    if nbytes > 8e9:
        raise MemoryError(
            f"Materialising the scoring pool needs {nbytes / 1e9:.1f} GB "
            f"({n} x {tuple(sample.shape)}). That is fine for CIFAR-100 (~0.55 GB) "
            f"but not for ImageNet, which needs a streaming variant."
        )

    loader = DataLoader(pool, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs).to(device), torch.cat(ys).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Scorer
# ─────────────────────────────────────────────────────────────────────────────
class TrajectoryScorer:
    """
    Scores many subsets against one reference trajectory.

    The loop is checkpoint-outer / subset-inner on purpose: each checkpoint
    is loaded once and its gbar_t computed once, then shared by every
    subset. Only one aggregate gradient is resident at a time, so memory is
    O(d) regardless of how many subsets are scored.
    """

    def __init__(self, cfg, trainset, ckpt_paths, weights, device,
                 scoring_transform='test', batch_size=256, num_workers=4,
                 metrics=None, gbar_cache_dir=None, cache_tag=None, logger=None):
        if len(weights) != len(ckpt_paths):
            raise ValueError("weights and ckpt_paths must have the same length")

        self.cfg = cfg
        self.ckpt_paths = list(ckpt_paths)
        self.weights = list(weights)
        self.device = device
        self.batch_size = batch_size
        self.metrics = list(metrics) if metrics is not None else [None] * len(self.ckpt_paths)
        self.gbar_cache_dir = gbar_cache_dir
        self.cache_tag = cache_tag
        self.logger = logger
        ckpt_identity = []
        for path in self.ckpt_paths:
            stat = os.stat(path)
            ckpt_identity.append({
                'path': os.path.realpath(path),
                'size': stat.st_size,
                'mtime_ns': stat.st_mtime_ns,
            })
        # The cached gbar_t is a mean over the pool, so the pool is part of the
        # key: a cache written for an old 90% split must never be reused for
        # the full training set, even if a checkpoint path were recycled.
        pool_indices = getattr(trainset, 'indices', None)
        pool_fingerprint = (common.ordered_indices_fingerprint(pool_indices)
                            if pool_indices is not None else None)
        self.trajectory_id = common.config_fingerprint({
            'pool_indices_sha256': pool_fingerprint,
            'checkpoints': ckpt_identity,
            'weights': [float(w) for w in self.weights],
            'architecture': cfg.get('architecture'),
            'scoring_transform': scoring_transform,
        })

        pool = common.scoring_subset(cfg, trainset, scoring_transform)
        self.X, self.Y = materialise(pool, device, batch_size, num_workers)
        self.n = int(self.Y.numel())
        self.pool_index = torch.arange(self.n, device=self.device)

        self.model = common.build_model(cfg, cfg['num_classes'], device)
        self.params = trainable_params(self.model)
        self.d = flat_dim(self.params)

        if gbar_cache_dir:
            os.makedirs(gbar_cache_dir, exist_ok=True)

    def _load(self, ckpt_path):
        self.model.load_state_dict(ref.load_weights(ckpt_path))
        return self.model

    def _gbar(self, ckpt_path):
        cache = None
        if self.gbar_cache_dir and self.cache_tag:
            name = os.path.basename(ckpt_path).replace('.pth', '')
            cache = os.path.join(
                self.gbar_cache_dir,
                f"gbar_{self.cache_tag}_{self.trajectory_id}_{name}.pt")
            if os.path.exists(cache):
                cached = torch.load(cache, map_location=self.device)
                if cached.numel() != self.d:
                    raise ValueError(
                        f"Cached gbar has {cached.numel()} coordinates, expected {self.d}: "
                        f"{cache}"
                    )
                return cached

        gsum, count = flat_grad_sum(self.model, self.X, self.Y, self.pool_index,
                                    self.batch_size, self.params)
        gbar = gsum / count
        if cache:
            tmp = f"{cache}.tmp.{os.getpid()}"
            torch.save(gbar.cpu(), tmp)
            os.replace(tmp, cache)
        return gbar.to(self.device)

    def score(self, subsets: dict) -> dict:
        """
        {key: LongTensor of pool positions} -> {key: D(alpha)}.

        Indices are positions into `trainset`, i.e. exactly what every
        train_*_selection.py script saves in its *_indices.pt.
        """
        totals = {key: 0.0 for key in subsets}
        prepared = {key: torch.as_tensor(idx, dtype=torch.long).to(self.device)
                    for key, idx in subsets.items()}

        for t, (w, ckpt) in enumerate(zip(self.weights, self.ckpt_paths)):
            self._load(ckpt)
            gbar = self._gbar(ckpt)
            for key, idx in prepared.items():
                u, m = flat_grad_sum(self.model, self.X, self.Y, idx,
                                     self.batch_size, self.params)
                totals[key] += w * weighted_sq_norm(u / m - gbar, self.metrics[t])
            if self.logger:
                self.logger.info(f"  checkpoint {t + 1}/{len(self.ckpt_paths)} "
                                 f"({os.path.basename(ckpt)}) scored "
                                 f"{len(prepared)} subsets")
        return totals
