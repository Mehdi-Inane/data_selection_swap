"""Trajectory-weighted gradient-feature kernel primitives.

LESS hands us, for every candidate example and every warm-up checkpoint, a
low-dimensional projected Adam-LoRA gradient feature. Stacking these gives one
matrix ``Phi_k`` of shape ``[n, d]`` per checkpoint ``k``. The selector only ever
needs aggregate kernel quantities built from the trajectory-weighted feature
inner product

    <a, b>_k = <M_k a, b>            (M_k = diag(metric_k), metric_k >= 0)
    <a, b>   = sum_k w_k <a, b>_k     (w_k >= 0 checkpoint weights)

so it is never necessary to materialize the full ``n x n`` Gram. This module owns
those primitives; :mod:`swap` turns them into a selection.

All arithmetic is float64. Swap decisions are differences of two nearly equal
aggregate quantities, and the CV selector learned (see ``troubleshoot.pdf``
referenced in ``tests/test_swap_algorithms.py``) that computing them in float32
can flip an accept. Same lesson, same fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch


def _l2_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norms = x.norm(dim=1, keepdim=True).clamp_min(eps)
    return x / norms


@dataclass(frozen=True)
class Metric:
    """Per-checkpoint coordinate metric ``M_k`` (``None`` means identity)."""

    values: Optional[torch.Tensor]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """``M_k x``."""
        return x if self.values is None else x * self.values

    def sq_norm(self, x: torch.Tensor) -> torch.Tensor:
        """``||x||^2_{M_k} = <M_k x, x>`` for each row of ``x``."""
        if self.values is None:
            return (x * x).sum(dim=1)
        return (self.values * x * x).sum(dim=1)


class TrajectoryFeatures:
    """Weighted trajectory of per-example gradient features.

    Parameters
    ----------
    features:
        Sequence of ``[n, d]`` tensors, one per checkpoint. Rows are examples in
        a fixed pool order shared by every checkpoint.
    weights:
        Non-negative checkpoint weights ``w_k`` (LESS's average learning rate
        ``eta_bar_k``). Normalized to sum to one so the reported objective does
        not depend on how many checkpoints are used.
    metrics:
        Optional per-checkpoint ``[d]`` metric diagonals. ``None`` (default)
        gives the Euclidean kernel.
    normalize:
        ``"l2"`` row-normalizes every checkpoint feature matrix (LESS cosine);
        ``"none"`` keeps raw features. The metric is applied to the (possibly
        normalized) features.
    """

    def __init__(
        self,
        features: Sequence[torch.Tensor],
        weights: Optional[Sequence[float]] = None,
        metrics: Optional[Sequence[Optional[torch.Tensor]]] = None,
        normalize: str = "l2",
    ) -> None:
        if len(features) == 0:
            raise ValueError("features must contain at least one checkpoint")
        if normalize not in ("l2", "none"):
            raise ValueError(f"normalize must be 'l2' or 'none', got {normalize!r}")

        feats = [f.to(torch.float64) for f in features]
        n = feats[0].shape[0]
        d = feats[0].shape[1]
        for f in feats:
            if f.dim() != 2:
                raise ValueError(f"each checkpoint must be 2-D [n, d], got shape {tuple(f.shape)}")
            if f.shape[0] != n or f.shape[1] != d:
                raise ValueError(
                    f"all checkpoints must share shape [{n}, {d}], got {tuple(f.shape)}")

        if normalize == "l2":
            feats = [_l2_rows(f) for f in feats]

        if weights is None:
            weights = [1.0 / len(feats)] * len(feats)
        weights = [float(w) for w in weights]
        if len(weights) != len(feats):
            raise ValueError("weights must have one entry per checkpoint")
        if any(w < 0 for w in weights):
            raise ValueError("checkpoint weights must be non-negative")
        total = sum(weights)
        if total <= 0:
            raise ValueError("checkpoint weights must not all be zero")
        weights = [w / total for w in weights]

        if metrics is None:
            metrics = [None] * len(feats)
        metric_objs: List[Metric] = []
        for k, m in enumerate(metrics):
            if m is None:
                metric_objs.append(Metric(None))
            else:
                m = m.to(torch.float64).reshape(-1)
                if m.numel() != d:
                    raise ValueError(f"metric {k} has {m.numel()} entries, expected {d}")
                if (m < 0).any():
                    raise ValueError(f"metric {k} must be non-negative")
                metric_objs.append(Metric(m))

        self.features = feats
        self.weights = weights
        self.metrics = metric_objs
        self.normalize = normalize
        self.n = n
        self.d = d

    # -- aggregate kernel quantities --------------------------------------
    def aggregate(self, index: torch.Tensor) -> List[torch.Tensor]:
        """Return ``A_k = M_k sum_{y in index} Phi_k[y]`` for every checkpoint.

        ``A_k`` lets any ``<Phi_x, sum_{y in I} Phi_y>_k`` be a single dot product,
        which is exactly what a swap delta needs from the "current subset" side.
        """
        index = index.to(torch.long)
        return [self.metrics[k].apply(self.features[k].index_select(0, index).sum(dim=0))
                for k in range(len(self.features))]

    def shift(self, A: List[torch.Tensor], out_index: torch.Tensor,
              add_index: torch.Tensor) -> List[torch.Tensor]:
        """Update a running aggregate by removing ``out_index`` and adding ``add_index``."""
        out_index = out_index.to(torch.long)
        add_index = add_index.to(torch.long)
        return [A[k]
                - self.metrics[k].apply(self.features[k].index_select(0, out_index).sum(dim=0))
                + self.metrics[k].apply(self.features[k].index_select(0, add_index).sum(dim=0))
                for k in range(len(self.features))]

    def kernel_dot(self, A: List[torch.Tensor], rows: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``x |-> <Phi_x, A>`` for all rows (or the given row index).

        With ``A`` the aggregate over ``I``, this returns ``<Phi_x, sum_{y in I} Phi_y>``
        for each requested ``x``.
        """
        if rows is None:
            idx = torch.arange(self.n)
        else:
            idx = rows.to(torch.long)
        out = torch.zeros(idx.numel(), dtype=torch.float64)
        for k in range(len(self.features)):
            block = self.features[k].index_select(0, idx)
            out += self.weights[k] * (block @ A[k])
        return out

    def qdiag(self) -> torch.Tensor:
        """``q_x = <Phi_x, Phi_x> = sum_k w_k ||Phi_k[x]||^2_{M_k}`` for every ``x``."""
        q = torch.zeros(self.n, dtype=torch.float64)
        for k in range(len(self.features)):
            q += self.weights[k] * self.metrics[k].sq_norm(self.features[k])
        return q

    def cross(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """``[|left|, |right|]`` matrix of ``<Phi_left[a], Phi_right[b]>``."""
        left = left.to(torch.long)
        right = right.to(torch.long)
        out = torch.zeros(left.numel(), right.numel(), dtype=torch.float64)
        for k in range(len(self.features)):
            l = self.metrics[k].apply(self.features[k].index_select(0, left))
            r = self.features[k].index_select(0, right)
            out += self.weights[k] * (l @ r.t())
        return out

    # -- linear score from a target gradient ------------------------------
    def linear_score(self, targets: Sequence[torch.Tensor]) -> torch.Tensor:
        """``c_x = sum_k w_k <Phi_k[x], target_k>_k`` for a single target per checkpoint."""
        if len(targets) != len(self.features):
            raise ValueError("targets must have one vector per checkpoint")
        c = torch.zeros(self.n, dtype=torch.float64)
        for k, t in enumerate(targets):
            t = t.to(torch.float64).reshape(-1)
            if t.numel() != self.d:
                raise ValueError(f"target {k} has {t.numel()} entries, expected {self.d}")
            if self.normalize == "l2":
                t = t / t.norm().clamp_min(1e-12)
            c += self.weights[k] * (self.features[k] @ (self.metrics[k].apply(t)
                                                         if self.metrics[k].values is not None else t))
        return c

    def pool_mean_targets(self) -> List[torch.Tensor]:
        """The mean pooled gradient ``gbar_k`` per checkpoint (Setting A target)."""
        return [self.features[k].mean(dim=0) for k in range(len(self.features))]

    def D(self, index: torch.Tensor, targets: Sequence[torch.Tensor]) -> float:
        """Trajectory discrepancy ``sum_k w_k ||mean_I Phi_k - target_k||^2_k``."""
        index = index.to(torch.long)
        m = index.numel()
        total = 0.0
        for k, t in enumerate(targets):
            t = t.to(torch.float64).reshape(-1)
            if self.normalize == "l2":
                t = t / t.norm().clamp_min(1e-12)
            mean = self.features[k].index_select(0, index).mean(dim=0)
            resid = mean - t
            if self.metrics[k].values is None:
                total += self.weights[k] * float((resid * resid).sum())
            else:
                total += self.weights[k] * float((self.metrics[k].values * resid * resid).sum())
        return total
