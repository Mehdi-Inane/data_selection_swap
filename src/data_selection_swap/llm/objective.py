"""The quadratic set objective shared by top-k, best-swap and ranked-swap.

With the trajectory kernel ``K`` (PSD) and the linear score ``c`` from
:mod:`targets`, the fixed-size subset objective is

    f(I) = (1/m^2) sum_{i,j in I} K_ij  -  (2/m) sum_{i in I} c_i,    |I| = m.

* For the faithful target this equals the manuscript discrepancy ``D(I)`` minus
  the constant ``sum_k w_k ||gbar_k||^2`` (identical argmin), so a swap descent on
  ``f`` *is* faithful selection.
* For the targeted score, ``top-k`` on ``c`` is LESS; ``f`` is LESS plus the
  ``-(1/m^2)``-weighted redundancy term that the swap optimizes.

Writing the objective this way means the solver never touches a target vector:
it only needs ``K``-aggregates and ``c``, which is what makes Setting A and
Setting B share one code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from .features import TrajectoryFeatures


@dataclass
class SwapObjective:
    """Fixed-size subset objective ``f(I)`` over a trajectory feature store."""

    features: TrajectoryFeatures
    c: torch.Tensor

    def __post_init__(self) -> None:
        self.c = self.c.to(torch.float64).reshape(-1)
        if self.c.numel() != self.features.n:
            raise ValueError("c must have one score per pool example")
        self._qdiag = self.features.qdiag()

    def value(self, index: torch.Tensor) -> float:
        """Exact ``f(I)`` in float64."""
        index = index.to(torch.long)
        m = index.numel()
        A = self.features.aggregate(index)
        sum_K = float(self.features.kernel_dot(A, index).sum())
        return sum_K / (m * m) - 2.0 * float(self.c.index_select(0, index).sum()) / m

    def swap_deltas(self, sel: torch.Tensor, A: List[torch.Tensor],
                    drop_idx: torch.Tensor, add_idx: torch.Tensor) -> torch.Tensor:
        """Exact ``f(I - {i} + {j}) - f(I)`` for every (drop i, add j) pair.

        ``sel`` and ``A`` describe the current subset; ``drop_idx`` must be a
        subset of ``sel`` (the "in" rows) and ``add_idx`` disjoint from it (the
        "out" candidates). Returns a ``[|drop|, |add|]`` float64 matrix.
        """
        m = sel.to(torch.long).numel()
        a = self.features.kernel_dot(A, drop_idx)          # sum over current I, for each drop i
        b = self.features.kernel_dot(A, add_idx)           # sum over current I, for each add j
        q_in = self._qdiag.index_select(0, drop_idx.to(torch.long))
        q_out = self._qdiag.index_select(0, add_idx.to(torch.long))
        cross = self.features.cross(drop_idx, add_idx)     # <Phi_i, Phi_j>
        c_in = self.c.index_select(0, drop_idx.to(torch.long))
        c_out = self.c.index_select(0, add_idx.to(torch.long))

        inv_m2 = 1.0 / (m * m)
        # f' - f = (1/m^2)(-2a_i + q_i + 2b_j - 2K_ij + q_j) - (2/m)(c_j - c_i)
        term = (inv_m2 * (-2.0 * a[:, None] + q_in[:, None]
                          + 2.0 * b[None, :] - 2.0 * cross + q_out[None, :])
                - (2.0 / m) * (c_out[None, :] - c_in[:, None]))
        return term


def select_topk(scores: torch.Tensor, m: int) -> torch.Tensor:
    """Independent top-``m`` selection — LESS's rule when used on the influence score."""
    scores = scores.to(torch.float64).reshape(-1)
    m = max(1, min(int(scores.numel()), int(m)))
    top = torch.topk(scores, m).indices.to(torch.long)
    return torch.sort(top).values
