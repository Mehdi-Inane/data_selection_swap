"""Diagnostics for the "set-level de-duplication beats LESS top-k" hypothesis.

None of these affect selection; they are the measurements the paper reports:

* ``less_influence`` — LESS's per-example score (the linear ``c`` of Setting B);
* ``mean_pairwise_cosine`` — within-subset gradient redundancy (the quantity the
  swap penalty is meant to lower);
* ``jaccard`` — overlap between two selected subsets (swap vs LESS);
* ``spearman`` — rank correlation between a selection score and a measured
  downstream quality (the residual-D-vs-accuracy analog of the CV experiment).
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch

from .features import TrajectoryFeatures
from .targets import targeted_score


def less_influence(features: TrajectoryFeatures, subtask_targets, aggregation="max") -> torch.Tensor:
    """LESS's per-example influence score (Setting B linear score)."""
    return targeted_score(features, subtask_targets, aggregation=aggregation)


def mean_pairwise_cosine(features: TrajectoryFeatures, index: torch.Tensor,
                         checkpoint: int = -1) -> float:
    """Mean off-diagonal cosine among the selected rows at one checkpoint.

    Rows are L2-normalized at construction (``normalize="l2"``), so a plain row
    inner product *is* cosine. Redundant subsets score near 1.
    """
    index = index.to(torch.long)
    m = index.numel()
    if m < 2:
        return float("nan")
    feats = features.features[checkpoint]
    block = feats.index_select(0, index)
    if features.metrics[checkpoint].values is not None:
        block = block * features.metrics[checkpoint].values.sqrt()
        block = block / block.norm(dim=1, keepdim=True).clamp_min(1e-12)
    gram = block @ block.t()
    off = gram.sum() - gram.diagonal().sum()
    return float(off / (m * (m - 1)))


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(int(i) for i in a), set(int(i) for i in b)
    union = sa | sb
    return float(len(sa & sb) / len(union)) if union else 1.0


def subset_length_stats(lengths: Sequence[float], index: Sequence[int]) -> Dict[str, float]:
    """Length distribution of a selected subset (LESS's short-sequence-bias check)."""
    vals = torch.tensor([float(lengths[int(i)]) for i in index], dtype=torch.float64)
    return {"mean": float(vals.mean()), "std": float(vals.std()),
            "min": float(vals.min()), "max": float(vals.max())}


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (ties averaged). For score-vs-quality analysis."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(list(x)), ranks(list(y))
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return float(cov / (vx * vy)) if vx > 0 and vy > 0 else float("nan")
