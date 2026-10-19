"""Build the per-example linear score ``c`` that every selection shares.

The swap solver only ever sees a kernel and this vector ``c`` (see
:mod:`objective`). Both settings differ *only* in how ``c`` is built:

Setting A — faithful (validation-free)
    ``c_x = sum_k w_k <Phi_k[x], gbar_k>_k`` with ``gbar_k`` the mean pooled
    gradient. With this ``c`` the swap objective equals the manuscript's
    trajectory discrepancy ``D(I)`` up to the additive constant
    ``sum_k w_k ||gbar_k||^2_k`` (proved in ``tests/test_objective.py``), so the
    selector optimizes exactly the same thing the CV code does.

Setting B — targeted (LESS-style)
    ``c_x = agg_j sum_k w_k cos(Phi_k[x], target_k^{(j)})`` where ``target_k^{(j)}``
    is the mean gradient of validation subtask ``j``. Selecting ``top-k`` on this
    ``c`` reproduces LESS (its Eq. 2 with ``max_j``); the swap then adds a
    gradient-redundancy penalty that LESS's independent scoring lacks.
"""

from __future__ import annotations

from typing import List, Sequence

import torch

from .features import TrajectoryFeatures


def faithful_score(features: TrajectoryFeatures) -> torch.Tensor:
    """Setting A linear score (pool-mean target)."""
    return features.linear_score(features.pool_mean_targets())


def targeted_score(
    features: TrajectoryFeatures,
    subtask_targets: Sequence[Sequence[torch.Tensor]],
    aggregation: str = "max",
) -> torch.Tensor:
    """Setting B linear score over one or more validation subtasks.

    Parameters
    ----------
    subtask_targets:
        ``subtask_targets[j][k]`` is the ``[d]`` mean validation gradient of
        subtask ``j`` at checkpoint ``k``. A single subtask is just a length-1
        outer list.
    aggregation:
        ``"max"`` (LESS's ``max_j``) or ``"mean"``.
    """
    if aggregation not in ("max", "mean"):
        raise ValueError(f"aggregation must be 'max' or 'mean', got {aggregation!r}")
    if len(subtask_targets) == 0:
        raise ValueError("subtask_targets must contain at least one subtask")

    per_subtask = torch.stack(
        [features.linear_score(targets) for targets in subtask_targets], dim=0)
    return per_subtask.max(dim=0).values if aggregation == "max" else per_subtask.mean(dim=0)


def build_score(
    features: TrajectoryFeatures,
    setting: str,
    subtask_targets: Sequence[Sequence[torch.Tensor]] | None = None,
    aggregation: str = "max",
) -> torch.Tensor:
    """Dispatch on ``setting``: ``'faithful'`` -> Setting A, ``'targeted'`` -> Setting B."""
    if setting == "faithful":
        return faithful_score(features)
    if setting == "targeted":
        if not subtask_targets:
            raise ValueError("targeted selection requires subtask_targets")
        return targeted_score(features, subtask_targets, aggregation=aggregation)
    raise ValueError(f"unknown setting {setting!r}")
