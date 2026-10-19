"""Fixed-size subset search: LESS top-k, plus the swap descent that adds
gradient-redundancy awareness. ``select`` ties the pieces together.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .config import SelectionConfig
from .features import TrajectoryFeatures
from .objective import SwapObjective, select_topk
from .targets import build_score

SelectionResult = Tuple[torch.Tensor, List[Dict], Dict]


def _initial_subset(n: int, m: int, seed: int) -> torch.Tensor:
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    return torch.sort(order[:m]).values


def _complement(sel: torch.Tensor, n: int) -> torch.Tensor:
    mask = torch.ones(n, dtype=torch.bool)
    mask[sel.to(torch.long)] = False
    return torch.nonzero(mask, as_tuple=False).reshape(-1)


def _chunks(index: torch.Tensor, size: int):
    for start in range(0, index.numel(), size):
        yield index[start:start + size]


def best_swap(objective: SwapObjective, m: int, *, max_iters: int = 2000,
              block_size: int = 4096, tolerance: float = 1e-9, seed: int = 0,
              init: Optional[torch.Tensor] = None) -> SelectionResult:
    """Algorithm-2-style exact best single swap each iteration (certified
    one-swap local optimum). Exact but O(|I| x |outside|) per sweep."""
    n = objective.features.n
    sel = (init.to(torch.long) if init is not None else _initial_subset(n, m, seed))
    if sel.numel() != m:
        raise ValueError(f"init subset size {sel.numel()} != m {m}")

    A = objective.features.aggregate(sel)
    value = objective.value(sel)
    history = [{"iteration": 0, "value": value, "swaps": 0}]
    swaps = 0
    iteration = 0
    reason = "local_optimum"

    while iteration < max_iters:
        iteration += 1
        outside = _complement(sel, n)
        best_delta, best_out, best_add = float("inf"), -1, -1
        for block in _chunks(outside, block_size):
            deltas = objective.swap_deltas(sel, A, sel, block)  # rows are current sel
            flat = deltas.reshape(-1)
            j = int(torch.argmin(flat))
            delta = float(flat[j])
            if delta < best_delta:
                best_delta = delta
                best_out = int(sel.reshape(-1)[j // block.numel()])
                best_add = int(block[j % block.numel()])
        if best_delta >= -tolerance:
            reason = "local_optimum"
            break
        out_t = torch.tensor([best_out])
        add_t = torch.tensor([best_add])
        A = objective.features.shift(A, out_t, add_t)
        keep = sel != best_out
        sel = torch.cat([sel[keep], add_t])
        value = objective.value(sel)
        swaps += 1
        history.append({"iteration": iteration, "value": value, "swaps": swaps})
    if iteration >= max_iters and reason == "local_optimum":
        reason = "local_optimum_or_max_iters"

    diag = {"rule": "best", "swaps": swaps, "iterations": iteration,
            "value": value, "reason": reason,
            "one_swap_local_optimum": reason.startswith("local_optimum")}
    return torch.sort(sel).values, history, diag


def ranked_swap(objective: SwapObjective, m: int, *, max_iters: int = 2000,
                block_size: int = 4096, tolerance: float = 1e-9, seed: int = 0,
                candidates: int = 1, init: Optional[torch.Tensor] = None) -> SelectionResult:
    """Algorithm-3-style gradient-ranked single-swap descent: rank candidates by a
    linear marginal, test the top ``candidates`` pairs exactly, accept the first
    improvement, stop when none improves. Cheap and typically near-optimal."""
    n = objective.features.n
    sel = (init.to(torch.long) if init is not None else _initial_subset(n, m, seed))
    if sel.numel() != m:
        raise ValueError(f"init subset size {sel.numel()} != m {m}")

    A = objective.features.aggregate(sel)
    value = objective.value(sel)
    history = [{"iteration": 0, "value": value, "swaps": 0}]
    swaps = 0
    reason = "local_optimum"

    for iteration in range(1, max_iters + 1):
        all_rows = torch.arange(n)
        keragg = objective.features.kernel_dot(A, all_rows)      # sum over current I
        benefit = objective.c - keragg / m                        # marginal include score
        outside = _complement(sel, n)

        drop_order = sel[torch.argsort(benefit.index_select(0, sel))]
        add_order = outside[torch.argsort(-benefit.index_select(0, outside), stable=True)]
        drop_order = drop_order[:candidates]
        add_order = add_order[:candidates]

        deltas = objective.swap_deltas(sel, A, drop_order, add_order)
        j = int(torch.argmin(deltas))
        delta = float(deltas.reshape(-1)[j])
        if delta >= -tolerance:
            reason = "local_optimum"
            break
        i_loc = j // add_order.numel()
        b_loc = j % add_order.numel()
        out_t = drop_order[i_loc:i_loc + 1]
        add_t = add_order[b_loc:b_loc + 1]
        A = objective.features.shift(A, out_t, add_t)
        sel = torch.cat([sel[sel != int(out_t[0])], add_t])
        value = objective.value(sel)
        swaps += 1
        history.append({"iteration": iteration, "value": value, "swaps": swaps})
    else:
        reason = "max_iters"

    diag = {"rule": "ranked", "swaps": swaps, "iterations": len(history) - 1,
            "value": value, "reason": reason}
    return torch.sort(sel).values, history, diag


def select(features: TrajectoryFeatures, setting: str, config: SelectionConfig,
           subtask_targets: Optional[Sequence[Sequence[torch.Tensor]]] = None) -> SelectionResult:
    """End-to-end selection: build ``c`` for the setting, then run the chosen rule."""
    c = build_score(features, setting, subtask_targets=subtask_targets,
                    aggregation=config.aggregation)
    objective = SwapObjective(features, c)
    m = config.subset_size(features.n)
    if config.rule == "ranked":
        return ranked_swap(objective, m, max_iters=config.max_iters,
                           block_size=config.block_size, tolerance=config.tolerance,
                           seed=config.seed)
    return best_swap(objective, m, max_iters=config.max_iters,
                     block_size=config.block_size, tolerance=config.tolerance,
                     seed=config.seed)
