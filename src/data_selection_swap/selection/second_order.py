"""
Second-order k x k gradient swap.

Fold the checkpoint weight, the metric and any sketch into one vector per
example, z_r = (sqrt(w_t) R P (g_r^t - gbar^t))_t. The objective is then
G(I) = ||u||^2 / m^2 with u = sum_{i in I} z_i, and exchanging i in I for
j not in I changes it by exactly (Eq. 9)

    Delta(i -> j) = d_i + a_j - 2 C_ij / m^2,
    d_i = -s_i + q_i / m^2      (exact change from removing i alone)
    a_j = +s_j + q_j / m^2      (exact change from adding j alone)

with s_r = (2/m^2) <z_r, u>, q_r = C_rr = ||z_r||^2 and C_ij = <z_i, z_j>.

Algorithm 1 ranks by s alone. |s_r| grows with ||z_r||, so its top pair is
made of high-norm examples whose curvature q/m^2 outweighs the linear gain,
the exchange is rejected, and Algorithm 1 stops. Here the k drops and k
adds are shortlisted by d and a, which already contain q/m^2, and the best
of the k x k exchanges is found with the exact cross term C_ij. An exchange
is accepted only if that exact change is below -accept_tol, so G strictly
decreases at every step (Theorem 4.6 (i)).

The rule only needs these members of a trajectory object, implemented by
``kl_full.CachedTrajectory`` (Mode A, cached sketch, no network pass per
sweep) and ``kl_full.Trajectory`` (Mode B, exact in R^d):

    X                   n = X.shape[0] and the device
    u                   list of float64 aggregates, updated in place
    set_subset(index)   recompute u from scratch
    aggregate(index)    aggregates over index, laid out like u
    D(m)                objective at the current u
    gradient(m)         linear scores s (up to a constant, which cancels)
    diagonal()          q, float64 [n]
    swap_block(drop, add, m)   exact float64 Delta for every drop x add pair
"""

import math

import torch

from .. import subsets


def _class_members(labels, n, num_classes, device):
    """Positions per class; the whole pool as one class when unlabelled."""
    if labels is None:
        return [torch.arange(n, device=device)]
    labels = torch.as_tensor(labels, dtype=torch.long, device=device).reshape(-1)
    if labels.numel() != n:
        raise ValueError(f"labels has {labels.numel()} entries; the pool has {n}")
    classes = int(num_classes) if num_classes is not None else int(labels.max()) + 1
    return [torch.where(labels == c)[0] for c in range(classes)]


def _smallest(candidates, score, size):
    """The ``size`` candidates with the smallest score, ties broken by position."""
    return candidates[torch.argsort(score[candidates], stable=True)[:size]]


def second_order_swap(traj, m, k=32, max_iters=1500, accept_tol=1e-12, labels=None,
                      num_classes=None, class_k=16, init_index=None, seed=None,
                      logger=None):
    """
    Diagonal-aware k x k swap descent. Returns ``(indices, history, diag)``.

    Each sweep scores every example once, shortlists the k selected examples
    with the smallest d and the k unselected ones with the smallest a (fewer
    if fewer exist), and applies the best exchange of that block if its
    exact change is below ``-accept_tol``; otherwise it stops. When the
    shortlists hold every example (k >= max(m, n - m)) the result is an
    exact one-swap local optimum, reported as ``one_swap_local_optimum``.

    With ``labels`` the start is stratified at ``subsets.per_class_budget``
    quotas (a given ``init_index`` must meet them) and exchanges stay inside
    a class: class c shortlists up to ``class_k`` drops from I ∩ S_c and up
    to ``class_k`` adds from S_c \\ I, the scores still use the global
    aggregate, and the best exchange over all classes is taken.

    Each accepted history entry records ``dropped``, ``added`` and the exact
    ``change``, so ``subsets.replay_swaps`` rebuilds any intermediate subset.
    """
    if k < 1 or class_k < 1:
        raise ValueError("k and class_k must be positive")
    if max_iters < 0:
        raise ValueError("max_iters must be non-negative")
    if not accept_tol >= 0:
        raise ValueError("accept_tol must be non-negative")
    n, device = int(traj.X.shape[0]), traj.X.device
    groups = _class_members(labels, n, num_classes, device)
    width = k if labels is None else class_k

    start = subsets.initial_subset(n, m, seed=seed, init_index=init_index,
                                   labels=labels, num_classes=num_classes)
    selected = torch.zeros(n, dtype=torch.bool, device=device)
    selected[start.to(device)] = True
    traj.set_subset(torch.where(selected)[0])
    q = traj.diagonal().to(device=device, dtype=torch.float64)
    inv = 1.0 / m ** 2

    history = [{"iteration": 0, "D": traj.D(m), "swaps": 0}]
    swaps = sweeps = 0
    reason, best, complete = 'max_iters', None, False
    for sweep in range(1, max_iters + 1):
        sweeps = sweep
        s = traj.gradient(m).to(device=device, dtype=torch.float64)
        drop_score, add_score = q * inv - s, q * inv + s

        best, pair, complete = math.inf, None, True
        for c, members in enumerate(groups):
            inside = selected[members]
            pool_in, pool_out = members[inside], members[~inside]
            if pool_in.numel() == 0 or pool_out.numel() == 0:
                continue
            drop = _smallest(pool_in, drop_score, width)
            add = _smallest(pool_out, add_score, width)
            complete &= drop.numel() == pool_in.numel() and add.numel() == pool_out.numel()
            block = traj.swap_block(drop, add, m)
            row, col = divmod(int(torch.argmin(block)), block.shape[1])
            value = float(block[row, col])
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite swap change at sweep {sweep}")
            if value < best:
                best, pair = value, (int(drop[row]), int(add[col]), c)

        if pair is None or not best < -accept_tol:
            reason = 'no improving swap in the shortlist'
            break
        i, j, c = pair
        deltas = [a - b for a, b in zip(traj.aggregate(torch.tensor([j], device=device)),
                                        traj.aggregate(torch.tensor([i], device=device)))]
        for u, du in zip(traj.u, deltas):
            u += du
        selected[i], selected[j] = False, True
        swaps += 1
        record = {"iteration": swaps, "D": traj.D(m), "swaps": swaps,
                  "dropped": [i], "added": [j], "change": best}
        if labels is not None:
            record["class"] = c
        history.append(record)
        if logger and (sweep % 10 == 0 or sweep == 1):
            logger.info(f"sweep {sweep}: D={record['D']:.6e}  change={best:.3e}")

    index = torch.where(selected)[0]
    # u accumulated swap differences; recompute it and report how far it drifted.
    incremental = [u.clone() for u in traj.u]
    traj.set_subset(index)
    diag = {"rule": "second_order", "reason": reason, "sweeps": sweeps, "swaps": swaps,
            "D": traj.D(m),
            "final_best_change": best if best is not None and math.isfinite(best) else None,
            "one_swap_local_optimum": reason != 'max_iters' and complete,
            "aggregate_drift_max_abs": max(float((a - b).abs().max())
                                           for a, b in zip(incremental, traj.u))}
    history.append({"iteration": swaps, "final": True, **diag})
    if logger:
        logger.info(f"second-order swap stopped after {sweeps} sweeps ({reason}): "
                    f"{swaps} swaps, D {history[0]['D']:.6e} -> {diag['D']:.6e}")
        if reason == 'max_iters':
            logger.warning("--max_iters reached, so the search had not settled; "
                           "the subset is the best found, not a local optimum.")
    return index.cpu(), history, diag
