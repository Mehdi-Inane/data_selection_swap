"""Subset bookkeeping shared by the selectors: quotas, starts, replays.

No vision dependencies, so the swap algorithms and their unit tests use the
same rules as the baselines without importing ``common``
(``common.per_class_budget`` re-exports the function below).
"""

import torch


def per_class_budget(targets: torch.Tensor, num_classes: int, budget: int) -> torch.Tensor:
    """Proportional per-class quotas that sum exactly to ``budget``.

    Each class gets ``floor(n_c * budget / n)`` and the remainder goes to the
    classes with the largest fractional parts.
    """
    counts = torch.bincount(targets, minlength=num_classes).float()
    raw    = counts * (budget / counts.sum())
    quota  = raw.floor().long()
    remainder = int(budget - quota.sum().item())
    if remainder > 0:
        frac_order = torch.argsort(raw - quota.float(), descending=True)
        for c in frac_order[:remainder]:
            quota[c] += 1
    return quota  # LongTensor [num_classes]


def long_tail_indices(targets, num_classes, ratio, seed=0):
    """
    Training positions of a long-tailed subsample (CIFAR-10-LT / CIFAR-100-LT).

    Class c keeps int(n_max * ratio^(-c / (C - 1))) of its examples, n_max
    being the largest class, so the first class keeps everything and the
    largest-to-smallest ratio is ``ratio``. Which examples a class keeps is a
    seeded draw. Returns sorted positions.
    """
    targets = torch.as_tensor(targets, dtype=torch.long).reshape(-1)
    counts = torch.bincount(targets, minlength=num_classes)
    n_max = int(counts.max())
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    kept = []
    for c in range(num_classes):
        members = torch.where(targets == c)[0]
        keep = min(int(n_max * ratio ** (-c / (num_classes - 1))), members.numel())
        kept.append(members[torch.randperm(members.numel(), generator=generator)[:keep]])
    return torch.cat(kept).sort().values.tolist()


def initial_subset(n, m, *, seed=None, init_index=None, labels=None, num_classes=None):
    """
    Sorted CPU positions of the subset a swap search starts from.

    ``init_index`` is validated and returned. Otherwise the start is a
    seeded uniform draw of ``m`` positions, so every swap rule starts from
    the same subset at the same seed. With ``labels`` the draw is stratified
    at ``per_class_budget`` quotas, and a given ``init_index`` must already
    meet them, because within-class swaps never change a class count.
    """
    if not 1 <= m < n:
        raise ValueError(f"m must satisfy 1 <= m < n (got m={m}, n={n})")
    quota = None
    if labels is not None:
        labels = torch.as_tensor(labels, dtype=torch.long).reshape(-1).cpu()
        if labels.numel() != n:
            raise ValueError(f"labels has {labels.numel()} entries; the pool has {n}")
        classes = int(num_classes) if num_classes is not None else int(labels.max()) + 1
        quota = per_class_budget(labels, classes, m)

    if init_index is not None:
        index = torch.as_tensor(init_index).detach().reshape(-1).cpu()
        if index.dtype.is_floating_point or index.dtype == torch.bool:
            raise ValueError(
                f"init_index must be integer pool positions, got dtype {index.dtype}")
        index = index.to(torch.long)
        if index.numel() != m:
            raise ValueError(f"init_index has {index.numel()} entries; the budget is {m}")
        if int(index.unique().numel()) != m:
            raise ValueError("init_index contains duplicate positions")
        if int(index.min()) < 0 or int(index.max()) >= n:
            raise ValueError(f"init_index positions must lie in 0..{n - 1}, "
                             f"got {int(index.min())}..{int(index.max())}")
        if quota is not None:
            counts = torch.bincount(labels[index], minlength=quota.numel())
            if not torch.equal(counts, quota):
                raise ValueError(
                    "Class-balanced swaps keep every class count fixed, so the start "
                    f"must meet the per_class_budget quotas {quota.tolist()}; it has "
                    f"{counts.tolist()}.")
        return index.sort().values

    generator = torch.Generator(device='cpu')
    if seed is not None:
        generator.manual_seed(int(seed))
    if quota is None:
        return torch.randperm(n, generator=generator)[:m].sort().values
    chosen = []
    for c, take in enumerate(quota.tolist()):
        members = torch.where(labels == c)[0]
        chosen.append(members[torch.randperm(members.numel(), generator=generator)[:take]])
    return torch.cat(chosen).sort().values


def replay_swaps(start_index, history, upto=None):
    """
    The subset after the first ``upto`` accepted updates of a swap history.

    Accepted entries record the positions they ``dropped`` and ``added``
    (a block of several for Algorithm 1 with --swaps_per_sweep > 1). With
    ``upto=None`` the whole history is replayed.
    """
    current = set(torch.as_tensor(start_index).reshape(-1).tolist())
    applied = 0
    for entry in history:
        if 'dropped' not in entry:
            continue
        if upto is not None and applied >= upto:
            break
        for i in entry['dropped']:
            current.remove(i)
        for j in entry['added']:
            if j in current:
                raise ValueError(f"replay adds {j}, which is already selected")
            current.add(j)
        applied += 1
    return torch.tensor(sorted(current), dtype=torch.long)
