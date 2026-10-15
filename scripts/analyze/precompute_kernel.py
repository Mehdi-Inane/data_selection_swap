#!/usr/bin/env python3
"""Fetch one trajectory and store P, r, and c for later subset scores.

``q`` depends on the budget, so the file stores ``r`` and ``q(m) = -2 r / m``
is applied when a subset is scored. One file per seed: the reference
checkpoints depend on it. Pass ``--seed`` to build a single seed, or
omit it to loop the experimental seeds.

    python scripts/analyze/precompute_kernel.py --dataset cifar10 --architecture resnet18 --seed 42 --data_dir /path/to/cifar10
"""

from data_selection_swap.analysis.kernel import precompute_main


if __name__ == '__main__':
    precompute_main()
