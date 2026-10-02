#!/usr/bin/env python3
"""Score every saved subset from a precomputed kernel. Does not load the dataset.

    python scripts/analyze/score_kernel.py --dataset cifar10 --architecture resnet18
"""

from data_selection_swap.analysis.kernel import score_main


if __name__ == '__main__':
    score_main()
