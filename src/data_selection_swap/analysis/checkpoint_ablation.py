#!/usr/bin/env python3
"""Checkpoint-count ablation: test accuracy against the selection budget.

Reads the runs produced by checkpoint_ablation.sh (``selection.kl`` with
``--num_checkpoints`` in {1, 3, 5, 10, 20}, every fraction and seed) and
draws one curve per K. Accuracy is ``100 - error``, with ``error`` the same
epoch-averaged test error stored in the baseline comparison tables.

The CSV uses that table's columns (``d_exact_<dataset>_<architecture>.csv``),
so the same reader can load an ablation and a baseline comparison.

Usage:
    python scripts/plot/checkpoint_ablation.py --dataset cifar100
    python scripts/plot/checkpoint_ablation.py --dataset cifar100 \
        --checkpoints 1 3 5 10 20 --fractions 0.1 0.2 0.4 0.5
"""

import argparse
import os

from .comparison import N_TRAIN
from .result_plots import (
    emphasizes_k1, k_accuracy_series, plot_accuracy_vs_fraction, with_accuracy,
    collect_runs, write_table,
)

METHOD = 'kl_faithful'


def collect_ablation_rows(scratch, dataset, architecture, fractions, seeds,
                          checkpoints, methods=None):
    """Per-run rows restricted to the requested checkpoint counts."""
    methods = list(methods) if methods else [METHOD]
    wanted = set(int(k) for k in checkpoints)
    rows = collect_runs(scratch, dataset, architecture, fractions, seeds, methods)
    return [row for row in rows if row.get('k') in wanted]


def print_accuracy(series):
    header = f"{'series':<28}{'fraction':<12}{'test acc %':<18}{'seeds'}"
    print("\n" + header + "\n" + "-" * len(header))
    for item in series:
        for fraction, mean, std, n in item['points']:
            print(f"{item['label']:<28}{fraction:<12g}{mean:6.2f} ± {std:<8.2f}{n}")
    print("-" * len(header))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='cifar100', choices=list(N_TRAIN))
    p.add_argument('--architecture', default='resnet18',
                   choices=['resnet18', 'vgg16', 'inception'])
    p.add_argument('--fractions', default=[0.1, 0.2, 0.4, 0.5], type=float, nargs='+')
    p.add_argument('--seeds', default=[42, 43, 44, 45, 46], type=int, nargs='+')
    p.add_argument('--checkpoints', default=[1, 3, 5, 10, 20], type=int, nargs='+',
                   help='--num_checkpoints values swept by checkpoint_ablation.sh')
    p.add_argument('--methods', default=None, nargs='+',
                   help='Default: kl_faithful.')
    p.add_argument('--scratch', default=os.environ.get('SCRATCH', '/home/mila/a/ahmedm/scratch'))
    p.add_argument('--out_dir', default=None,
                   help='Default: $SCRATCH/data_selection_baselines/checkpoint_ablation')
    args = p.parse_args(argv)

    rows = collect_ablation_rows(
        args.scratch, args.dataset, args.architecture, args.fractions,
        args.seeds, args.checkpoints, args.methods)
    if not rows:
        print(f"No checkpoint-ablation runs found under {args.scratch} for "
              f"{args.dataset}/{args.architecture}.")
        return

    out_dir = args.out_dir or os.path.join(
        args.scratch, 'data_selection_baselines', 'checkpoint_ablation')
    csv_path = os.path.join(
        out_dir, f"d_exact_{args.dataset}_{args.architecture}.csv")
    scoring = f"checkpoint_ablation|{args.dataset}|{args.architecture}"
    write_table(csv_path, rows, scoring, args.dataset, args.architecture)
    print(f"Saved → {csv_path}")

    series = k_accuracy_series(with_accuracy(rows))
    print_accuracy(series)
    plot_accuracy_vs_fraction(
        series,
        os.path.join(out_dir, f"{args.architecture}_{args.dataset}.pdf"),
        emphasize=emphasizes_k1)


if __name__ == "__main__":
    main()
