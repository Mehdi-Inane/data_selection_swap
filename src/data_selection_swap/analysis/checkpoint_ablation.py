#!/usr/bin/env python3
"""Checkpoint-count ablation: test accuracy against the selection budget.

Reads full-network KL runs (``kl_faithful_full_<architecture>_sgd``) and
draws one accuracy curve per K. Also writes a booktabs table with one row
per architecture and checkpoint count. Each fraction has a mean-and-std
column and a variance column.

The CSV uses that table's columns (``d_exact_<dataset>_<architecture>.csv``),
so the same reader can load an ablation and a baseline comparison.

Usage:
    python scripts/plot/checkpoint_ablation.py --dataset cifar100
    python scripts/plot/checkpoint_ablation.py --dataset cifar100 \
        --checkpoints 1 4 10 15 20 --fractions 0.1 0.2 0.4 0.5
"""

import argparse
import os

from .comparison import N_TRAIN, mean_std
from .result_plots import (
    emphasizes_k1, k_accuracy_series, plot_accuracy_vs_fraction, with_accuracy,
    collect_runs, write_table,
)

METHOD = 'kl_faithful'
ARCHITECTURES = ('resnet18', 'vgg16', 'inception')
ARCHITECTURE_NAMES = {
    'resnet18': 'ResNet-18',
    'vgg16': 'VGG-16',
    'inception': 'Inception',
}


def full_method(architecture):
    """Filename prefix written by ``selection.kl_full`` with the SGD objective."""
    return f"kl_faithful_full_{architecture}_sgd"


def methods_for(architecture, methods):
    if methods:
        return list(methods)
    return [full_method(architecture)]


def collect_ablation_rows(scratch, dataset, architecture, fractions, seeds,
                          checkpoints, methods=None):
    """Per-run rows restricted to the requested checkpoint counts."""
    methods = list(methods) if methods else [METHOD]
    wanted = set(int(k) for k in checkpoints)
    rows = collect_runs(scratch, dataset, architecture, fractions, seeds, methods)
    return [row for row in rows if row.get('k') in wanted]


def write_k_latex(path, scratch, dataset, architectures, fractions, seeds,
                  checkpoints, methods=None):
    """Booktabs table: architecture, K, and test accuracy at each fraction.

    Each fraction has two cells: mean ± std, and the sample variance over
    seeds (the square of that std). ``methods`` overrides the per-architecture
    ``kl_faithful_full_<architecture>_sgd`` prefix.
    """
    buckets = {}
    for architecture in architectures:
        rows = with_accuracy(collect_ablation_rows(
            scratch, dataset, architecture, fractions, seeds, checkpoints,
            methods=methods_for(architecture, methods)))
        for row in rows:
            key = (architecture, int(row['k']), row['fraction'])
            buckets.setdefault(key, []).append(float(row['accuracy']))

    frac_headers = ' & '.join(
        rf'\multicolumn{{2}}{{c}}{{{100 * fraction:g}\%}}' for fraction in fractions)
    stat_headers = ' & '.join(['Mean $\\pm$ std & Var.'] * len(fractions))
    rules = ' '.join(
        rf'\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}' for i in range(len(fractions)))
    lines = [
        r'\begin{table}',
        r'\centering',
        r'\caption{Test accuracy (\%) of full-network Gradient Swap on '
        + dataset + r'. Mean, standard deviation, and variance are over seeds.}',
        r'\begin{tabular}{ll' + 'cc' * len(fractions) + '}',
        r'\toprule',
        r'Architecture & $K$ & ' + frac_headers + r' \\',
        rules,
        r' & & ' + stat_headers + r' \\',
        r'\midrule',
    ]
    for architecture in architectures:
        for k in checkpoints:
            cells = [ARCHITECTURE_NAMES.get(architecture, architecture), str(int(k))]
            for fraction in fractions:
                values = buckets.get((architecture, int(k), fraction), [])
                if not values:
                    cells.extend(['--', '--'])
                    continue
                mean, std = mean_std(values)
                cells.append(f'{mean:.2f} $\\pm$ {std:.2f}')
                cells.append(f'{std * std:.2f}')
            lines.append(' & '.join(cells) + r' \\')
        lines.append(r'\midrule')
    if lines[-1] == r'\midrule':
        lines.pop()
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))


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
                   choices=list(ARCHITECTURES),
                   help='Architecture drawn in the accuracy figure.')
    p.add_argument('--architectures', default=list(ARCHITECTURES), nargs='+',
                   choices=list(ARCHITECTURES),
                   help='Architectures included in the LaTeX table.')
    p.add_argument('--fractions', default=[0.1, 0.2, 0.4, 0.5], type=float, nargs='+')
    p.add_argument('--seeds', default=[42, 43, 44, 45, 46], type=int, nargs='+')
    p.add_argument('--checkpoints', default=[1, 4, 10, 15, 20], type=int, nargs='+',
                   help='--num_checkpoints values in the saved sweep.')
    p.add_argument('--methods', default=None, nargs='+',
                   help='Filename prefix before _<budget>_K. '
                        'Default: kl_faithful_full_<architecture>_sgd.')
    p.add_argument('--scratch', default=os.environ.get('SCRATCH', '/home/mila/a/ahmedm/scratch'))
    p.add_argument('--out_dir', default=None,
                   help='Default: $SCRATCH/data_selection_baselines/checkpoint_ablation')
    args = p.parse_args(argv)

    out_dir = args.out_dir or os.path.join(
        args.scratch, 'data_selection_baselines', 'checkpoint_ablation')
    tex_path = os.path.join(out_dir, f'{args.dataset}_k_accuracy.tex')
    write_k_latex(
        tex_path, args.scratch, args.dataset, args.architectures, args.fractions,
        args.seeds, args.checkpoints, args.methods)
    print(f"Saved → {tex_path}")

    rows = collect_ablation_rows(
        args.scratch, args.dataset, args.architecture, args.fractions,
        args.seeds, args.checkpoints,
        methods_for(args.architecture, args.methods))
    if not rows:
        print(f"No checkpoint-ablation runs found under {args.scratch} for "
              f"{args.dataset}/{args.architecture}.")
        return

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
