#!/usr/bin/env python3
"""JL-embedding ablation: test accuracy against the selection budget.

Reads full-network KL runs at checkpoint counts K and count-sketch widths p.
``p=0`` is the exact aggregate from the checkpoint-count sweep
(``kl_faithful_full_<architecture>_sgd``). Sketched runs use the
``_jl<p>`` filename tag. Draws one accuracy curve per ``(K, p)`` and writes
a booktabs table with one row per architecture, checkpoint count, and width.
Each fraction has a mean-and-std column and a variance column.

The CSV uses the baseline table's columns
(``d_exact_<dataset>_<architecture>.csv``), so the same reader can load an
ablation and a baseline comparison.

Usage:
    python scripts/plot/jl_ablation.py --dataset cifar10 --architecture resnet18
    python scripts/plot/jl_ablation.py --dataset cifar100 \
        --checkpoints 1 4 --proj_dims 0 1024 2048 4096
"""

import argparse
import os
import re

from ..paths import tree
from .checkpoint_ablation import (
    ARCHITECTURE_NAMES, ARCHITECTURES, full_method, print_accuracy,
)
from .comparison import N_TRAIN, mean_std
from .result_plots import (
    _aggregate_accuracy, collect_runs, plot_accuracy_vs_fraction, with_accuracy,
    write_table,
)

PROJ_DIMS = (0, 1024, 2048, 4096)
_JL_DIM = re.compile(r'_jl(\d+)$')


def proj_dim_of(method):
    """Sketch width encoded in a method name. Unsuffixed names are exact (``p=0``)."""
    match = _JL_DIM.search(str(method))
    return int(match.group(1)) if match else 0


def method_name(architecture, proj_dim, variant='sgd'):
    """Filename prefix written by ``selection.kl_full`` for this width."""
    base = full_method(architecture, variant)
    if int(proj_dim) == 0:
        return base
    return f'{base}_jl{int(proj_dim)}'


def methods_for(architecture, proj_dims, methods=None, variant='sgd'):
    if methods:
        return list(methods)
    return [method_name(architecture, proj_dim, variant) for proj_dim in proj_dims]


def collect_ablation_rows(scratch, dataset, architecture, fractions, seeds,
                          checkpoints, proj_dims=PROJ_DIMS, methods=None,
                          variant=''):
    """Per-run rows restricted to the requested checkpoint counts and widths."""
    wanted_k = set(int(k) for k in checkpoints)
    rows = collect_runs(
        scratch, dataset, architecture, fractions, seeds,
        methods_for(architecture, proj_dims, methods, variant), variant=variant)
    kept = []
    for row in rows:
        if row.get('k') not in wanted_k:
            continue
        copied = dict(row)
        copied['proj_dim'] = proj_dim_of(row['method'])
        kept.append(copied)
    return kept


def jl_accuracy_series(rows):
    """Seed-averaged accuracy vs fraction, one curve per ``(K, p)``."""
    tagged = [row for row in rows if row.get('k') is not None]

    def label_of(row):
        proj_dim = row['proj_dim'] if 'proj_dim' in row else proj_dim_of(row['method'])
        return f"K={int(row['k'])}, p={int(proj_dim)}"

    series = _aggregate_accuracy(tagged, label_of)

    def sort_key(label):
        k_text, _, p_text = label.partition(',')
        try:
            k = int(k_text.split('=')[-1])
        except ValueError:
            k = 0
        try:
            proj_dim = int(p_text.split('=')[-1])
        except ValueError:
            proj_dim = 0
        return (k, proj_dim, label)

    return [dict(label=label, points=sorted(series[label], key=lambda point: point[0]))
            for label in sorted(series, key=sort_key)]


def emphasizes_exact(label):
    """Draw the exact aggregate (``p=0``) thicker than the sketches."""
    return str(label).endswith('p=0')


def write_jl_latex(path, scratch, dataset, architectures, fractions, seeds,
                   checkpoints, proj_dims, methods=None, variant=''):
    """Booktabs table: architecture, K, p, and test accuracy at each fraction.

    Each fraction has two cells: mean ± std, and the sample variance over
    seeds (the square of that std). ``p=0`` is the exact aggregate.
    ``methods`` overrides the per-architecture filename prefixes.
    """
    buckets = {}
    for architecture in architectures:
        rows = with_accuracy(collect_ablation_rows(
            scratch, dataset, architecture, fractions, seeds, checkpoints,
            proj_dims, methods=methods, variant=variant))
        for row in rows:
            key = (architecture, int(row['k']), int(row['proj_dim']), row['fraction'])
            buckets.setdefault(key, []).append(float(row['accuracy']))

    frac_headers = ' & '.join(
        rf'\multicolumn{{2}}{{c}}{{{100 * fraction:g}\%}}' for fraction in fractions)
    stat_headers = ' & '.join(['Mean $\\pm$ std & Var.'] * len(fractions))
    rules = ' '.join(
        rf'\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}' for i in range(len(fractions)))
    lines = [
        r'\begin{table}',
        r'\centering',
        r'\caption{Test accuracy (\%) of full-network Gradient Swap with a '
        r'count sketch of width $p$ on ' + dataset
        + r'. $p=0$ is the exact aggregate. Mean, standard deviation, and '
        r'variance are over seeds.}',
        r'\begin{tabular}{lll' + 'cc' * len(fractions) + '}',
        r'\toprule',
        r'Architecture & $K$ & $p$ & ' + frac_headers + r' \\',
        rules,
        r' & & & ' + stat_headers + r' \\',
        r'\midrule',
    ]
    for architecture in architectures:
        for k in checkpoints:
            for proj_dim in proj_dims:
                cells = [ARCHITECTURE_NAMES.get(architecture, architecture),
                         str(int(k)), str(int(proj_dim))]
                for fraction in fractions:
                    values = buckets.get(
                        (architecture, int(k), int(proj_dim), fraction), [])
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='cifar10', choices=list(N_TRAIN))
    p.add_argument('--architecture', default='resnet18',
                   choices=list(ARCHITECTURES),
                   help='Architecture drawn in the accuracy figure.')
    p.add_argument('--architectures', default=list(ARCHITECTURES), nargs='+',
                   choices=list(ARCHITECTURES),
                   help='Architectures included in the LaTeX table.')
    p.add_argument('--fractions', default=[0.1, 0.2, 0.4, 0.5], type=float, nargs='+')
    p.add_argument('--seeds', default=[42, 43, 44, 45, 46], type=int, nargs='+')
    p.add_argument('--checkpoints', default=[1, 4], type=int, nargs='+',
                   help='--num_checkpoints values in the saved sweep.')
    p.add_argument('--proj_dims', default=list(PROJ_DIMS), type=int, nargs='+',
                   help='Count-sketch widths. 0 is the exact aggregate.')
    p.add_argument('--methods', default=None, nargs='+',
                   help='Filename prefix before _<budget>_K. '
                        'Default: kl_faithful_full_<architecture>_sgd[_jl<p>].')
    p.add_argument('--scratch', default=os.environ.get('SCRATCH', '/home/mila/a/ahmedm/scratch'))
    p.add_argument('--variant', default='sgd', choices=['sgd', 'adam'],
                   help="Result tree. 'sgd' is the historical layout; 'adam' reads "
                        "the adam/ subfolders and the _adam filename prefix.")
    p.add_argument('--out_dir', default=None,
                   help='Default: $SCRATCH/data_selection_baselines[/adam]/jl_ablation')
    args = p.parse_args(argv)

    out_dir = args.out_dir or os.path.join(
        tree(args.scratch, 'data_selection_baselines', args.variant),
        'jl_ablation')
    tex_path = os.path.join(out_dir, f'{args.dataset}_jl_accuracy.tex')
    write_jl_latex(
        tex_path, args.scratch, args.dataset, args.architectures, args.fractions,
        args.seeds, args.checkpoints, args.proj_dims, args.methods,
        variant=args.variant)
    print(f"Saved → {tex_path}")

    rows = collect_ablation_rows(
        args.scratch, args.dataset, args.architecture, args.fractions,
        args.seeds, args.checkpoints, args.proj_dims,
        methods=args.methods, variant=args.variant)
    if not rows:
        print(f"No JL-ablation runs found under {args.scratch} for "
              f"{args.dataset}/{args.architecture}.")
        return

    csv_path = os.path.join(
        out_dir, f"d_exact_{args.dataset}_{args.architecture}.csv")
    scoring = f"jl_ablation|{args.dataset}|{args.architecture}"
    write_table(csv_path, rows, scoring, args.dataset, args.architecture)
    print(f"Saved → {csv_path}")

    series = jl_accuracy_series(with_accuracy(rows))
    print_accuracy(series)
    plot_accuracy_vs_fraction(
        series,
        os.path.join(out_dir, f"{args.architecture}_{args.dataset}.pdf"),
        emphasize=emphasizes_exact)


if __name__ == "__main__":
    main()
