#!/usr/bin/env python3
"""Before and after warm-starting full-network KL from a baseline subset.

Reads saved baseline runs and the ``kl_faithful_full_<architecture>_sgd_from_<method>``
runs produced by ``--init_indices``. Accuracy is the last logged test epoch.
G_T is the all-checkpoint full-network objective (``K=0`` in the result-plot
cache). Missing G_T values are scored with ``result_plots.score_missing`` when
``--data_dir`` is set.

Usage:
    python scripts/plot/baseline_warmstart.py --dataset cifar10 --architecture resnet18
    python scripts/plot/baseline_warmstart.py --dataset cifar10 --architecture resnet18 \\
        --data_dir /path/to/cifar10
"""

import argparse
import csv
import logging
import os
import sys

import numpy as np

from ..paths import normalize_variant, tree
from .checkpoint_ablation import full_method
from .comparison import (
    N_TRAIN, discover_runs, indices_path_for, load_json, mean_std, run_dir,
)
from .result_plots import (
    LINESTYLES, MARKERS, TOL_BRIGHT, load_cached_d, score_missing, scoring_id,
)

BASELINES = ('graphcut', 'herding', 'random', 'el2n')
BASELINE_NAMES = {
    'graphcut': 'GraphCut',
    'herding': 'Herding',
    'random': 'Random',
    'el2n': 'EL2N',
}
SELECTION_K = 10
PAIR_FIELDS = (
    'method', 'seed', 'fraction', 'budget',
    'acc_before', 'g_before', 'acc_after', 'g_after',
    'indices_before', 'indices_after',
)


def final_accuracy(metrics_path):
    """Last logged test accuracy, matching ``score_subsets.final_accuracy``."""
    metrics = load_json(metrics_path)
    if not metrics:
        return None
    last_epoch = str(max(int(epoch) for epoch in metrics))
    return float(metrics[last_epoch])


def warm_method(architecture, baseline, variant='sgd'):
    """Filename prefix of a full-network KL run started from ``baseline``."""
    return f"{full_method(architecture, variant)}_from_{baseline}"


def gt_scoring_id(dataset, architecture, variant='sgd'):
    """Id of the all-checkpoint objective cached by ``result_plots``."""
    weighting = 'adam' if normalize_variant(variant) == 'adam' else 'sgd'
    return scoring_id(
        dataset, architecture, weighting, 'auto', 0, 'spread', 'point', 'test', None)


def _saved_run(scratch, method, dataset, architecture, budget, seed, variant, k):
    """One saved (metrics, indices, accuracy) triple, or None."""
    found = discover_runs(
        run_dir(scratch, method, dataset, architecture, budget, seed, variant=variant),
        method, budget)
    if k is None:
        found = [item for item in found if item[0] is None]
    else:
        found = [item for item in found if item[0] == int(k)]
    if not found:
        return None
    if len(found) > 1:
        raise RuntimeError(
            f"Several runs for method={method!r} budget={budget} seed={seed} K={k}")
    tag, metrics_path, _ = found[0]
    return dict(
        k=tag,
        metrics_path=metrics_path,
        indices_path=indices_path_for(metrics_path),
        acc=final_accuracy(metrics_path),
    )


def collect_pairs(scratch, dataset, architecture, fraction, seeds, methods,
                  variant=''):
    """One row per baseline and seed, with the warm start beside it when present."""
    budget = int(fraction * N_TRAIN[dataset])
    pairs = []
    for baseline in methods:
        started = warm_method(architecture, baseline, variant)
        for seed in seeds:
            before = _saved_run(
                scratch, baseline, dataset, architecture, budget, seed, variant, None)
            after = _saved_run(
                scratch, started, dataset, architecture, budget, seed, variant,
                SELECTION_K)
            pairs.append(dict(
                method=baseline,
                warm_method=started,
                seed=int(seed),
                fraction=float(fraction),
                budget=budget,
                acc_before=None if before is None else before['acc'],
                acc_after=None if after is None else after['acc'],
                indices_before=None if before is None else before['indices_path'],
                indices_after=None if after is None else after['indices_path'],
                g_before=None,
                g_after=None,
                k_after=None if after is None else after['k'],
            ))
    return pairs


def fill_g(pairs, cache_path, scoring):
    """Fill blank G_T entries from a ``d_exact`` cache. Existing values stay."""
    cached = load_cached_d(cache_path, scoring)
    for pair in pairs:
        for side in ('before', 'after'):
            key = f'g_{side}'
            if pair.get(key) is not None:
                continue
            path = pair.get(f'indices_{side}')
            if not path or not os.path.exists(path):
                continue
            pair[key] = cached.get(os.path.normcase(os.path.realpath(path)))
    return pairs


def score_rows_for(pairs):
    """Cache rows ``score_missing`` can score and write back."""
    rows = []
    for pair in pairs:
        sides = (
            ('before', pair['method'], None, pair.get('indices_before'), pair.get('g_before')),
            ('after', pair['warm_method'], pair.get('k_after'),
             pair.get('indices_after'), pair.get('g_after')),
        )
        for _, method, k, path, value in sides:
            if not path:
                continue
            rows.append(dict(
                fraction=pair['fraction'],
                budget=pair['budget'],
                method=method,
                k=k,
                seed=pair['seed'],
                error=None,
                selection_s=None,
                indices_path=path,
                d=value,
            ))
    return rows


def apply_scored(pairs, rows):
    """Copy scored ``d`` values back onto the before/after pairs."""
    by_path = {}
    for row in rows:
        path = row.get('indices_path')
        if path and row.get('d') is not None:
            by_path[os.path.normcase(os.path.realpath(path))] = float(row['d'])
    for pair in pairs:
        for side in ('before', 'after'):
            path = pair.get(f'indices_{side}')
            if not path or not os.path.exists(path):
                continue
            value = by_path.get(os.path.normcase(os.path.realpath(path)))
            if value is not None:
                pair[f'g_{side}'] = value
    return pairs


def _finite_values(pairs, key):
    values = []
    for pair in pairs:
        value = pair.get(key)
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value):
            values.append(value)
    return values


def summarize(pairs, methods):
    """Mean and sample std of each before/after quantity, in method order."""
    summary = []
    for method in methods:
        rows = [pair for pair in pairs if pair['method'] == method]
        item = dict(method=method, label=BASELINE_NAMES.get(method, method))
        for key in ('acc_before', 'acc_after', 'g_before', 'g_after'):
            values = _finite_values(rows, key)
            if not values:
                item[key] = (None, None, 0)
                continue
            mean, std = mean_std(values)
            item[key] = (mean, std, len(values))
        summary.append(item)
    return summary


def _cell(stat, spec):
    mean, std, _ = stat
    if mean is None or not np.isfinite(mean):
        return '--'
    if std is None or not np.isfinite(std):
        std = 0.0
    return f'{mean:{spec}} $\\pm$ {std:{spec}}'


def write_pairs_csv(path, pairs):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PAIR_FIELDS))
        writer.writeheader()
        for pair in pairs:
            writer.writerow({field: '' if pair.get(field) is None else pair[field]
                             for field in PAIR_FIELDS})


def write_warmstart_latex(path, summary, dataset, architecture, fraction):
    """Booktabs table of mean ± std over seeds."""
    lines = [
        r'\begin{table}',
        r'\centering',
        r'\caption{Warm start of full-network Gradient Swap on '
        + dataset + ' / ' + architecture
        + r' at ' + f'{100 * fraction:g}'
        + r'\% of the pool. Accuracy is the last logged test epoch. '
        r'$G_T$ is the full-network objective on every reference checkpoint. '
        r'The after columns are exact full-network Gradient Swap at $K=10$ '
        r'started from the baseline subset. Mean and standard deviation are over seeds.}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'Method & Acc. before & Acc. after & $G_T$ before & $G_T$ after \\',
        r'\midrule',
    ]
    for item in summary:
        cells = [
            item['label'],
            _cell(item['acc_before'], '.2f'),
            _cell(item['acc_after'], '.2f'),
            _cell(item['g_before'], '.4g'),
            _cell(item['g_after'], '.4g'),
        ]
        lines.append(' & '.join(cells) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))


def plot_slopes(summary, out_path):
    """Two panels. Each method is a segment from the baseline value to after KL."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    panels = (
        ('acc_before', 'acc_after', 'Test accuracy'),
        ('g_before', 'g_after', r'$G_T$'),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    drew = False
    for index, item in enumerate(summary):
        style = dict(
            color=TOL_BRIGHT[index % len(TOL_BRIGHT)],
            marker=MARKERS[index % len(MARKERS)],
            linestyle=LINESTYLES[index % len(LINESTYLES)],
        )
        for ax, (before_key, after_key, ylabel) in zip(axes, panels):
            before, after = item[before_key], item[after_key]
            if before[0] is None or after[0] is None:
                continue
            drew = True
            yerr = [
                0.0 if before[1] is None else before[1],
                0.0 if after[1] is None else after[1],
            ]
            ax.errorbar(
                [0, 1], [before[0], after[0]], yerr=yerr, label=item['label'],
                color=style['color'], marker=style['marker'],
                linestyle=style['linestyle'], capsize=3, lw=2, ms=8)
            ax.set_ylabel(ylabel)
    if not drew:
        plt.close(fig)
        print(f"Skipping {os.path.basename(out_path)}: no paired before/after values.")
        return
    for ax in axes:
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Baseline', 'After KL'])
        ax.set_xlim(-0.25, 1.25)
        ax.grid(True, linestyle='--', alpha=0.4)
        if ax.lines:
            ax.legend(fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved -> {out_path}")


def _score_namespace(args, gbar_cache_dir):
    weighting = 'adam' if normalize_variant(args.variant) == 'adam' else 'sgd'
    return argparse.Namespace(
        dataset=args.dataset,
        architecture=args.architecture,
        data_dir=args.data_dir,
        download=args.download,
        checkpoint_dir=args.checkpoint_dir,
        scratch=args.scratch,
        variant=args.variant,
        weighting=weighting,
        metric='auto',
        num_checkpoints=0,
        checkpoint_selection='spread',
        checkpoint_weighting='point',
        checkpoint_epochs=None,
        scoring_transform='test',
        batch_size=128,
        num_workers=4,
        base_lr=0.01,
        beta1=0.9,
        momentum_response='persistent',
        adam_eps=1e-8,
        gbar_cache_dir=gbar_cache_dir,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='cifar10', choices=list(N_TRAIN))
    p.add_argument('--architecture', default='resnet18')
    p.add_argument('--fraction', default=0.1, type=float)
    p.add_argument('--seeds', default=[42, 43, 44, 45, 46], type=int, nargs='+')
    p.add_argument('--methods', default=list(BASELINES), nargs='+',
                   choices=list(BASELINES))
    p.add_argument('--scratch', default=os.environ.get('SCRATCH', '/home/mila/a/ahmedm/scratch'))
    p.add_argument('--variant', default='sgd', choices=['sgd', 'adam'])
    p.add_argument('--out_dir', default=None,
                   help='Default: $SCRATCH/data_selection_baselines[/adam]/baseline_warmstart')
    p.add_argument('--data_dir', default=None,
                   help='Dataset root. Required the first time a missing G_T is scored.')
    p.add_argument('--checkpoint_dir', default=None,
                   help='Reference trajectory. Default: '
                        '$SCRATCH/data_selection_swap/<dataset>/<architecture>/checkpoints')
    p.add_argument('--download', action='store_true', default=False)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger('baseline_warmstart')

    out_dir = args.out_dir or os.path.join(
        tree(args.scratch, 'data_selection_baselines', args.variant),
        'baseline_warmstart')
    plots = os.path.join(
        tree(args.scratch, 'data_selection_baselines', args.variant), 'plots')
    shared_cache = os.path.join(plots, f"d_exact_{args.dataset}_{args.architecture}.csv")
    local_cache = os.path.join(out_dir, f"d_exact_{args.dataset}_{args.architecture}.csv")
    gbar_cache = os.path.join(plots, f"gbar_{args.dataset}_{args.architecture}")
    scoring = gt_scoring_id(args.dataset, args.architecture, args.variant)

    pairs = collect_pairs(
        args.scratch, args.dataset, args.architecture, args.fraction,
        args.seeds, args.methods, variant=args.variant)
    fill_g(pairs, shared_cache, scoring)
    fill_g(pairs, local_cache, scoring)

    pending = [pair for pair in pairs
               if (pair.get('indices_before') and pair.get('g_before') is None)
               or (pair.get('indices_after') and pair.get('g_after') is None)]
    if pending and args.data_dir:
        rows = score_rows_for(pairs)
        score_missing(rows, _score_namespace(args, gbar_cache), scoring, local_cache, logger)
        apply_scored(pairs, rows)
    elif pending:
        logger.info(
            "Some G_T values are missing from the all-checkpoint cache. "
            "Pass --data_dir to score them.")

    os.makedirs(out_dir, exist_ok=True)
    tag = f"{args.dataset}_{args.architecture}_f{args.fraction:g}"
    csv_path = os.path.join(out_dir, f"warmstart_{tag}.csv")
    tex_path = os.path.join(out_dir, f"warmstart_{tag}.tex")
    pdf_path = os.path.join(out_dir, f"warmstart_{tag}.pdf")
    write_pairs_csv(csv_path, pairs)
    print(f"Saved -> {csv_path}")
    summary = summarize(pairs, args.methods)
    write_warmstart_latex(tex_path, summary, args.dataset, args.architecture, args.fraction)
    print(f"Saved -> {tex_path}")
    plot_slopes(summary, pdf_path)


if __name__ == "__main__":
    main()
