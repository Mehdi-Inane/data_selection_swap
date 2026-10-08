#!/usr/bin/env python3
"""Plot baseline and KL-Faithful results, one figure per dataset and architecture.

Figures, mean ± std over seeds:

  * test accuracy vs fraction of the training set
        accuracy is ``100 - error``. ``error`` is the mean of ``100 - accuracy``
        over the epochs stored in ``*_metrics.json`` (logged every 50 epochs,
        plus the last). The baseline comparison keeps KL at K=1; the KL figure
        draws one curve per saved K.
  * test error vs exact full-network D
        every method is scored on the same trajectory: all checkpoints,
        Euclidean gradients, ``D = sum_k w_k ||u_I^k / m - gbar_k||^2``
  * selection time vs budget
        ``selection_seconds`` from ``*_timing.json``

Baselines live under ``$SCRATCH/data_selection_baselines``. KL runs live
under ``$SCRATCH/gradmatch_swap`` and tag ``_K`` plus a config id in the
filename. Discovery is shared with ``analysis.comparison``, so a method
that has not been run is simply absent, and two configs in one seed
directory are refused rather than averaged.

The n-by-n kernel P is not materialised on this path. Mean gradients
``gbar_k`` are cached per checkpoint, and the resulting per-subset D
values are written to ``d_exact_<dataset>_<architecture>.csv``. A later
run reads that file and redraws the D figure without another backward
pass.

To score every saved subset from a stored kernel instead, precompute
``P`` and ``r`` with ``scripts/analyze/precompute_kernel.py`` (Slurm:
``scripts/slurm/run_precompute_kernel.sh``), then run
``scripts/analyze/score_kernel.py``. ``q`` depends on the budget:
``q(m) = -2 r / m``.

Usage:
    python scripts/plot/result_plots.py --dataset cifar10 --architecture resnet18
    python scripts/plot/result_plots.py --dataset cifar10 --architecture resnet18 \\
        --data_dir /path/to/cifar10
"""

import argparse
import csv
import logging
import os
import sys

import numpy as np

from ..paths import default_results_root, tree
from .comparison import (N_TRAIN, discover_methods, discover_runs,
                         indices_path_for, load_json, mean_std, run_dir)

CACHE_FIELDS = [
    'scoring_id', 'dataset', 'architecture', 'fraction', 'budget', 'method',
    'k', 'seed', 'error', 'selection_seconds', 'indices_path', 'd_alpha',
]


def mean_test_error(metrics):
    """Unweighted mean of ``100 - accuracy`` over the logged epochs."""
    if not metrics:
        return None
    errors = [100.0 - float(acc) for acc in metrics.values()]
    if not errors:
        return None
    return float(sum(errors) / len(errors))


def series_label(method, k):
    if k is None:
        return method
    return f"{method}, K={k}"


def collect_runs(scratch, dataset, architecture, fractions, seeds, methods=None,
                 variant=''):
    """One row per saved run that discovery can see. Missing methods contribute nothing."""
    methods = methods or discover_methods(
        scratch, dataset, architecture, variant=variant)
    rows = []
    n_train = N_TRAIN[dataset]
    for fraction in fractions:
        budget = int(fraction * n_train)
        for method in methods:
            for seed in seeds:
                directory = run_dir(
                    scratch, method, dataset, architecture, budget, seed,
                    variant=variant)
                for k, metrics_path, timing_path in discover_runs(
                        directory, method, budget):
                    metrics = load_json(metrics_path)
                    timing = load_json(timing_path) if timing_path else None
                    selection_s = None
                    if timing and timing.get('selection_seconds') is not None:
                        selection_s = float(timing['selection_seconds'])
                    rows.append(dict(
                        fraction=fraction,
                        budget=budget,
                        method=method,
                        k=k,
                        seed=seed,
                        error=mean_test_error(metrics),
                        selection_s=selection_s,
                        indices_path=indices_path_for(metrics_path),
                        metrics_path=metrics_path,
                        d=None,
                    ))
    return rows


def _finite(value):
    return value is not None and np.isfinite(value)


def group_by_budget(rows, value_key):
    """``(method, k)`` series of ``(budget, mean, std)``, seeds averaged."""
    buckets = {}
    for row in rows:
        value = row.get(value_key)
        if not _finite(value):
            continue
        key = (row['method'], row['k'], row['budget'])
        buckets.setdefault(key, []).append(float(value))
    series = {}
    for (method, k, budget), values in buckets.items():
        mean, std = mean_std(values)
        series.setdefault((method, k), []).append((budget, mean, std))
    out = []
    for (method, k), points in series.items():
        points.sort(key=lambda point: point[0])
        out.append(dict(method=method, k=k, label=series_label(method, k),
                        points=points))
    out.sort(key=lambda item: (item['method'], item['k'] is None, item['k'] or 0))
    return out


def group_error_vs_d(rows):
    """``(method, k)`` series of ``(d_mean, d_std, error_mean, error_std)`` per budget."""
    buckets = {}
    for row in rows:
        if not _finite(row.get('error')) or not _finite(row.get('d')):
            continue
        key = (row['method'], row['k'], row['budget'])
        buckets.setdefault(key, []).append((float(row['d']), float(row['error'])))
    series = {}
    for (method, k, budget), pairs in buckets.items():
        d_mean, d_std = mean_std([pair[0] for pair in pairs])
        e_mean, e_std = mean_std([pair[1] for pair in pairs])
        series.setdefault((method, k), []).append(
            (budget, d_mean, d_std, e_mean, e_std))
    out = []
    for (method, k), points in series.items():
        points.sort(key=lambda point: point[0])
        out.append(dict(method=method, k=k, label=series_label(method, k),
                        points=points))
    out.sort(key=lambda item: (item['method'], item['k'] is None, item['k'] or 0))
    return out


def scoring_id(dataset, architecture, weighting, metric, num_checkpoints,
               checkpoint_selection, checkpoint_weighting, scoring_transform,
               checkpoint_epochs):
    epochs = '' if not checkpoint_epochs else ','.join(str(e) for e in checkpoint_epochs)
    return '|'.join([
        dataset, architecture, weighting, metric, f'K={num_checkpoints}',
        checkpoint_selection, checkpoint_weighting, scoring_transform, epochs,
    ])


def _blank(value):
    if value is None:
        return ''
    if isinstance(value, float) and not np.isfinite(value):
        return ''
    return value


def latex_path_for(csv_path, suffix=''):
    root, ext = os.path.splitext(csv_path)
    if ext.lower() != '.csv':
        root = csv_path
    return root + suffix + '.tex'


def is_kl_method(method):
    return str(method).startswith('kl_faithful')


KL_DISPLAY_NAME = "Gradient Swap, 1 epoch (ours)"
TOL_BRIGHT = ['#4477AA', '#EE6677', '#228833', '#AA3377', '#66CCEE', '#CCBB44', '#000000']
LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.']
MARKERS = ["*", "o", "s", "^", "v", "D", "<", ">"]
ACCURACY_RC = {
    "font.size": 20,
    "axes.labelsize": 26,
    "axes.titlesize": 24,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 20,
}


def with_accuracy(rows):
    """Copy rows that have a finite error and set ``accuracy = 100 - error``."""
    converted = []
    for row in rows:
        if not _finite(row.get('error')):
            continue
        copied = dict(row)
        copied['accuracy'] = 100.0 - float(row['error'])
        converted.append(copied)
    return converted


def baseline_comparison_rows(rows, kl_name=KL_DISPLAY_NAME):
    """Rows for the baseline figure: KL restricted to K=1, EL2N's smallest fraction dropped."""
    kept = []
    for row in rows:
        copied = dict(row)
        if is_kl_method(copied['method']):
            if copied.get('k') != 1:
                continue
            copied['method'] = kl_name
        kept.append(copied)
    el2n_fractions = [row['fraction'] for row in kept if row['method'] == 'el2n']
    if el2n_fractions:
        first = min(el2n_fractions)
        kept = [row for row in kept
                if not (row['method'] == 'el2n' and row['fraction'] == first)]
    return kept


def _aggregate_accuracy(rows, label_of):
    """``(fraction, mean, std, n)`` points per label. ``std`` matches pandas (ddof=1)."""
    buckets = {}
    for row in rows:
        if not _finite(row.get('accuracy')):
            continue
        key = (label_of(row), row['fraction'])
        buckets.setdefault(key, []).append(float(row['accuracy']))
    series = {}
    for (label, fraction), values in buckets.items():
        mean, std = mean_std(values)
        if not np.isfinite(std):
            std = 0.0
        series.setdefault(label, []).append((fraction, mean, std, len(values)))
    return series


def comparison_accuracy_series(rows):
    """Seed-averaged accuracy vs fraction. ``rows`` are already the baseline-comparison subset."""
    series = _aggregate_accuracy(rows, lambda row: row['method'])
    others = sorted(label for label in series if label not in (KL_DISPLAY_NAME, 'el2n'))
    if 'el2n' in series:
        others.append('el2n')
    order = ([KL_DISPLAY_NAME] if KL_DISPLAY_NAME in series else []) + others
    for label in series:
        if label not in order:
            order.append(label)
    return [dict(label=label, points=sorted(series[label], key=lambda point: point[0]))
            for label in order]


def k_accuracy_series(rows):
    """Seed-averaged accuracy vs fraction, one curve per saved checkpoint count."""
    tagged = [row for row in rows if row.get('k') is not None]
    methods = {row['method'] for row in tagged}

    def label_of(row):
        k = int(row['k'])
        if len(methods) <= 1:
            return f"K={k}"
        return f"{row['method']}, K={k}"

    series = _aggregate_accuracy(tagged, label_of)

    def sort_key(label):
        k_text = label.split('K=')[-1]
        try:
            k = int(k_text)
        except ValueError:
            k = 0
        return (k, label)

    return [dict(label=label, points=sorted(series[label], key=lambda point: point[0]))
            for label in sorted(series, key=sort_key)]


def emphasizes_k1(label):
    return label == 'K=1' or str(label).endswith(', K=1')


def _latex_escape(text):
    return (str(text).replace('\\', r'\textbackslash{}')
            .replace('&', r'\&').replace('%', r'\%').replace('$', r'\$')
            .replace('#', r'\#').replace('_', r'\_').replace('{', r'\{')
            .replace('}', r'\}'))


def _latex_number(value, spec):
    if value is None:
        return '--'
    try:
        number = float(value)
    except (TypeError, ValueError):
        return '--'
    if not np.isfinite(number):
        return '--'
    return format(number, spec)


def _mean_finite(values):
    kept = [float(value) for value in values if _finite(value)]
    if not kept:
        return None
    return float(np.mean(kept))


def summarize_for_latex(rows):
    """One record per method, K, and budget: means over seeds. Accuracy is ``100 - error``."""
    buckets = {}
    for row in rows:
        key = (row['method'], row['k'], row['budget'])
        bucket = buckets.setdefault(key, {'acc': [], 'selection_s': [], 'd': []})
        if _finite(row.get('error')):
            bucket['acc'].append(100.0 - float(row['error']))
        if _finite(row.get('selection_s')):
            bucket['selection_s'].append(float(row['selection_s']))
        if _finite(row.get('d')):
            bucket['d'].append(float(row['d']))
    summary = []
    for (method, k, budget), bucket in buckets.items():
        summary.append(dict(
            method=method, k=k, budget=budget,
            accuracy=_mean_finite(bucket['acc']),
            selection_s=_mean_finite(bucket['selection_s']),
            d=_mean_finite(bucket['d']),
        ))
    best = {}
    for item in summary:
        if not _finite(item['accuracy']):
            continue
        shown = round(item['accuracy'], 2)
        best[item['budget']] = shown if item['budget'] not in best else max(best[item['budget']], shown)
    for item in summary:
        shown = round(item['accuracy'], 2) if _finite(item['accuracy']) else None
        item['best'] = shown is not None and shown == best.get(item['budget'])
    summary.sort(key=lambda item: (
        item['budget'], item['method'], item['k'] is None, item['k'] or 0))
    return summary


def write_latex(path, rows, dataset, architecture, kl_only=False):
    """Booktabs table of seed-averaged accuracy. The best accuracy at each budget is bold."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    scope = 'KL selectors only' if kl_only else 'all methods'
    caption = (r'Mean test accuracy over seeds (' + scope + ') for '
               + _latex_escape(dataset) + ' / ' + _latex_escape(architecture) + '.')
    lines = [
        r'\begin{table}',
        r'\centering',
        r'\caption{' + caption + '}',
        r'\begin{tabular}{llrrrr}',
        r'\toprule',
        r'Method & $K$ & Budget & Accuracy (\%) & Time (s) & $D$ \\',
        r'\midrule',
    ]
    for item in summarize_for_latex(rows):
        k = '--' if item['k'] is None else str(int(item['k']))
        accuracy = _latex_number(item['accuracy'], '.2f')
        if item['best'] and accuracy != '--':
            accuracy = r'\textbf{' + accuracy + '}'
        lines.append(' & '.join([
            _latex_escape(item['method']),
            k,
            str(int(item['budget'])),
            accuracy,
            _latex_number(item['selection_s'], '.1f'),
            _latex_number(item['d'], '.4g'),
        ]) + r' \\')
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
        '',
    ])
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))


def write_table(path, rows, scoring, dataset, architecture):
    """Per-subset table. ``d`` is blank until that subset has been scored."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=CACHE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                'scoring_id': scoring,
                'dataset': dataset,
                'architecture': architecture,
                'fraction': row['fraction'],
                'budget': row['budget'],
                'method': row['method'],
                'k': '' if row['k'] is None else row['k'],
                'seed': row['seed'],
                'error': _blank(row.get('error')),
                'selection_seconds': _blank(row.get('selection_s')),
                'indices_path': row.get('indices_path') or '',
                'd_alpha': _blank(row.get('d')),
            })
    write_latex(latex_path_for(path), rows, dataset, architecture)


def load_cached_d(path, scoring):
    """``indices path -> D`` for rows written by this scoring configuration."""
    cached = {}
    if not path or not os.path.exists(path):
        return cached
    with open(path, newline='') as fh:
        for record in csv.DictReader(fh):
            if record.get('scoring_id') != scoring:
                continue
            raw = record.get('d_alpha') or ''
            indices = record.get('indices_path') or ''
            if not raw or not indices:
                continue
            cached[os.path.normcase(os.path.realpath(indices))] = float(raw)
    return cached


def attach_cached_d(rows, path, scoring):
    cached = load_cached_d(path, scoring)
    for row in rows:
        indices = row.get('indices_path')
        if not indices:
            row['d'] = None
            continue
        row['d'] = cached.get(os.path.normcase(os.path.realpath(indices)))
    return rows


def locate_checkpoint_dir(template, seed):
    """Use the directory as given, or its ``seed_<seed>`` child when that is where the run lives."""
    path = template.format(seed=seed)
    manifest = os.path.join(path, 'trajectory.json')
    if os.path.isfile(manifest):
        return path
    if os.path.basename(os.path.normpath(path)) == f'seed_{seed}':
        return path
    seeded = os.path.join(path, f'seed_{seed}')
    if os.path.isdir(seeded):
        return seeded
    return path


def _plot(title, xlabel, ylabel, out_path, draw):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with plt.rc_context(ACCURACY_RC):
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        draw(ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=18)
        fig.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f"Saved -> {out_path}")


def _curve_style(index):
    return dict(color=TOL_BRIGHT[index % 7],
                ls=LINESTYLES[(index + index // 7) % 7],
                marker=MARKERS[index % len(MARKERS)])


def plot_accuracy_vs_fraction(series, out_path, emphasize=None, exclude_from_limits=()):
    """Accuracy vs fraction with the baseline-comparison styling.

    ``emphasize`` is a label, or a callable of the label, drawn thicker.
    Series named in ``exclude_from_limits`` stay on the figure but do not
    set the y-limits, which is how EL2N is allowed to run off the bottom.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not series:
        print(f"Skipping {os.path.basename(out_path)}: nothing to plot.")
        return

    def is_emphasized(label):
        if emphasize is None:
            return False
        if callable(emphasize):
            return bool(emphasize(label))
        return label == emphasize

    with plt.rc_context(ACCURACY_RC):
        fig, ax = plt.subplots(figsize=(9, 6.5))
        for index, item in enumerate(series):
            xs = [point[0] for point in item['points']]
            ys = [point[1] for point in item['points']]
            stds = [point[2] for point in item['points']]
            style = _curve_style(index)
            highlighted = is_emphasized(item['label'])
            ax.plot(xs, ys, color=style['color'], linestyle=style['ls'],
                    marker=style['marker'], lw=3.0 if highlighted else 2.2,
                    ms=16 if highlighted else 9, label=item['label'],
                    markeredgecolor=style['color'], zorder=10 if highlighted else 3)
            ax.fill_between(xs, [y - s for y, s in zip(ys, stds)],
                            [y + s for y, s in zip(ys, stds)],
                            color=style['color'], alpha=0.15, lw=0, zorder=2)

        reference = [point for item in series if item['label'] not in exclude_from_limits
                     for point in item['points']]
        if not reference:
            reference = [point for item in series for point in item['points']]
        lo = min(point[1] - point[2] for point in reference)
        hi = max(point[1] + point[2] for point in reference)
        pad = 0.06 * (hi - lo) if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)

        fracs = sorted({point[0] for item in series for point in item['points']})
        ax.set_xticks(fracs)
        ax.set_xlim(min(fracs) - 0.04, max(fracs) + 0.04)
        ax.set_xlabel("Fraction of the training set")
        ax.set_ylabel("Test accuracy")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=20)
        fig.tight_layout()
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_vs_budget(series, title, xlabel, ylabel, out_path):
    if not series:
        print(f"Skipping {os.path.basename(out_path)}: nothing to plot.")
        return

    def draw(ax):
        for item in series:
            xs = [point[0] for point in item['points']]
            ys = [point[1] for point in item['points']]
            yerr = [point[2] for point in item['points']]
            ax.errorbar(xs, ys, yerr=yerr, marker='o', capsize=3, label=item['label'])

    _plot(title, xlabel, ylabel, out_path, draw)


def plot_error_vs_d(series, title, out_path,
                    xlabel='Exact full-network D (all checkpoints)'):
    if not series:
        print(f"Skipping {os.path.basename(out_path)}: no D values yet. "
              f"Pass --data_dir to score all checkpoints, or reuse the D cache.")
        return

    def draw(ax):
        for item in series:
            xs = [point[1] for point in item['points']]
            ys = [point[3] for point in item['points']]
            xerr = [point[2] for point in item['points']]
            yerr = [point[4] for point in item['points']]
            ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, marker='o', capsize=3,
                        label=item['label'])

    _plot(title, xlabel, 'Mean test error (%)', out_path, draw)


def score_missing(rows, args, scoring, cache_path, logger):
    """Fill ``row['d']`` for subsets not already in the cache. Writes the cache after each seed."""
    import torch

    from .. import common
    from .. import objective as d_alpha
    from .. import trajectory as ref

    missing = [row for row in rows
               if row.get('indices_path') and row.get('d') is None]
    if not missing:
        logger.info("Exact D cache already covers every saved subset.")
        return

    cfg = common.get_dataset_config(
        args.dataset, args.data_dir, download=args.download,
        architecture=args.architecture)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    template = args.checkpoint_dir or os.path.join(
        tree(args.scratch, 'data_selection_swap', args.variant),
        args.dataset, cfg['architecture'], 'checkpoints')
    by_seed = {}
    for row in missing:
        by_seed.setdefault(row['seed'], []).append(row)

    for seed, seed_rows in sorted(by_seed.items()):
        trainset, _, _, n_train = common.make_loaders(
            cfg, seed, args.batch_size, args.num_workers, args.scoring_transform)
        pool_hash = common.ordered_indices_fingerprint(trainset.indices)
        checkpoint_dir = locate_checkpoint_dir(template, seed)
        checkpoints, _, description = ref.resolve(
            checkpoint_dir, optimizer=args.weighting,
            num_checkpoints=args.num_checkpoints,
            selection=args.checkpoint_selection,
            epochs=args.checkpoint_epochs, n_train=n_train,
            batch_size=args.batch_size, base_lr=args.base_lr,
            num_epochs=cfg['num_epochs'], beta1=args.beta1,
            momentum=args.momentum_response, metric=args.metric,
            checkpoint_weighting=args.checkpoint_weighting,
            expected_dataset=args.dataset, expected_seed=seed,
            expected_architecture=cfg.get('architecture'),
            expected_pool_indices_sha256=pool_hash,
            allow_manifest_mismatch=getattr(args, 'allow_manifest_mismatch', False))
        logger.info(f"Scoring seed {seed} on {len(checkpoints)} checkpoints "
                    f"({description}), {len(seed_rows)} subsets.")

        metrics = None
        if ref.resolve_metric(args.weighting, args.metric):
            named = list(common.build_model(
                cfg, cfg['num_classes'], 'cpu').named_parameters())
            metrics = [
                ref.adam_metric_flat(c, named, eps=args.adam_eps, power=2).to(device)
                for c in checkpoints]
            scale = max(float(M.mean()) for M in metrics)
            metrics = [M / scale for M in metrics]
            logger.info(f"Metric M_k = D_k^2 rescaled by 1/{scale:.4g}")

        scorer = d_alpha.TrajectoryScorer(
            cfg, trainset, [c.path for c in checkpoints],
            [c.weight for c in checkpoints], device,
            scoring_transform=args.scoring_transform,
            batch_size=args.batch_size, num_workers=args.num_workers,
            metrics=metrics, gbar_cache_dir=args.gbar_cache_dir,
            cache_tag=f"{args.dataset}_{cfg['architecture']}_seed{seed}_{args.scoring_transform}",
            logger=logger)
        subsets = {}
        for i, row in enumerate(seed_rows):
            idx = common.torch_load(row['indices_path'], map_location='cpu')
            subsets[i] = torch.as_tensor(idx, dtype=torch.long).reshape(-1)
        values = scorer.score(subsets)
        for i, row in enumerate(seed_rows):
            row['d'] = float(values[i])
        del scorer
        if device == 'cuda':
            torch.cuda.empty_cache()
        write_table(cache_path, rows, scoring, args.dataset, args.architecture)
        logger.info(f"Wrote D cache -> {cache_path}")


def _figure_path(out_dir, dataset, architecture, name, ext='png'):
    return os.path.join(out_dir, f"{dataset}_{architecture}_{name}.{ext}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='cifar10', choices=list(N_TRAIN))
    p.add_argument('--architecture', default='resnet18',
                   choices=['resnet18', 'vgg16', 'inception'])
    p.add_argument('--fractions', default=[0.1, 0.2, 0.4, 0.5], type=float, nargs='+')
    p.add_argument('--seeds', default=[42, 43, 44, 45, 46], type=int, nargs='+')
    p.add_argument('--methods', default=None, nargs='+',
                   help='Default: every KL variant and baseline found on disk.')
    p.add_argument('--scratch', default=default_results_root())
    p.add_argument('--variant', default='sgd', choices=['sgd', 'adam'],
                   help="Result tree. 'sgd' is the historical layout; 'adam' reads "
                        "and writes the adam/ subfolders.")
    p.add_argument('--out_dir', default=None,
                   help='Default: $SCRATCH/data_selection_baselines[/adam]/plots')

    p.add_argument('--data_dir', default=None,
                   help='Dataset root. Required the first time exact D is scored. '
                        'Omit it to redraw from the D cache only.')
    p.add_argument('--checkpoint_dir', default=None,
                   help='Reference trajectory. Default: '
                        '<results>/data_selection_swap/<dataset>/<architecture>/checkpoints '
                        'with a seed_<seed> subdirectory when that is where the run was saved. '
                        'A "{seed}" placeholder is substituted. Point this at seed_42 and pass '
                        '--allow_manifest_mismatch for the manuscript protocol.')
    p.add_argument('--allow_manifest_mismatch', action='store_true',
                   help='Score the seed-42 reference trajectory for every selection seed '
                        '(Appendix G.1).')
    p.add_argument('--skip_objective', action='store_true', default=False,
                   help='Do not score D, even if --data_dir is set. Plot 2 uses the cache.')
    p.add_argument('--num_checkpoints', default=0, type=int,
                   help='Checkpoints in the shared scoring trajectory. 0 = every checkpoint.')
    p.add_argument('--checkpoint_selection', default='spread', choices=['spread', 'prefix'])
    p.add_argument('--checkpoint_weighting', default='point', choices=['point', 'window'])
    p.add_argument('--checkpoint_epochs', default=None, type=int, nargs='+')
    p.add_argument('--weighting', '--optimizer', dest='weighting', default=None,
                   choices=['uniform', 'sgd', 'adam', 'adam_update'],
                   help='Step weights w_k. Default sgd, or adam when --variant adam. '
                        'M = I unless --metric adam (or an Adam weighting) says otherwise.')
    p.add_argument('--metric', default='auto', choices=['auto', 'none', 'adam'])
    p.add_argument('--momentum_response', default='persistent',
                   choices=['persistent', 'isolated'])
    p.add_argument('--base_lr', default=0.01, type=float)
    p.add_argument('--adam_eps', default=1e-8, type=float)
    p.add_argument('--beta1', default=0.9, type=float)
    p.add_argument('--scoring_transform', default='test', choices=['train', 'test'])
    p.add_argument('--batch_size', default=128, type=int)
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--gbar_cache_dir', default=None,
                   help='Where mean gradients are cached (~90 MB per checkpoint per seed).')
    p.add_argument('--cache_csv', default=None,
                   help='Per-subset D cache. Default: <out_dir>/d_exact_<dataset>_<architecture>.csv')
    p.add_argument('--download', action='store_true', default=False)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger('result_plots')

    if args.weighting is None:
        args.weighting = 'adam' if args.variant == 'adam' else 'sgd'
    out_dir = args.out_dir or os.path.join(
        tree(args.scratch, 'data_selection_baselines', args.variant), 'plots')
    cache_path = args.cache_csv or os.path.join(
        out_dir, f"d_exact_{args.dataset}_{args.architecture}.csv")
    if args.gbar_cache_dir is None:
        args.gbar_cache_dir = os.path.join(
            out_dir, f"gbar_{args.dataset}_{args.architecture}")
    scoring = scoring_id(
        args.dataset, args.architecture, args.weighting, args.metric,
        args.num_checkpoints, args.checkpoint_selection, args.checkpoint_weighting,
        args.scoring_transform, args.checkpoint_epochs)

    rows = collect_runs(
        args.scratch, args.dataset, args.architecture, args.fractions,
        args.seeds, args.methods, variant=args.variant)
    if not rows:
        logger.error(f"No runs found under {args.scratch} for "
                     f"{args.dataset}/{args.architecture}.")
        return
    logger.info(f"Found {len(rows)} saved runs.")
    attach_cached_d(rows, cache_path, scoring)

    if args.data_dir and not args.skip_objective:
        score_missing(rows, args, scoring, cache_path, logger)
    write_table(cache_path, rows, scoring, args.dataset, args.architecture)
    logger.info(f"Per-subset table -> {cache_path}")
    logger.info(f"LaTeX table -> {latex_path_for(cache_path)}")

    prefix = f"{args.dataset} / {args.architecture}"
    kl_rows = [row for row in rows if is_kl_method(row['method'])]
    if kl_rows:
        kl_tex = latex_path_for(cache_path, '_kl')
        write_latex(kl_tex, kl_rows, args.dataset, args.architecture, kl_only=True)
        logger.info(f"KL-only LaTeX table -> {kl_tex}")
        plot_accuracy_vs_fraction(
            k_accuracy_series(with_accuracy(kl_rows)),
            _figure_path(out_dir, args.dataset, args.architecture,
                         'kl_accuracy_vs_fraction', ext='pdf'),
            emphasize=emphasizes_k1)
    else:
        logger.info("No KL runs found; skipping the KL-only plot and table.")

    plot_accuracy_vs_fraction(
        comparison_accuracy_series(baseline_comparison_rows(with_accuracy(rows))),
        _figure_path(out_dir, args.dataset, args.architecture,
                     'accuracy_vs_fraction', ext='pdf'),
        emphasize=KL_DISPLAY_NAME,
        exclude_from_limits=('el2n',))
    plot_error_vs_d(
        group_error_vs_d(rows),
        f"{prefix}: test error vs exact D",
        _figure_path(out_dir, args.dataset, args.architecture, 'error_vs_d'))
    plot_vs_budget(
        group_by_budget(rows, 'selection_s'),
        f"{prefix}: selection time vs budget",
        'Budget (selected points)', 'Selection time (s)',
        _figure_path(out_dir, args.dataset, args.architecture, 'time_vs_budget'))


if __name__ == '__main__':
    main()
