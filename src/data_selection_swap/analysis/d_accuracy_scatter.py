#!/usr/bin/env python3
"""Scatter exact D(alpha) against final test accuracy, one point per saved run.

Input is the per-subset table written by ``analysis.score_subsets``: one row
per ``(dataset, architecture, fraction, method, K, seed)``. Unlike
``analysis.result_plots``, nothing is averaged here — every row becomes its own
marker, so one figure shows all (seed, selection method) combinations and the
spread across seeds is visible rather than collapsed into an error bar.

One figure per ``(dataset, architecture, fraction)``. Fixing the fraction is
what makes the x axis meaningful: D falls roughly like 1/m, so subsets of
different budgets do not share a scale and must not be pooled into one panel.

Reported correlation
────────────────────
Points from one seed share both the 90/10 split and the reference trajectory,
which gives each seed its own offset in D and in accuracy. A single Spearman
over every point would therefore partly measure that offset. So the figure is
annotated with the Spearman computed within each seed across methods (mean and
standard deviation over seeds), plus a pooled value computed after centring
both quantities inside each seed. Negative means lower D goes with higher
accuracy, i.e. the objective ranks subsets the way accuracy does.

The random draws that ``score_subsets`` uses for E_U[D] have no accuracy, so
they are excluded from the points and drawn as a vertical reference band
instead. Their spread is the noise floor: D differences between methods that
fall inside it are not meaningful.

Usage:
    python scripts/plot/d_accuracy_scatter.py \\
        --input $SCRATCH/data_selection_baselines/d_exact_cifar10_resnet18_f0.1.csv
    python scripts/plot/d_accuracy_scatter.py \\
        --input $SCRATCH/data_selection_baselines --annotate_seed
"""

import argparse
import csv
import glob
import logging
import os
import sys

import numpy as np
from scipy.stats import spearmanr

from .comparison import RANDOM_METHOD
from .result_plots import is_kl_method

KL_MARKER = 'D'
BASELINE_MARKER = 'o'


# ─────────────────────────────────────────────────────────────────────────────
# Reading the per-subset tables
# ─────────────────────────────────────────────────────────────────────────────
def expand_inputs(inputs):
    """Accept files, directories, and unexpanded globs; return CSV paths."""
    paths = []
    for raw in inputs:
        matches = glob.glob(raw)
        if not matches and not os.path.exists(raw):
            raise FileNotFoundError(f"No such file, directory, or glob match: {raw}")
        for match in matches or [raw]:
            if os.path.isdir(match):
                paths.extend(sorted(glob.glob(os.path.join(match, '*.csv'))))
            else:
                paths.append(match)
    unique = sorted({os.path.realpath(p) for p in paths})
    if not unique:
        raise FileNotFoundError(f"No CSV files found in {inputs}")
    return unique


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_rows(paths, dataset=None, architecture=None):
    """Rows of the per-subset tables, keeping those with a finite D.

    ``dataset``/``architecture`` are fallbacks for tables written before those
    columns existed; a row that identifies neither is refused rather than
    silently grouped with another configuration.
    """
    rows = []
    for path in paths:
        # utf-8-sig drops a byte-order mark if the table has been round-tripped
        # through a spreadsheet; without it the BOM lands in the first fieldname.
        with open(path, newline='', encoding='utf-8-sig') as fh:
            reader = csv.DictReader(fh)
            fields = set(reader.fieldnames or ())
            if not {'acc', 'd_alpha'} <= fields:
                raise ValueError(
                    f"{path} is not a per-subset table from "
                    f"scripts/analyze/score_subsets.py: it has no 'acc' and 'd_alpha' "
                    f"columns. The similarly named analysis.result_plots cache stores "
                    f"epoch-averaged 'error' instead of a final accuracy.")
            for record in reader:
                d = _float(record.get('d_alpha'))
                fraction = _float(record.get('fraction'))
                if d is None or not np.isfinite(d) or fraction is None:
                    continue
                row_dataset = record.get('dataset') or dataset
                row_architecture = record.get('architecture') or architecture
                if not row_dataset or not row_architecture:
                    raise ValueError(
                        f"{path} has no dataset/architecture column; pass --dataset "
                        f"and --architecture for tables written before they existed.")
                rows.append(dict(
                    dataset=row_dataset,
                    architecture=row_architecture,
                    fraction=fraction,
                    budget=_float(record.get('budget')),
                    method=record.get('method', ''),
                    k=record.get('num_checkpoints_tag') or None,
                    seed=record.get('seed', ''),
                    acc=_float(record.get('acc')),
                    d=d,
                    e_u_d=_float(record.get('e_u_d')),
                    trajectory=record.get('scoring_trajectory', ''),
                    source=path,
                ))
    if not rows:
        raise ValueError(f"No scored rows found in {paths}")
    return rows


def group_by_panel(rows):
    """``(dataset, architecture, fraction)`` -> rows, sorted for stable output."""
    panels = {}
    for row in rows:
        key = (row['dataset'], row['architecture'], row['fraction'])
        panels.setdefault(key, []).append(row)
    return dict(sorted(panels.items()))


def split_points_and_reference(panel_rows):
    """Separate the scatter points from the random-draw reference rows.

    A point needs both coordinates, so rows without an accuracy (the random
    draws, and any run made with --selection_only) cannot be plotted.
    """
    points = [row for row in panel_rows
              if row['method'] != RANDOM_METHOD
              and row['acc'] is not None and np.isfinite(row['acc'])]
    reference = [row for row in panel_rows if row['method'] == RANDOM_METHOD]
    points.sort(key=lambda row: (row['method'], str(row['k']), str(row['seed'])))
    return points, reference


def series_label(method, k):
    return method if k in (None, '') else f"{method}, K={k}"


def group_by_series(points):
    """``(method, K)`` -> its points, so the legend has one entry per method."""
    series = {}
    for row in points:
        series.setdefault((row['method'], row['k']), []).append(row)
    return dict(sorted(series.items(), key=lambda item: (
        not is_kl_method(item[0][0]), item[0][0], str(item[0][1]))))


def trajectory_label(panel_rows):
    """The scoring trajectory shared by the panel, with the per-seed suffix dropped."""
    descriptions = set()
    for row in panel_rows:
        text = row['trajectory'] or ''
        descriptions.add(text.split(', seed=')[0].strip())
    descriptions.discard('')
    if len(descriptions) == 1:
        return descriptions.pop()
    return ''


# ─────────────────────────────────────────────────────────────────────────────
# Correlation
# ─────────────────────────────────────────────────────────────────────────────
def _spearman(a, b):
    """Spearman rho, or NaN where it is undefined rather than merely small."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        return float('nan')
    rho = spearmanr(a, b).statistic
    return float(rho) if np.isfinite(rho) else float('nan')


def correlation(points, min_group=3):
    """Within-seed Spearman across methods, plus the seed-centred pooled value.

    A seed is skipped when it has too few methods, or when its D or its
    accuracy is constant across them: such a group carries no ranking
    information for either estimator. ``n_seeds`` counts those that did
    contribute.
    """
    within, pooled_d, pooled_acc = [], [], []
    for seed in sorted({row['seed'] for row in points}):
        group = [row for row in points if row['seed'] == seed]
        if len(group) < min_group:
            continue
        d = np.array([row['d'] for row in group], dtype=np.float64)
        acc = np.array([row['acc'] for row in group], dtype=np.float64)
        rho = _spearman(d, acc)
        if not np.isfinite(rho):
            continue
        within.append(rho)
        pooled_d.extend(d - d.mean())
        pooled_acc.extend(acc - acc.mean())
    return dict(
        within_mean=float(np.mean(within)) if within else float('nan'),
        within_std=(float(np.std(within, ddof=1 if len(within) > 1 else 0))
                    if within else float('nan')),
        n_seeds=len(within),
        pooled_centred=_spearman(pooled_d, pooled_acc),
        n_points=len(pooled_d),
    )


def correlation_text(stats):
    if not stats['n_seeds']:
        return ("no correlation defined: no seed has 3+ methods with both "
                "D and accuracy varying")
    return (f"Spearman(D, accuracy)\n"
            f"within seed: {stats['within_mean']:+.3f} ± {stats['within_std']:.3f} "
            f"({stats['n_seeds']} seeds)\n"
            f"pooled, seed-centred: {stats['pooled_centred']:+.3f} "
            f"({stats['n_points']} points)\n"
            f"negative supports the objective")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def figure_name(dataset, architecture, fraction):
    return f"{dataset}_{architecture}_f{fraction:g}_d_vs_accuracy.png"


def plot_panel(panel_rows, out_path, annotate_seed=False):
    """One figure: a marker per (seed, method), coloured by method."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dataset = panel_rows[0]['dataset']
    architecture = panel_rows[0]['architecture']
    fraction = panel_rows[0]['fraction']
    points, reference = split_points_and_reference(panel_rows)
    if not points:
        print(f"Skipping {os.path.basename(out_path)}: no row has both D and accuracy.")
        return None

    series = group_by_series(points)
    stats = correlation(points)
    cmap = plt.get_cmap('tab20' if len(series) > 10 else 'tab10')
    colors = [cmap(i % cmap.N) for i in range(len(series))]

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    for color, ((method, k), rows) in zip(colors, series.items()):
        ax.scatter([row['d'] for row in rows], [row['acc'] for row in rows],
                   s=46, alpha=0.85, color=color, edgecolor='black', linewidth=0.4,
                   marker=KL_MARKER if is_kl_method(method) else BASELINE_MARKER,
                   label=f"{series_label(method, k)} ({len(rows)} seeds)")
        if annotate_seed:
            for row in rows:
                ax.annotate(str(row['seed']), (row['d'], row['acc']),
                            textcoords='offset points', xytext=(4, 3), fontsize=6)

    random_d = [row['d'] for row in reference if np.isfinite(row['d'])]
    if random_d:
        ax.axvspan(min(random_d), max(random_d), color='grey', alpha=0.13,
                   label=f"random draws, D range (n={len(random_d)})")
        ax.axvline(float(np.mean(random_d)), color='grey', ls='--', lw=1.2,
                   label='random draws, mean D')

    if all(row['d'] > 0.0 for row in points) and (not random_d or min(random_d) > 0.0):
        ax.set_xscale('log')

    budget = points[0]['budget']
    budget_text = f", budget {int(budget)}" if budget else ''
    ax.set_title(f"{dataset} / {architecture}: fraction {fraction:g}{budget_text}")
    trajectory = trajectory_label(panel_rows)
    ax.set_xlabel('Exact all-parameter D(alpha)'
                  + (f"\n{trajectory}" if trajectory else ''))
    ax.set_ylabel('Final test accuracy (%)')
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.02, correlation_text(stats), transform=ax.transAxes,
            fontsize=8, va='bottom', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, lw=0.5))
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return stats


def plot_all(rows, out_dir, annotate_seed=False, logger=None):
    """One figure per (dataset, architecture, fraction). Returns their stats."""
    results = {}
    for key, panel_rows in group_by_panel(rows).items():
        dataset, architecture, fraction = key
        out_path = os.path.join(out_dir, figure_name(dataset, architecture, fraction))
        stats = plot_panel(panel_rows, out_path, annotate_seed=annotate_seed)
        if stats is None:
            continue
        results[key] = stats
        if logger:
            points, _ = split_points_and_reference(panel_rows)
            logger.info(
                f"{dataset}/{architecture} fraction={fraction:g}: {len(points)} points, "
                f"within-seed rho={stats['within_mean']:+.3f} ± {stats['within_std']:.3f} "
                f"over {stats['n_seeds']} seeds, pooled centred rho="
                f"{stats['pooled_centred']:+.3f} -> {out_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', required=True, nargs='+',
                   help='Per-subset CSVs from scripts/analyze/score_subsets.py. '
                        'Files, directories, or globs.')
    p.add_argument('--out_dir', default=None,
                   help='Default: $SCRATCH/data_selection_baselines/plots')
    p.add_argument('--scratch', default=os.environ.get('SCRATCH', '/home/mila/a/ahmedm/scratch'))
    p.add_argument('--dataset', default=None,
                   help='Fallback for tables written without a dataset column.')
    p.add_argument('--architecture', default=None,
                   help='Fallback for tables written without an architecture column.')
    p.add_argument('--annotate_seed', action='store_true', default=False,
                   help='Label every marker with its seed, which makes the '
                        'per-seed clustering visible.')
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger('d_accuracy_scatter')

    out_dir = args.out_dir or os.path.join(
        args.scratch, 'data_selection_baselines', 'plots')
    paths = expand_inputs(args.input)
    logger.info(f"Reading {len(paths)} per-subset table(s).")
    rows = read_rows(paths, args.dataset, args.architecture)
    results = plot_all(rows, out_dir, annotate_seed=args.annotate_seed, logger=logger)
    if not results:
        logger.error("No figure drawn: no row had both a D value and an accuracy. "
                     "Runs made with --selection_only have no accuracy to pair.")
        return
    logger.info(f"Wrote {len(results)} figure(s) -> {out_dir}")


if __name__ == '__main__':
    main()
