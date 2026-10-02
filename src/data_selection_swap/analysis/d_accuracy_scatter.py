#!/usr/bin/env python3
"""Scatter sqrt(D(alpha)) against final test accuracy (journal-ready figure).

Input is the per-subset table written by ``analysis.score_subsets``: one row
per ``(dataset, architecture, fraction, method, K, seed)``.

One figure per ``(dataset, architecture, fraction)``. Fixing the fraction is
what makes the x axis meaningful: D falls roughly like 1/m, so subsets of
different budgets do not share a scale and must not be pooled into one panel.

Figure
──────
A plain scatter: one marker per (seed, selection method, K), x = sqrt(D),
y = final test accuracy. The grey band/dashed line mark the range/mean of the
random draws (the noise floor).

The x axis is sqrt(D). This is a monotone transform, so every Spearman value
is identical to the one computed on D itself.

Reported correlation
────────────────────
Nothing statistical is drawn on the figure. The Spearman values (within-seed
mean and std over seeds, plus the seed-centred pooled value) and the
colour/marker of every series are printed to the terminal, to be reported in
the text or caption. The legend is off by default (--legend turns it on).

The random draws used for E_U[D] have no accuracy, so they are drawn as a
vertical band (noise floor) instead of points.

Usage:
    python scripts/plot/d_accuracy_scatter.py \\
        --input $SCRATCH/data_selection_baselines/d_exact_cifar10_vgg16_f0.1.csv
    python scripts/plot/d_accuracy_scatter.py --input DIR --width double \\
        --formats pdf png --no_title
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
                    x=float(np.sqrt(max(d, 0.0))),   # plotted coordinate: sqrt(D)
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
    """Separate the scatter points from the random-draw reference rows."""
    points = [row for row in panel_rows
              if row['method'] != RANDOM_METHOD
              and row['acc'] is not None and np.isfinite(row['acc'])]
    reference = [row for row in panel_rows if row['method'] == RANDOM_METHOD]
    points.sort(key=lambda row: (row['method'], str(row['k']), str(row['seed'])))
    return points, reference


def _k_key(k):
    """Sort K numerically (1, 4, 10, 15, 20) instead of as strings."""
    try:
        return (0, int(k))
    except (TypeError, ValueError):
        return (1, 0)


def group_by_series(points):
    """``(method, K)`` -> its points: KL methods first, K ascending."""
    series = {}
    for row in points:
        series.setdefault((row['method'], row['k']), []).append(row)
    return dict(sorted(series.items(), key=lambda item: (
        not is_kl_method(item[0][0]), item[0][0], _k_key(item[0][1]))))


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
# Correlation (invariant to the sqrt transform: Spearman is rank-based)
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
    """Within-seed Spearman across methods, plus the seed-centred pooled value."""
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



# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
WIDTHS_IN = {'single': 3.4, 'onehalf': 5.2, 'double': 7.2}   # typical journal columns

# Colour-blind-safe baseline colours (Okabe-Ito subset, no blues: those are KL's).
BASELINE_COLORS = ['#E69F00', '#009E73', '#D55E00', '#CC79A7', '#8C564B', '#000000']
BASELINE_MARKERS = ['o', 's', '^', 'v', 'P', 'X', 'h']
RANDOM_COLOR = '#7F7F7F'

RC = {
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.7,
    'pdf.fonttype': 42,   # embed TrueType fonts (journals reject Type 3)
    'ps.fonttype': 42,
}


def series_legend_label(method, k, kl_label):
    if is_kl_method(method):
        return kl_label if k in (None, '') else f"{kl_label}, $K$={k}"
    return method.replace('_', ' ')


def assign_styles(series, kl_label, plt):
    """``(method, K)`` -> dict(color, marker, label)."""
    kl_keys = [key for key in series if is_kl_method(key[0])]
    other_keys = [key for key in series if not is_kl_method(key[0])]
    blues = plt.get_cmap('Blues')
    kl_shades = np.linspace(0.45, 0.95, max(len(kl_keys), 1))

    styles = {}
    for key, shade in zip(kl_keys, kl_shades):
        styles[key] = dict(color=blues(shade), marker='D')
    i = 0
    for key in other_keys:
        if key[0] in (RANDOM_METHOD, 'random'):
            styles[key] = dict(color=RANDOM_COLOR, marker='o')
            continue
        styles[key] = dict(color=BASELINE_COLORS[i % len(BASELINE_COLORS)],
                           marker=BASELINE_MARKERS[i % len(BASELINE_MARKERS)])
        i += 1
    for key in series:
        styles[key]['label'] = series_legend_label(key[0], key[1], kl_label)
    return styles


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def figure_stem(dataset, architecture, fraction):
    return f"{dataset}_{architecture}_f{fraction:g}_sqrtd_vs_accuracy"


def draw_series(ax, series, styles):
    """One plain marker per (seed, method, K)."""
    for key, rows in series.items():
        st = styles[key]
        ax.scatter([r['x'] for r in rows], [r['acc'] for r in rows],
                   s=22, color=st['color'], marker=st['marker'],
                   edgecolor='black', linewidth=0.4, alpha=0.9, zorder=3)


def draw_reference(ax, random_x):
    if not random_x:
        return
    ax.axvspan(min(random_x), max(random_x), color='0.5', alpha=0.15, lw=0, zorder=0)
    ax.axvline(float(np.mean(random_x)), color='0.4', ls='--', lw=0.8, zorder=1)


def plot_panel(panel_rows, out_stem, width='double', formats=('pdf',), title=True,
               kl_label='KL', xscale='log', legend=False):
    """One scatter figure.

    Returns ``(stats, entries)``: the correlation stats and, per series,
    ``(label, hex colour, marker)`` for the caption. ``(None, None)`` if nothing
    could be drawn.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    dataset = panel_rows[0]['dataset']
    architecture = panel_rows[0]['architecture']
    fraction = panel_rows[0]['fraction']
    points, reference = split_points_and_reference(panel_rows)
    if not points:
        print(f"Skipping {os.path.basename(out_stem)}: no row has both D and accuracy.")
        return None, None

    series = group_by_series(points)
    stats = correlation(points)
    random_x = [r['x'] for r in reference if np.isfinite(r['x'])]

    with plt.rc_context(RC):
        w = WIDTHS_IN[width]
        styles = assign_styles(series, kl_label, plt)
        fig, ax = plt.subplots(figsize=(w, 0.62 * w))

        draw_reference(ax, random_x)
        draw_series(ax, series, styles)
        if xscale == 'log' and all(r['x'] > 0 for r in points) \
                and (not random_x or min(random_x) > 0):
            ax.set_xscale('log')
        ax.set_xlabel(r'$\sqrt{D(\alpha)}$')
        ax.set_ylabel('Test accuracy (%)')
        ax.grid(alpha=0.25, lw=0.4)

        if title:
            budget = points[0]['budget']
            budget_text = f", budget {int(budget)}" if budget else ''
            ax.set_title(f"{dataset} / {architecture}, fraction {fraction:g}"
                         f"{budget_text}")

        fig.tight_layout()
        if legend:
            handles = [Line2D([], [], ls='', marker=styles[k]['marker'], ms=5,
                              mfc=styles[k]['color'], mec='black', mew=0.5,
                              label=styles[k]['label']) for k in series]
            if random_x:
                handles.append(Patch(facecolor='0.5', alpha=0.25,
                                     label=f"random draws, range ($n$={len(random_x)})"))
                handles.append(Line2D([], [], color='0.4', ls='--', lw=0.8,
                                      label='random draws, mean'))
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.0),
                       ncol=4 if width != 'single' else 2, frameon=False,
                       columnspacing=1.2, handletextpad=0.4)

        os.makedirs(os.path.dirname(out_stem) or '.', exist_ok=True)
        for fmt in formats:
            fig.savefig(f"{out_stem}.{fmt}", dpi=600 if fmt != 'pdf' else None,
                        bbox_inches='tight')
        plt.close(fig)
    entries = [(styles[k]['label'].replace('$', ''), to_hex(styles[k]['color']),
                styles[k]['marker']) for k in series]
    return stats, entries


def plot_all(rows, out_dir, logger, **kwargs):
    """One figure per (dataset, architecture, fraction). Returns their stats."""
    results = {}
    for key, panel_rows in group_by_panel(rows).items():
        dataset, architecture, fraction = key
        out_stem = os.path.join(out_dir, figure_stem(dataset, architecture, fraction))
        stats, entries = plot_panel(panel_rows, out_stem, **kwargs)
        if stats is None:
            continue
        results[key] = stats
        points, _ = split_points_and_reference(panel_rows)
        logger.info(
            f"\n=== {dataset} / {architecture}, fraction {fraction:g} "
            f"({len(points)} points) -> {out_stem}\n"
            f"Spearman(D, accuracy)  [identical for sqrt(D); negative = lower D "
            f"goes with higher accuracy]\n"
            f"  within seed : {stats['within_mean']:+.3f} ± {stats['within_std']:.3f} "
            f"(over {stats['n_seeds']} seeds)\n"
            f"  pooled, seed-centred : {stats['pooled_centred']:+.3f} "
            f"({stats['n_points']} points)")
        trajectory = trajectory_label(panel_rows)
        if trajectory:
            logger.info(f"Scoring trajectory: {trajectory}")
        logger.info("Series in the figure (label | colour | marker):\n" + "\n".join(
            f"  {label:<40s} {color}  {marker}" for label, color, marker in entries))
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
    p.add_argument('--legend', action='store_true', default=False,
                   help='Draw a legend below the figure (off by default).')
    p.add_argument('--width', choices=list(WIDTHS_IN), default='double',
                   help='Figure width: single (3.4in), onehalf (5.2in), double (7.2in).')
    p.add_argument('--formats', nargs='+', default=['pdf'],
                   help='Output formats, e.g. pdf png. Default: pdf (vector).')
    p.add_argument('--no_title', action='store_true', default=False,
                   help='Omit the figure title (use the caption instead).')
    p.add_argument('--kl_label', default='KL',
                   help='Legend name for the KL-based selection method.')
    p.add_argument('--xscale', choices=['log', 'linear'], default='log',
                   help='x axis scale.')
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
    results = plot_all(rows, out_dir, logger, legend=args.legend, width=args.width,
                       formats=tuple(args.formats), title=not args.no_title,
                       kl_label=args.kl_label, xscale=args.xscale)
    if not results:
        logger.error("No figure drawn: no row had both a D value and an accuracy. "
                     "Runs made with --selection_only have no accuracy to pair.")
        return
    logger.info(f"Wrote {len(results)} figure(s) -> {out_dir}")


if __name__ == '__main__':
    main()