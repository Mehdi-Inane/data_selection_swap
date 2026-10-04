#!/usr/bin/env python3
"""Scatter sqrt(G_T(I) / 2) against the risk gap R_P(A(S_I)) - R_P(A(S)).

Input is the per-subset table written by ``analysis.score_subsets``: one row
per ``(dataset, architecture, fraction, method, K, seed)``. G_T(I) is stored in
the ``d_alpha`` column. The figure checks the guarantee

    | E_{theta~A_n(S)}[q(theta)] - E_{theta~A_m(S_I)}[q(theta)] |
        <= (b - a) * sqrt(G_T(I) / 2)

with q = test accuracy expressed as a fraction in [0, 1], so b - a = 1 and the
bound reads  y <= x  for  x = sqrt(G_T(I) / 2). The y coordinate is

    y = | acc_full - acc_subset |      (a fraction, not percent).

``acc_full`` is read from the full-data trajectory's ``metrics.json``
(``final_test_acc``, stored as a fraction). Subset accuracies in the scored
table are in percent and are converted. Each run contributes a single draw of
theta, so y is a one-sample estimate of the left-hand side.

One figure per ``(dataset, architecture, fraction)``. Fixing the fraction is
what makes the x axis meaningful: G falls roughly like 1/m, so subsets of
different budgets do not share a scale and must not be pooled into one panel.

Figure
──────
A single scatter with one marker per run (method, K, seed): NO seed averaging,
no legend, one colour. An inset zooms on the densest zone of the cloud and is
outlined by a dotted box on the main axes. Pearson and Spearman correlations
are written in the corner opposite the inset and printed to the terminal.

Correlations
────────────
Both are computed on the plotted coordinates (x = sqrt(G/2), y = gap):
  * Spearman is rank-based, so it is identical for G and sqrt(G/2).
  * Pearson is NOT invariant to the sqrt, so it is computed on sqrt(G/2).
Three variants are reported: pooled over every point; within seed (computed
per seed, then mean ± std over seeds); and pooled after centring x and y inside
each seed. No p-values are given: runs sharing a seed or a method are not
independent, so the naive p-values would be overconfident.

Usage:
    python scripts/plot/d_accuracy_scatter.py \\
        --input $SCRATCH/data_selection_baselines/d_exact_*_f0.1.csv
    python scripts/plot/d_accuracy_scatter.py --input DIR --width double \\
        --formats pdf png --no_title --zoom_frac 0.5
"""

import argparse
import csv
import glob
import json
import logging
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

from .comparison import RANDOM_METHOD


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
                # Only the scorer's tables. The same directory also holds
                # comparison_*.csv and other aggregators that fail the schema.
                paths.extend(sorted(glob.glob(os.path.join(match, 'd_exact_*.csv'))))
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


def as_percent(value):
    """Convert a fraction in [0, 1] to percent; leave values already in percent."""
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"accuracy must be finite, got {value}")
    if 0.0 <= value <= 1.0:
        return 100.0 * value
    return value


def full_metrics_candidates(scratch, dataset, architecture, seed=None, template=None):
    """Locations of the full-pool trajectory's metrics.json, most specific last.

    The user-facing convention is
    ``$SCRATCH/data_selection_swap/{dataset}/{architecture}/checkpoints/metrics.json``.
    The maintained launcher also writes a per-seed copy under ``seed_{seed}/``.
    """
    if template:
        return [template.format(
            scratch=scratch, dataset=dataset, architecture=architecture,
            seed=seed if seed is not None else '')]
    root = os.path.join(scratch, 'data_selection_swap', dataset, architecture, 'checkpoints')
    paths = [os.path.join(root, 'metrics.json')]
    if seed not in (None, ''):
        paths.append(os.path.join(root, f'seed_{seed}', 'metrics.json'))
    return paths


def load_full_accuracy(path):
    """Official full-pool test accuracy, in percent.

    The trajectory file stores ``final_test_acc`` as a fraction (e.g. 0.7549).
    The last ``val_acc`` entry is used only if that key is missing.
    """
    with open(path, encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a metrics object")
    raw = payload.get('final_test_acc')
    if raw is None:
        val = payload.get('val_acc')
        if isinstance(val, list) and val:
            raw = val[-1]
    if raw is None:
        raise ValueError(f"{path} has no final_test_acc (or val_acc) field")
    return as_percent(raw)


def resolve_full_accuracy(scratch, dataset, architecture, seed=None, template=None,
                          cache=None):
    """Return (accuracy_percent, path used). Prefer a per-seed file when both exist."""
    cache = {} if cache is None else cache
    key = (dataset, architecture, str(seed), template)
    if key in cache:
        return cache[key]
    tried = []
    found = None
    for path in full_metrics_candidates(scratch, dataset, architecture, seed, template):
        tried.append(path)
        if os.path.isfile(path):
            found = path
    if found is None:
        raise FileNotFoundError(
            "No full-data metrics.json for "
            f"dataset={dataset} architecture={architecture} seed={seed}. "
            "Expected one of: " + "; ".join(tried))
    result = (load_full_accuracy(found), found)
    cache[key] = result
    return result


def attach_accuracy_gap(rows, scratch, template=None):
    """Set ``y = |acc_full - acc_subset|`` (a fraction, so b - a = 1) on every row."""
    cache = {}
    for row in rows:
        if row['acc'] is None or not np.isfinite(row['acc']):
            row['y'] = None
            row['full_acc'] = None
            continue
        full_acc, path = resolve_full_accuracy(
            scratch, row['dataset'], row['architecture'], row['seed'],
            template=template, cache=cache)
        row['full_acc'] = full_acc
        row['full_metrics_path'] = path
        row['y'] = abs(full_acc - row['acc']) / 100.0   # percent -> fraction
    return rows


def read_rows(paths, dataset=None, architecture=None):
    """Rows of the per-subset tables, keeping those with a finite G (``d_alpha``).

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
                    x=float(np.sqrt(max(d, 0.0) / 2.0)),   # plotted: sqrt(G_T(I) / 2)
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


def select_points(panel_rows, include_random=True):
    """Every run with a finite x and y. Random draws are points like any other."""
    points = [row for row in panel_rows
              if row.get('y') is not None and np.isfinite(row['y'])
              and np.isfinite(row['x'])
              and (include_random or row['method'] != RANDOM_METHOD)]
    points.sort(key=lambda row: (row['method'], str(row['k']), str(row['seed'])))
    return points


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
# Correlations
# ─────────────────────────────────────────────────────────────────────────────
def _degenerate(a, b):
    return a.size < 3 or np.ptp(a) == 0.0 or np.ptp(b) == 0.0


def _pearson(a, b):
    """Pearson r, or NaN where it is undefined rather than merely small."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if _degenerate(a, b):
        return float('nan')
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else float('nan')


def _spearman(a, b):
    """Spearman rho, or NaN where it is undefined rather than merely small."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if _degenerate(a, b):
        return float('nan')
    rho = spearmanr(a, b).statistic
    return float(rho) if np.isfinite(rho) else float('nan')


def _mean_std(values):
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return float('nan'), float('nan'), 0
    return (float(np.mean(values)),
            float(np.std(values, ddof=1 if len(values) > 1 else 0)),
            len(values))


def correlation(points, min_group=3):
    """Pearson and Spearman of x = sqrt(G/2) against y = risk gap.

    ``pooled``: all points together. ``within``: per seed, then mean ± std over
    seeds. ``centred``: pooled after subtracting each seed's mean of x and y.
    """
    x = np.array([r['x'] for r in points], dtype=np.float64)
    y = np.array([r['y'] for r in points], dtype=np.float64)
    seeds = np.array([str(r['seed']) for r in points])

    within_p, within_s, cx, cy = [], [], [], []
    for seed in sorted(set(seeds)):
        m = seeds == seed
        if m.sum() < min_group:
            continue
        within_p.append(_pearson(x[m], y[m]))
        within_s.append(_spearman(x[m], y[m]))
        cx.extend(x[m] - x[m].mean())
        cy.extend(y[m] - y[m].mean())

    wp_mean, wp_std, wp_n = _mean_std(within_p)
    ws_mean, ws_std, ws_n = _mean_std(within_s)
    return dict(
        n=len(points),
        pearson=_pearson(x, y), spearman=_spearman(x, y),
        within_pearson=wp_mean, within_pearson_std=wp_std, within_pearson_n=wp_n,
        within_spearman=ws_mean, within_spearman_std=ws_std, within_spearman_n=ws_n,
        centred_pearson=_pearson(cx, cy), centred_spearman=_spearman(cx, cy),
        n_centred=len(cx),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
WIDTHS_IN = {'single': 3.4, 'onehalf': 5.2, 'double': 7.2}   # typical journal columns
POINT_COLOR = '#2B6CB0'

RC = {
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'axes.linewidth': 0.7,
    'pdf.fonttype': 42,   # embed TrueType fonts (journals reject Type 3)
    'ps.fonttype': 42,
}

X_LABEL = r'$\sqrt{G_T(I)/2}$'
Y_LABEL = (r'$R_P\left(\mathcal{A}(S_{\widehat{I}})\right)'
           r' - R_P\left(\mathcal{A}(S)\right)$')

# Inset bounds in axes coordinates (x0, y0, w, h) and the corner of the
# correlation text (x, y, ha, va), placed opposite the inset. The bound y <= x
# keeps the upper-left of the plot mostly empty, hence the default.
INSET_LAYOUT = {
    'upper_left':  ([0.13, 0.52, 0.40, 0.42], (0.97, 0.04, 'right', 'bottom')),
    'upper_right': ([0.57, 0.52, 0.40, 0.42], (0.03, 0.04, 'left', 'bottom')),
    'lower_right': ([0.57, 0.12, 0.40, 0.42], (0.03, 0.96, 'left', 'top')),
    'lower_left':  ([0.13, 0.12, 0.40, 0.42], (0.97, 0.96, 'right', 'top')),
}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def figure_stem(dataset, architecture, fraction):
    return f"{dataset}_{architecture}_f{fraction:g}_riskgap_vs_sqrtG"


def densest_box(x, y, frac=0.6, radius=0.15, pad=0.08):
    """Bounds ``(x0, x1, y0, y1)`` of the densest zone, or None if zooming is pointless.

    Density is measured in rank space (so far-away outliers do not distort it):
    the centre is the point with the most neighbours within ``radius`` (Chebyshev
    distance on ranks scaled to [0, 1]); the box is the bounding box of the
    ``frac`` fraction of points closest to that centre, with a little padding.
    """
    n = len(x)
    if n < 6:
        return None
    u = rankdata(x) / n
    v = rankdata(y) / n
    dist = np.maximum(np.abs(u[:, None] - u[None, :]), np.abs(v[:, None] - v[None, :]))
    centre = int(np.argmax((dist <= radius).sum(axis=1)))
    k = max(3, int(np.ceil(frac * n)))
    keep = np.argsort(dist[centre])[:k]
    x0, x1 = x[keep].min(), x[keep].max()
    y0, y1 = y[keep].min(), y[keep].max()
    pad_x = pad * ((x1 - x0) or 1.0)
    pad_y = pad * ((y1 - y0) or 1.0)
    box = (max(x0 - pad_x, 0.0), x1 + pad_x, max(y0 - pad_y, 0.0), y1 + pad_y)
    inside = (x >= box[0]) & (x <= box[1]) & (y >= box[2]) & (y <= box[3])
    if inside.all():
        return None
    return box


def plot_panel(panel_rows, out_stem, width='double', formats=('pdf',), title=True,
               xscale='log', zoom_frac=0.6, zoom_box=None, inset_loc='upper_left',
               corr_text=True, include_random=True):
    """One scatter of every run, with a zoom inset and the correlations in a corner.

    Returns the correlation stats, or None if nothing could be drawn.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    dataset = panel_rows[0]['dataset']
    architecture = panel_rows[0]['architecture']
    fraction = panel_rows[0]['fraction']
    points = select_points(panel_rows, include_random)
    if not points:
        print(f"Skipping {os.path.basename(out_stem)}: no row has both G and accuracy.")
        return None

    x = np.array([r['x'] for r in points], dtype=np.float64)
    y = np.array([r['y'] for r in points], dtype=np.float64)
    stats = correlation(points)
    box = tuple(zoom_box) if zoom_box else densest_box(x, y, zoom_frac)
    inset_bounds, (tx, ty, tha, tva) = INSET_LAYOUT[inset_loc]
    scatter_kw = dict(s=18, color=POINT_COLOR, edgecolor='black', linewidth=0.4,
                      alpha=0.85, zorder=3)

    with plt.rc_context(RC):
        w = WIDTHS_IN[width]
        fig, ax = plt.subplots(figsize=(w, 0.7 * w))
        ax.scatter(x, y, **scatter_kw)
        ax.set_xlabel(X_LABEL, fontsize=16)
        ax.set_ylabel(Y_LABEL, fontsize=16)
        ax.grid(alpha=0.25, lw=0.4)

        if xscale == 'log' and np.all(x > 0):
            ax.set_xscale('log')
            # Plain numeric ticks (0.5, 1, 2, ...) instead of a lone "10^0".
            lo, hi = ax.get_xlim()
            ticks = [t for t in (0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5)
                     if lo <= t <= hi]
            if ticks:
                ax.set_xticks(ticks)
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.xaxis.set_minor_formatter(NullFormatter())

        if box:
            x0, x1, y0, y1 = box
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   ec='black', lw=0.7, ls=':', zorder=5))
            ax.set_xlim(xlim)   # the patch must not rescale the main axes
            ax.set_ylim(ylim)
            ins = ax.inset_axes(inset_bounds)
            ins.scatter(x, y, **scatter_kw)
            ins.set_xlim(x0, x1)
            ins.set_ylim(y0, y1)
            ins.tick_params(labelsize=11, length=3)
            ins.grid(alpha=0.25, lw=0.4)
            for spine in ins.spines.values():
                spine.set_linewidth(0.9)

        if corr_text:
            ax.text(tx, ty,
                    f"Pearson $r$ = {stats['pearson']:+.2f}\n"
                    f"Spearman $\\rho$ = {stats['spearman']:+.2f}",
                    transform=ax.transAxes, ha=tha, va=tva, fontsize=13, zorder=6,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.6',
                              lw=0.5, alpha=0.9))

        if title:
            budget = points[0]['budget']
            budget_text = f", budget {int(budget)}" if budget else ''
            ax.set_title(f"{dataset} / {architecture}, fraction {fraction:g}"
                         f"{budget_text}", fontsize=16)

        fig.tight_layout()
        os.makedirs(os.path.dirname(out_stem) or '.', exist_ok=True)
        for fmt in formats:
            fig.savefig(f"{out_stem}.{fmt}", dpi=600 if fmt != 'pdf' else None,
                        bbox_inches='tight')
        plt.close(fig)
    return stats


def plot_all(rows, out_dir, logger=None, **kwargs):
    """One figure per (dataset, architecture, fraction). Returns their stats."""
    if logger is None:
        logger = logging.getLogger('d_accuracy_scatter')
    include_random = kwargs.get('include_random', True)
    results = {}
    for key, panel_rows in group_by_panel(rows).items():
        dataset, architecture, fraction = key
        out_stem = os.path.join(out_dir, figure_stem(dataset, architecture, fraction))
        stats = plot_panel(panel_rows, out_stem, **kwargs)
        if stats is None:
            continue
        results[key] = stats
        points = select_points(panel_rows, include_random)
        logger.info(
            f"\n=== {dataset} / {architecture}, fraction {fraction:g} "
            f"({len(points)} points, one per run) -> {out_stem}\n"
            f"Correlation of x = sqrt(G/2) with y = |full - subset| gap "
            f"(positive = lower G goes with a smaller gap)\n"
            f"  pooled        : Pearson {stats['pearson']:+.3f} | "
            f"Spearman {stats['spearman']:+.3f}\n"
            f"  within seed   : Pearson {stats['within_pearson']:+.3f} ± "
            f"{stats['within_pearson_std']:.3f} | Spearman "
            f"{stats['within_spearman']:+.3f} ± {stats['within_spearman_std']:.3f} "
            f"(over {stats['within_pearson_n']} seeds)\n"
            f"  seed-centred  : Pearson {stats['centred_pearson']:+.3f} | "
            f"Spearman {stats['centred_spearman']:+.3f} "
            f"({stats['n_centred']} points)\n"
            f"  (Spearman is identical for G and sqrt(G/2); Pearson is on sqrt(G/2).)")
        violations = [r for r in points if r['y'] > r['x'] + 1e-12]
        logger.info(f"Bound y <= sqrt(G/2) (accuracy as a fraction, b-a=1): "
                    f"violated by {len(violations)} of {len(points)} points")
        trajectory = trajectory_label(panel_rows)
        if trajectory:
            logger.info(f"Scoring trajectory: {trajectory}")
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
    p.add_argument('--width', choices=list(WIDTHS_IN), default='double',
                   help='Figure width: single (3.4in), onehalf (5.2in), double (7.2in).')
    p.add_argument('--formats', nargs='+', default=['pdf'],
                   help='Output formats, e.g. pdf png. Default: pdf (vector).')
    p.add_argument('--no_title', action='store_true', default=False,
                   help='Omit the figure title (use the caption instead).')
    p.add_argument('--xscale', choices=['log', 'linear'], default='log',
                   help='x axis scale of the main axes (the inset is always linear).')
    p.add_argument('--zoom_frac', type=float, default=0.6,
                   help='The inset covers the bounding box of this fraction of the '
                        'points, centred on the densest zone.')
    p.add_argument('--zoom_box', type=float, nargs=4, default=None,
                   metavar=('X0', 'X1', 'Y0', 'Y1'),
                   help='Explicit inset bounds in data units (overrides --zoom_frac).')
    p.add_argument('--inset_loc', choices=list(INSET_LAYOUT), default='upper_left',
                   help='Where the inset sits in the main axes.')
    p.add_argument('--no_corr_text', action='store_true', default=False,
                   help='Do not write the correlations on the figure (still printed).')
    p.add_argument('--exclude_random', action='store_true', default=False,
                   help='Leave the random-subset runs out of the plot and correlations.')
    p.add_argument('--full_metrics', default=None,
                   help='Override path to the full-data metrics.json. May contain '
                        '{scratch}, {dataset}, {architecture}, {seed}. Default: '
                        '$SCRATCH/data_selection_swap/{dataset}/{architecture}/'
                        'checkpoints/metrics.json, then the seed_{seed} child.')
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
    attach_accuracy_gap(rows, args.scratch, template=args.full_metrics)
    used = sorted({row['full_metrics_path'] for row in rows if row.get('full_metrics_path')})
    for path in used:
        logger.info(f"Full-data accuracy from {path}")
    results = plot_all(rows, out_dir, logger, width=args.width,
                       formats=tuple(args.formats), title=not args.no_title,
                       xscale=args.xscale, zoom_frac=args.zoom_frac,
                       zoom_box=args.zoom_box, inset_loc=args.inset_loc,
                       corr_text=not args.no_corr_text,
                       include_random=not args.exclude_random)
    if not results:
        logger.error("No figure drawn: no row had both a G value and an accuracy. "
                     "Runs made with --selection_only have no accuracy to pair.")
        return
    logger.info(f"Wrote {len(results)} figure(s) -> {out_dir}")


if __name__ == '__main__':
    main()