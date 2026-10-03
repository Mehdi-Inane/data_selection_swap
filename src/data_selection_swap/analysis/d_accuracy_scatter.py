#!/usr/bin/env python3
"""Scatter sqrt(D_alpha / 2) against the full-data accuracy gap.

Input is the per-subset table written by ``analysis.score_subsets``: one row
per ``(dataset, architecture, fraction, method, K, seed)``. The figure checks
the guarantee

    | E_{theta~A_n(S)}[q(theta)] - E_{theta~A_m(S_I)}[q(theta)] |
        <= (b - a) * sqrt(D_alpha / 2)

with q = test accuracy expressed as a fraction in [0, 1], so b - a = 1 and the
bound reads  y <= x  for  x = sqrt(D_alpha / 2). The y coordinate is

    y = | acc_full - acc_subset |      (a fraction, not percent).

``acc_full`` is read from the full-data trajectory's ``metrics.json``
(``final_test_acc``, stored as a fraction). Subset accuracies in the scored
table are in percent and are converted. Each run contributes a single draw of
theta, so y is a one-sample estimate of the left-hand side.

One figure per ``(dataset, architecture, fraction)``. Fixing the fraction is
what makes the x axis meaningful: D falls roughly like 1/m, so subsets of
different budgets do not share a scale and must not be pooled into one panel.

Figure
──────
Two panels side by side, each a plain scatter with one marker per (selection
method, K), averaged over seeds (--per_seed plots every seed instead):
  (a) every method, including the far-away ones (e.g. el2n);
  (b) a zoom on the dense region, outlined by a dotted box in (a).
The grey band/dashed line mark the range/mean of the random draws (the noise
floor of D).

Seed averaging
──────────────
The bound holds for each seed (each has its own split S, subset S_I and D), so
averaging the per-seed inequalities gives

    mean_s y_s  <=  mean_s sqrt(D_s / 2).

The default --x_average mean_sqrt therefore plots exactly the two sides of that
averaged bound. --x_average sqrt_mean uses sqrt(mean_s D_s / 2) instead, which
is never smaller (Jensen: sqrt is concave), i.e. a looser but still valid
bound. y is the mean of the per-seed absolute gaps, not |mean of the gaps|.

Spearman is rank-based, so the sqrt(./2) transform leaves every value
identical to the one computed on D itself. A *positive* Spearman on the gap
supports the objective: lower D goes with a smaller accuracy drop.

Nothing statistical is drawn on the figure. The Spearman values and the
colour/marker of every series are printed to the terminal. The legend is drawn
below the figure (--no_legend removes it); method names come from METHOD_NAMES.

Usage:
    python scripts/plot/d_accuracy_scatter.py \\
        --input $SCRATCH/data_selection_baselines/d_exact_*_f0.1.csv
    python scripts/plot/d_accuracy_scatter.py --input DIR --width double \\
        --formats pdf png --no_title --zoom_xmax 1.0
"""

import argparse
import csv
import glob
import json
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


def plotted_y(row):
    """Y coordinate: accuracy gap when attached, otherwise subset accuracy (%)."""
    return row.get('y', row.get('acc'))


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
                    x=float(np.sqrt(max(d, 0.0) / 2.0)),   # plotted: sqrt(D_alpha / 2)
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
    """Within-seed Spearman of D against the plotted y (gap, or accuracy)."""
    within, pooled_d, pooled_y = [], [], []
    for seed in sorted({row['seed'] for row in points}):
        group = [row for row in points if row['seed'] == seed]
        if len(group) < min_group:
            continue
        d = np.array([row['d'] for row in group], dtype=np.float64)
        y = np.array([plotted_y(row) for row in group], dtype=np.float64)
        if not np.all(np.isfinite(y)):
            continue
        rho = _spearman(d, y)
        if not np.isfinite(rho):
            continue
        within.append(rho)
        pooled_d.extend(d - d.mean())
        pooled_y.extend(y - y.mean())
    return dict(
        within_mean=float(np.mean(within)) if within else float('nan'),
        within_std=(float(np.std(within, ddof=1 if len(within) > 1 else 0))
                    if within else float('nan')),
        n_seeds=len(within),
        pooled_centred=_spearman(pooled_d, pooled_y),
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

# Legend names. Keys are the ``method`` strings in the scored tables; values are
# the names used in the paper. Methods not listed fall back to the raw name with
# underscores replaced by spaces.
METHOD_NAMES = {
    'el2n': 'EL2N',
    'facility_location': 'Facility Location',
    'graphcut': 'GraphCut',
    'herding': 'Herding',
    'moderate': 'Moderate',
    'moso': 'MoSo',
    'random': 'Random',
}
# Our method: every ``kl_*`` variant (the table name embeds the architecture) is
# shown as "Gradient Swap, K epochs", with K from the table's K column.
GRADIENT_SWAP_NAME = 'Gradient Swap'

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

# Axis labels mirror the bound  |E_A_n(S)[q] - E_A_m(S_I)[q]| <= (b-a) sqrt(D_alpha/2).
X_LABEL = r'$\sqrt{D_\alpha/2}$'
Y_LABEL_GAP = (r'$\left|\,\mathbb{E}_{\theta\sim\mathcal{A}_n(S)}[q(\theta)]'
               r'-\mathbb{E}_{\theta\sim\mathcal{A}_m(S_I)}[q(\theta)]\,\right|$')
Y_LABEL_ACC = 'Test accuracy (%)'


def series_legend_label(method, k, kl_label=GRADIENT_SWAP_NAME):
    if is_kl_method(method):
        if k in (None, ''):
            return kl_label
        unit = 'epoch' if str(k) == '1' else 'epochs'
        return f"{kl_label}, {k} {unit}"
    return METHOD_NAMES.get(method, method.replace('_', ' '))


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
def figure_stem(dataset, architecture, fraction, y_name='accuracy_gap'):
    return f"{dataset}_{architecture}_f{fraction:g}_sqrtd_vs_{y_name}"


def uses_gap(rows):
    return any(row.get('y') is not None for row in rows)


def _average_x(xs, mode):
    """Average of x = sqrt(D/2) values: mean of the roots, or root of the mean D."""
    xs = np.asarray(xs, dtype=np.float64)
    if mode == 'sqrt_mean':
        return float(np.sqrt(np.mean(xs ** 2)))   # = sqrt(mean(D) / 2)
    return float(np.mean(xs))                      # = mean(sqrt(D / 2))


def aggregate_series(series, x_average='mean_sqrt'):
    """``(method, K)`` -> seed-averaged x, y, their across-seed std, and seed count."""
    out = {}
    for key, rows in series.items():
        xs = np.array([r['x'] for r in rows], dtype=np.float64)
        ys = np.array([plotted_y(r) for r in rows], dtype=np.float64)
        ddof = 1 if len(rows) > 1 else 0
        out[key] = dict(x=_average_x(xs, x_average), y=float(ys.mean()),
                        x_std=float(xs.std(ddof=ddof)), y_std=float(ys.std(ddof=ddof)),
                        n=len(rows))
    return out


def draw_series(ax, series, styles, size=22):
    """One plain marker per row of each series."""
    for key, rows in series.items():
        st = styles[key]
        ax.scatter([r['x'] for r in rows], [plotted_y(r) for r in rows],
                   s=size, color=st['color'], marker=st['marker'],
                   edgecolor='black', linewidth=0.5, alpha=0.95, zorder=3)


def draw_reference(ax, random_x, mean_x=None):
    if not random_x:
        return
    ax.axvspan(min(random_x), max(random_x), color='0.5', alpha=0.15, lw=0, zorder=0)
    ax.axvline(float(np.mean(random_x)) if mean_x is None else mean_x,
               color='0.4', ls='--', lw=0.8, zorder=1)


def zoom_region(points, random_x, factor, xmax=None):
    """Bounds ``(x0, x1, y0, y1)`` of the dense region, or None if zooming is pointless.

    The region holds every point with x <= ``xmax`` (default: ``factor`` times
    the largest x of the random draws, or the 75th percentile without them).
    """
    xs = np.array([r['x'] for r in points])
    if xmax is not None:
        limit = xmax
    elif random_x:
        limit = factor * max(random_x)
    else:
        limit = float(np.percentile(xs, 75))
    inside = [r for r in points if r['x'] <= limit]
    if len(inside) < 3 or len(inside) == len(points):
        return None
    x = np.array([r['x'] for r in inside] + list(random_x))
    y = np.array([plotted_y(r) for r in inside])
    pad_x = 0.06 * (x.max() - x.min() or 1.0)
    pad_y = 0.12 * (y.max() - y.min() or 1.0)
    return (max(x.min() - pad_x, 0.0), x.max() + pad_x,
            max(y.min() - pad_y, 0.0), y.max() + pad_y)


def plot_panel(panel_rows, out_stem, width='double', formats=('pdf',), title=True,
               kl_label=GRADIENT_SWAP_NAME, xscale='log', legend=True, zoom_factor=2.0,
               zoom_xmax=None, average=True, x_average='mean_sqrt'):
    """Two side-by-side scatters: all methods, and a zoom on the dense region.

    With ``average`` each (method, K) is one seed-averaged marker; otherwise
    every seed is plotted.

    Returns ``(stats, entries)``: the correlation stats and, per series,
    ``(label, hex colour, marker)`` for the caption. ``(None, None)`` if nothing
    could be drawn.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

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
    gap = uses_gap(points)
    if average:
        agg = aggregate_series(series, x_average)
        y_key = 'y' if gap else 'acc'
        plot_series = {key: [{'x': a['x'], y_key: a['y']}] for key, a in agg.items()}
        marker_size = 46
        random_mean = _average_x(random_x, x_average) if random_x else None
    else:
        plot_series, marker_size, random_mean = series, 22, None
    plot_points = [r for rows in plot_series.values() for r in rows]
    zoom = zoom_region(plot_points, random_x, zoom_factor, zoom_xmax)

    with plt.rc_context(RC):
        w = WIDTHS_IN[width]
        styles = assign_styles(series, kl_label, plt)
        fig, axes = plt.subplots(1, 2 if zoom else 1,
                                 figsize=(w, 0.5 * w if zoom else 0.62 * w),
                                 squeeze=False)
        axes = list(axes[0])
        ax = axes[0]

        for a in axes:
            draw_reference(a, random_x, random_mean)
            draw_series(a, plot_series, styles, marker_size)
            a.set_xlabel(X_LABEL, fontsize=10)
            a.grid(alpha=0.25, lw=0.4)
        ax.set_ylabel(Y_LABEL_GAP if gap else Y_LABEL_ACC, fontsize=7.5 if gap else 9)

        if xscale == 'log' and all(r['x'] > 0 for r in plot_points) \
                and (not random_x or min(random_x) > 0):
            ax.set_xscale('log')
            # Plain numeric ticks (0.5, 1, 2, ...) instead of a lone "10^0".
            from matplotlib.ticker import NullFormatter, ScalarFormatter
            lo, hi = ax.get_xlim()
            ticks = [t for t in (0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50)
                     if lo <= t <= hi]
            if ticks:
                ax.set_xticks(ticks)
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.xaxis.set_minor_formatter(NullFormatter())

        if zoom:
            x0, x1, y0, y1 = zoom
            zax = axes[1]
            zax.set_xlim(x0, x1)
            zax.set_ylim(y0, y1)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   ec='black', lw=0.7, ls=':', zorder=5))
            ax.set_title('(a) All methods', loc='left', fontsize=8.5)
            zax.set_title('(b) Zoom on dotted region', loc='left', fontsize=8.5)

        if title:
            budget = points[0]['budget']
            budget_text = f", budget {int(budget)}" if budget else ''
            fig.suptitle(f"{dataset} / {architecture}, fraction {fraction:g}"
                         f"{budget_text}", fontsize=9)

        fig.tight_layout()
        if legend:
            handles = [Line2D([], [], ls='', marker=styles[k]['marker'], ms=5,
                              mfc=styles[k]['color'], mec='black', mew=0.5,
                              label=styles[k]['label']) for k in series]
            if random_x:
                handles.append(Patch(facecolor='0.5', alpha=0.25,
                                     label=f"D of random subsets, range (n={len(random_x)})"))
                handles.append(Line2D([], [], color='0.4', ls='--', lw=0.8,
                                      label='D of random subsets, mean'))
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


def plot_all(rows, out_dir, logger=None, **kwargs):
    """One figure per (dataset, architecture, fraction). Returns their stats."""
    if logger is None:
        logger = logging.getLogger('d_accuracy_scatter')
    results = {}
    y_name = 'accuracy_gap' if uses_gap(rows) else 'accuracy'
    y_stat = '|full - subset| accuracy gap' if y_name == 'accuracy_gap' else 'accuracy'
    support = ('positive = lower D goes with a smaller accuracy drop'
               if y_name == 'accuracy_gap' else
               'negative = lower D goes with higher accuracy')
    for key, panel_rows in group_by_panel(rows).items():
        dataset, architecture, fraction = key
        averaged = kwargs.get('average', True)
        out_stem = os.path.join(out_dir, figure_stem(dataset, architecture, fraction, y_name)
                                + ('_seedavg' if averaged else ''))
        stats, entries = plot_panel(panel_rows, out_stem, **kwargs)
        if stats is None:
            continue
        results[key] = stats
        points, _ = split_points_and_reference(panel_rows)
        logger.info(
            f"\n=== {dataset} / {architecture}, fraction {fraction:g} "
            f"({len(points)} points) -> {out_stem}\n"
            f"Spearman(D, {y_stat})  [identical for sqrt(D/2); {support}]\n"
            f"  within seed : {stats['within_mean']:+.3f} ± {stats['within_std']:.3f} "
            f"(over {stats['n_seeds']} seeds)\n"
            f"  pooled, seed-centred : {stats['pooled_centred']:+.3f} "
            f"({stats['n_points']} points)")
        if y_name == 'accuracy_gap':
            violations = [r for r in points if r['y'] > r['x'] + 1e-12]
            logger.info(f"Bound y <= sqrt(D/2) (accuracy as a fraction, b-a=1): "
                        f"violated by {len(violations)} of {len(points)} points")
        if averaged:
            mode = kwargs.get('x_average', 'mean_sqrt')
            agg = aggregate_series(group_by_series(points), mode)
            kl_label = kwargs.get('kl_label', GRADIENT_SWAP_NAME)
            lines = [f"  {series_legend_label(m, k, kl_label).replace('$', ''):<22s} "
                     f"x={a['x']:.4f} (std {a['x_std']:.4f})  "
                     f"y={a['y']:.4f} (std {a['y_std']:.4f})  n={a['n']}"
                     for (m, k), a in agg.items()]
            rho = _spearman([a['x'] for a in agg.values()], [a['y'] for a in agg.values()])
            n_violated = sum(a['y'] > a['x'] + 1e-12 for a in agg.values()) \
                if y_name == 'accuracy_gap' else None
            logger.info(
                f"Seed-averaged points (x = {mode}: "
                f"{'mean_s sqrt(D_s/2)' if mode == 'mean_sqrt' else 'sqrt(mean_s D_s / 2)'})\n"
                + "\n".join(lines)
                + f"\nSpearman across the {len(agg)} method means: {rho:+.3f}"
                + (f"\nAveraged bound y <= x violated by {n_violated} of {len(agg)} "
                   f"methods" if n_violated is not None else ''))
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
    p.add_argument('--no_legend', action='store_true', default=False,
                   help='Omit the legend below the figure.')
    p.add_argument('--width', choices=list(WIDTHS_IN), default='double',
                   help='Figure width: single (3.4in), onehalf (5.2in), double (7.2in).')
    p.add_argument('--formats', nargs='+', default=['pdf'],
                   help='Output formats, e.g. pdf png. Default: pdf (vector).')
    p.add_argument('--no_title', action='store_true', default=False,
                   help='Omit the figure title (use the caption instead).')
    p.add_argument('--kl_label', default=GRADIENT_SWAP_NAME,
                   help='Legend name of our (KL-based) method.')
    p.add_argument('--xscale', choices=['log', 'linear'], default='log',
                   help='x axis scale of panel (a).')
    p.add_argument('--zoom_factor', type=float, default=2.0,
                   help='Panel (b) holds points with sqrt(D/2) <= factor * the largest '
                        'sqrt(D/2) of the random draws.')
    p.add_argument('--zoom_xmax', type=float, default=None,
                   help='Explicit right edge of the zoom, in sqrt(D/2) units '
                        '(overrides --zoom_factor).')
    p.add_argument('--per_seed', action='store_true', default=False,
                   help='Plot every seed as its own marker instead of averaging '
                        'over seeds for each (method, K).')
    p.add_argument('--x_average', choices=['mean_sqrt', 'sqrt_mean'], default='mean_sqrt',
                   help='How x is averaged over seeds: mean_s sqrt(D_s/2) (default; '
                        'matches the averaged bound) or sqrt(mean_s D_s / 2).')
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
    results = plot_all(rows, out_dir, logger, legend=not args.no_legend, width=args.width,
                       formats=tuple(args.formats), title=not args.no_title,
                       kl_label=args.kl_label, xscale=args.xscale,
                       zoom_factor=args.zoom_factor, zoom_xmax=args.zoom_xmax,
                       average=not args.per_seed, x_average=args.x_average)
    if not results:
        logger.error("No figure drawn: no row had both a D value and an accuracy. "
                     "Runs made with --selection_only have no accuracy to pair.")
        return
    logger.info(f"Wrote {len(results)} figure(s) -> {out_dir}")


if __name__ == '__main__':
    main()