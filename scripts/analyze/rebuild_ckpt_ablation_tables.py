#!/usr/bin/env python3
"""Rebuild the checkpoint-count ablation tables from per-run metrics.json.

Three independent sources are cross-checked before any cell is printed:

  A. RAW      - every kl_faithful_full_*_K{k}_cfg*_metrics.json is read here
                and the final-epoch accuracy taken (same rule as
                score_subsets.final_accuracy: max epoch key).
  B. AGG      - the saved retrain_accuracy_vs_K.csv aggregate.
  C. D_EXACT  - the d_exact_<ds>_<arch>_f<f>.csv rows for the same runs
                (acc column joined in by score_subsets), per-seed comparison.

A vs B must match to rounding (0.005) in mean, std and n; A vs C is compared
seed by seed at 1e-6. Missing seeds are rendered with a dagger and their
count, never silently averaged. Std is the sample standard deviation
(ddof=1), matching analysis.comparison.mean_std.

Writes tables/ablation_resnet18_cifar10.tex (10-40%), and
tables/ablation_ckpt_{cifar10,cifar100}.tex (10-50%, all architectures).
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

SCRATCH = os.environ.get('SCRATCH', '/home/mila/a/ahmedm/scratch')
BASE = os.path.join(SCRATCH, 'data_selection_baselines')
DATASETS = ['cifar10', 'cifar100']
ARCHS = ['resnet18', 'vgg16', 'inception']
ARCH_NAMES = {'resnet18': 'ResNet-18', 'vgg16': 'VGG-16', 'inception': 'Inception'}
KS = [1, 4, 10, 15, 20]
FRACTIONS = [0.1, 0.2, 0.4, 0.5]
SEEDS = [42, 43, 44, 45, 46]
N_TRAIN = 50000
METRICS_RE = re.compile(
    r'^kl_faithful_full_(?:resnet18|vgg16|inception)_(?:\w+?)_(\d+)_K(\d+)'
    r'_cfg[0-9a-f]+_metrics\.json$')


def source_raw():
    """(dataset, arch, fraction, K, seed) -> final-epoch accuracy."""
    out = {}
    for ds in DATASETS:
        for arch in ARCHS:
            for d in glob.glob(f'{SCRATCH}/gradmatch_swap/{ds}/{arch}/*/seed_*'):
                seed, budget = int(d.split('seed_')[1]), int(d.split('/')[-2])
                for fname in os.listdir(d):
                    m = METRICS_RE.match(fname)
                    if not m or '_from_' in fname:
                        continue
                    metrics = json.load(open(os.path.join(d, fname)))
                    last = max(metrics, key=lambda e: int(e))
                    out[(ds, arch, round(budget / N_TRAIN, 3),
                         int(m.group(2)), seed)] = float(metrics[last])
    return out


def source_agg():
    df = pd.read_csv(os.path.join(BASE, 'retrain_accuracy_vs_K.csv'))
    df['fraction'] = (df['budget'] / N_TRAIN).round(3)
    return {(r.dataset, r.architecture, r.fraction, int(r.K)): (r.mean, r.std, int(r.n))
            for r in df.itertuples()}


def source_d_exact():
    """(dataset, arch, fraction, K, seed) -> acc from the d_exact tables."""
    out = {}
    for ds in DATASETS:
        for arch in ARCHS:
            for f in FRACTIONS:
                path = os.path.join(BASE, f'd_exact_{ds}_{arch}_f{f}.csv')
                if not os.path.exists(path):
                    continue
                df = pd.read_csv(path)
                kl = df[df['method'].str.startswith('kl_faithful')].dropna(subset=['acc'])
                for r in kl.itertuples():
                    if int(r.num_checkpoints_tag) in KS:
                        out[(ds, arch, round(r.fraction, 3),
                             int(r.num_checkpoints_tag), int(r.seed))] = float(r.acc)
    return out


def cross_check(raw, agg):
    problems = []
    for ds in DATASETS:
        for arch in ARCHS:
            for f in FRACTIONS:
                for K in KS:
                    key = (ds, arch, f, K)
                    vals = [raw[k] for k in raw if k[:4] == key]
                    if key not in agg:
                        problems.append(f'B missing aggregate for {key}')
                        continue
                    b_mean, b_std, b_n = agg[key]
                    mean = round(float(np.mean(vals)), 2)
                    std = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 2)
                    if len(vals) != b_n or abs(mean - round(b_mean, 2)) > 0.005 \
                            or abs(std - round(b_std, 2)) > 0.005:
                        problems.append(f'A/B mismatch {key}: raw {mean}±{std} (n={len(vals)}) '
                                        f'vs agg {b_mean}±{b_std} (n={b_n})')
    return problems


def cross_check_d_exact(raw, d_exact):
    mismatches = []
    compared = 0
    for key, acc_c in d_exact.items():
        if key in raw:
            compared += 1
            if abs(raw[key] - acc_c) > 1e-6:
                mismatches.append(f'A/C mismatch {key}: raw {raw[key]} vs d_exact {acc_c}')
    return compared, mismatches


def cell(raw, ds, arch, f, K):
    vals = [raw[k] for k in raw if k[:4] == (ds, arch, f, K)]
    if not vals:
        return '—'
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    txt = f'{mean:.2f} $\\pm$ {std:.2f}'
    if len(vals) < len(SEEDS):
        txt += f'^{{\\dagger({len(vals)})}}'
    return txt


def full_table_tex(ds, fractions):
    cols = ' & '.join(f'{int(f * 100)}\\%' for f in fractions)
    lines = [
        '\\begin{table}', '\\centering',
        f'\\caption{{Test accuracy (\\%) of full-network Gradient Swap on {ds.upper()}. '
        'Values are mean $\\pm$ standard deviation over seeds.}}',
        f'\\label{{tab:ablation_ckpt_{ds}}}',
        f'\\begin{{tabular}}{{ll{"cc" * len(fractions)}}}',
        '\\toprule',
        f'Architecture & $\\tau$ & \\multicolumn{{{len(fractions)}}}{{c}}{{Test accuracy}} \\\\',
        f'\\cmidrule(lr){{3-{2 + len(fractions)}}}',
        f' & & {cols} \\\\',
        '\\midrule',
    ]
    for arch in ARCHS:
        for i, K in enumerate(KS):
            name = ARCH_NAMES[arch] if i == 0 else ''
            cells = ' & '.join(cell(raw, ds, arch, f, K) for f in fractions)
            lines.append(f'{name} & {K} & {cells} \\\\')
        lines.append('\\midrule' if arch != ARCHS[-1] else '\\bottomrule')
    lines += ['\\end{tabular}', '\\end{table}']
    return '\n'.join(lines)


if __name__ == '__main__':
    raw, agg, d_exact = source_raw(), source_agg(), source_d_exact()
    print(f'source A (raw metrics.json cells): {len(raw)}')
    print(f'source B (aggregate csv cells)   : {len(agg)}')
    print(f'source C (d_exact seed rows)     : {len(d_exact)}')

    problems = cross_check(raw, agg)
    compared, mismatches = cross_check_d_exact(raw, d_exact)
    print(f'A vs B: {len(agg)} cells checked, {len(problems)} problems')
    print(f'A vs C: {compared} seed-level accs compared, {len(mismatches)} mismatches')
    for p in problems + mismatches:
        print('  !!', p)
    if problems or mismatches:
        raise SystemExit('cross-check FAILED - not writing tables')

    out_dir = os.path.join(BASE, 'tables')
    os.makedirs(out_dir, exist_ok=True)

    rn_small = (
        '% Rebuilt from *_metrics.json; see rebuild_ckpt_ablation_tables.py.\n'
        '\\begin{tabular}{c ccc}\n\\toprule\n$\\tau$ & 10\\% & 20\\% & 40\\% \\\\\n\\midrule\n' +
        '\n'.join(f'{K:<2d} & ' + ' & '.join(cell(raw, 'cifar10', 'resnet18', f, K)
                                             for f in (0.1, 0.2, 0.4)) + ' \\\\'
                  for K in KS) +
        '\n\\bottomrule\n\\end{tabular}')
    path = os.path.join(out_dir, 'ablation_resnet18_cifar10.tex')
    with open(path, 'w') as fh:
        fh.write(rn_small + '\n')
    print(f'Saved -> {path}')

    for ds in DATASETS:
        path = os.path.join(out_dir, f'ablation_ckpt_{ds}.tex')
        with open(path, 'w') as fh:
            fh.write(full_table_tex(ds, FRACTIONS) + '\n')
        print(f'Saved -> {path}')

    short = [(ds, arch, f, K)
             for ds in DATASETS for arch in ARCHS for f in FRACTIONS for K in KS
             if sum((ds, arch, f, K, s) in raw for s in SEEDS) < 5]
    print('cells with fewer than 5 seeds (daggered):', short if short else 'none')
