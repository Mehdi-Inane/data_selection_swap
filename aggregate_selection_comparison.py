#!/usr/bin/env python3
"""
aggregate_selection_comparison.py

Summarises KL-Faithful vs. every baseline in the selection suite on the two
quantities of interest, mean ± std over seeds, per (fraction, method):

  * final-epoch test accuracy                         (<method>_<budget>_metrics.json)
  * selection time, three ways                        (<method>_<budget>_timing.json)
      selection  : wall-clock of the method's own work (extra training +
                   scoring + solver) — what the method adds on top of the
                   shared full-data trajectory
      reference  : time of the first `reference_epochs_used` epochs of the
                   shared trajectory (train_full_data.py's train_time.json),
                   i.e. the part of it the method actually needed
      total      : selection + reference — the end-to-end cost of producing
                   the subset from scratch

Usage:
    python aggregate_selection_comparison.py --dataset cifar100
"""

import argparse
import csv
import json
import os

import numpy as np

N_TRAIN = {'cifar100': 45000, 'imagenet': 1281167 - int(0.1 * 1281167)}


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def run_dir(scratch, method, dataset, budget, seed):
    if method == 'kl_faithful':
        return os.path.join(scratch, 'gradmatch_swap', dataset, str(budget), f'seed_{seed}')
    return os.path.join(scratch, 'data_selection_baselines', method, dataset,
                        str(budget), f'seed_{seed}')


def discover_methods(scratch, dataset):
    root = os.path.join(scratch, 'data_selection_baselines')
    found = sorted(m for m in os.listdir(root)
                   if os.path.isdir(os.path.join(root, m, dataset))) if os.path.isdir(root) else []
    return ['kl_faithful'] + found


def mean_std(values):
    return (float(np.mean(values)), float(np.std(values))) if values else (float('nan'),) * 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cifar100', choices=list(N_TRAIN))
    p.add_argument('--fractions', default=[0.1, 0.2, 0.4, 0.5], type=float, nargs='+')
    p.add_argument('--seeds', default=[42, 43, 44, 45, 46], type=int, nargs='+')
    p.add_argument('--methods', default=None, nargs='+',
                   help='Default: kl_faithful + every method found on disk.')
    p.add_argument('--scratch', default=os.environ.get('SCRATCH', '/home/mila/a/ahmedm/scratch'))
    p.add_argument('--reference_timing', default=None,
                   help='train_time.json of the shared trajectory. Default: '
                        '$SCRATCH/data_selection_swap/<dataset>/checkpoints/train_time.json')
    args = p.parse_args()

    ref_path = args.reference_timing or os.path.join(
        args.scratch, 'data_selection_swap', args.dataset, 'checkpoints', 'train_time.json')
    ref = load_json(ref_path)
    if ref is None:
        print(f"[WARNING] No reference timing at {ref_path}; 'reference' cost will be NaN.")
    epoch_seconds = ref['epoch_seconds'] if ref else None

    methods = args.methods or discover_methods(args.scratch, args.dataset)
    rows = []
    for fraction in args.fractions:
        budget = int(fraction * N_TRAIN[args.dataset])
        for method in methods:
            acc, sel, refc, tot = [], [], [], []
            for seed in args.seeds:
                d = run_dir(args.scratch, method, args.dataset, budget, seed)
                metrics = load_json(os.path.join(d, f'{method}_{budget}_metrics.json'))
                timing  = load_json(os.path.join(d, f'{method}_{budget}_timing.json'))
                if metrics:
                    acc.append(float(metrics[str(max(int(k) for k in metrics))]))
                if timing:
                    s = timing['selection_seconds']
                    e = timing.get('reference_epochs_used', 0)
                    r = sum(epoch_seconds[:e]) if epoch_seconds else float('nan')
                    sel.append(s); refc.append(r); tot.append(s + r)
            if not acc and not sel:
                continue
            rows.append(dict(fraction=fraction, method=method, n_seeds=len(acc),
                             acc=mean_std(acc), selection_s=mean_std(sel),
                             reference_s=mean_std(refc), total_s=mean_std(tot)))

    header = (f"{'frac':<6}{'method':<34}{'seeds':<7}{'test acc %':<16}"
              f"{'selection (s)':<20}{'reference (s)':<16}{'total (s)':<16}")
    print("\n" + header + "\n" + "-" * len(header))
    last = None
    for r in rows:
        if last is not None and r['fraction'] != last:
            print("-" * len(header))
        last = r['fraction']
        print(f"{r['fraction']:<6}{r['method']:<34}{r['n_seeds']:<7}"
              f"{r['acc'][0]:6.2f} ± {r['acc'][1]:<6.2f}"
              f"{r['selection_s'][0]:9.1f} ± {r['selection_s'][1]:<8.1f}"
              f"{r['reference_s'][0]:<16.1f}{r['total_s'][0]:<16.1f}")

    out_csv = os.path.join(args.scratch, 'data_selection_baselines',
                           f'comparison_{args.dataset}.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['fraction', 'method', 'n_seeds', 'acc_mean', 'acc_std',
                    'selection_s_mean', 'selection_s_std', 'reference_s', 'total_s_mean'])
        for r in rows:
            w.writerow([r['fraction'], r['method'], r['n_seeds'], *r['acc'],
                        *r['selection_s'], r['reference_s'][0], r['total_s'][0]])
    print(f"\nSaved → {out_csv}")


if __name__ == "__main__":
    main()
