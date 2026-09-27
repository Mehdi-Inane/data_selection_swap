#!/usr/bin/env python3
"""
plot_checkpoint_ablation.py

Checkpoint-count ablation for KL-Faithful selection: how much of the
training trajectory does the algorithm actually need to select a good
subset? Reads the results produced by checkpoint_ablation.sh (which runs
train_kl_selection.py with --num_checkpoints in {1, 3, 5, 10, 20}, for
every fraction and seed) and reports test accuracy — and selection
wall-clock — as a function of K, one curve per fraction, averaged over
seeds.

This deliberately reuses the file-discovery logic in
aggregate_selection_comparison.py (same run_dir convention, same
'_K<n>' filename tag introduced for the checkpoint-count experiments)
rather than re-implementing it, restricted here to method='kl_faithful'
and grouped explicitly by K instead of just reported alongside it.

Usage:
    python plot_checkpoint_ablation.py --dataset cifar100
    python plot_checkpoint_ablation.py --dataset cifar100 \
        --checkpoints 1 3 5 10 20 --fractions 0.1 0.2 0.4 0.5
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from aggregate_selection_comparison import (N_TRAIN, discover_runs, load_json,
                                             mean_std, run_dir)

METHOD = 'kl_faithful'


# ─────────────────────────────────────────────────────────────────────────────
# Collect results: fraction -> K -> {acc, selection_s, reference_s, total_s}
# ─────────────────────────────────────────────────────────────────────────────
def collect(scratch, dataset, fractions, seeds, checkpoints, ref_path):
    ref = load_json(ref_path)
    epoch_seconds = ref['epoch_seconds'] if ref else None
    if ref is None:
        print(f"[WARNING] No reference timing at {ref_path}; "
              f"'reference'/'total' cost will be NaN.")

    results = {}
    for fraction in fractions:
        budget = int(fraction * N_TRAIN[dataset])
        by_k = {}
        for seed in seeds:
            d = run_dir(scratch, METHOD, dataset, budget, seed)
            for k, metrics_path, timing_path in discover_runs(d, METHOD, budget):
                if k is None or k not in checkpoints:
                    continue  # skip untagged/legacy runs and any K outside the sweep
                bucket = by_k.setdefault(k, dict(acc=[], sel=[], refc=[], tot=[]))

                metrics = load_json(metrics_path)
                if metrics:
                    last_epoch = str(max(int(ep) for ep in metrics))
                    bucket['acc'].append(float(metrics[last_epoch]))

                timing = load_json(timing_path) if timing_path else None
                if timing:
                    s = timing['selection_seconds']
                    e = timing.get('reference_epochs_used', k)
                    r = sum(epoch_seconds[:e]) if epoch_seconds else float('nan')
                    bucket['sel'].append(s)
                    bucket['refc'].append(r)
                    bucket['tot'].append(s + r)
        results[fraction] = by_k
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
def print_table(results, checkpoints):
    header = (f"{'frac':<6}{'K':<6}{'seeds':<7}{'test acc %':<16}"
              f"{'selection (s)':<18}{'total (s)':<14}")
    print("\n" + header + "\n" + "-" * len(header))
    for fraction, by_k in results.items():
        for k in checkpoints:
            b = by_k.get(k)
            if not b or not b['acc']:
                continue
            acc_m, acc_s = mean_std(b['acc'])
            sel_m, sel_s = mean_std(b['sel'])
            tot_m, _     = mean_std(b['tot'])
            print(f"{fraction:<6}{k:<6}{len(b['acc']):<7}{acc_m:6.2f} ± {acc_s:<6.2f}"
                  f"{sel_m:9.1f} ± {sel_s:<6.1f}{tot_m:<14.1f}")
        print("-" * len(header))


def save_csv(results, checkpoints, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['fraction', 'num_checkpoints', 'n_seeds', 'acc_mean', 'acc_std',
                    'selection_s_mean', 'selection_s_std', 'total_s_mean'])
        for fraction, by_k in results.items():
            for k in checkpoints:
                b = by_k.get(k)
                if not b or not b['acc']:
                    continue
                acc_m, acc_s = mean_std(b['acc'])
                sel_m, sel_s = mean_std(b['sel'])
                tot_m, _     = mean_std(b['tot'])
                w.writerow([fraction, k, len(b['acc']), acc_m, acc_s, sel_m, sel_s, tot_m])
    print(f"Saved → {out_csv}")


def plot(results, checkpoints, fractions, out_png):
    fig, (ax_acc, ax_time) = plt.subplots(1, 2, figsize=(11, 4.5))

    for fraction in fractions:
        by_k = results.get(fraction, {})
        ks, means, stds = [], [], []
        for k in checkpoints:
            b = by_k.get(k)
            if not b or not b['acc']:
                continue
            m, s = mean_std(b['acc'])
            ks.append(k); means.append(m); stds.append(s)
        if ks:
            ax_acc.errorbar(ks, means, yerr=stds, marker='o', capsize=3,
                            label=f'fraction={fraction}')

    ax_acc.set_xlabel('Number of checkpoints used (K)')
    ax_acc.set_ylabel('Test accuracy (%)')
    ax_acc.set_title('Selection quality vs. trajectory length')
    ax_acc.set_xticks(checkpoints)
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    for fraction in fractions:
        by_k = results.get(fraction, {})
        ks, means = [], []
        for k in checkpoints:
            b = by_k.get(k)
            if not b or not b['sel']:
                continue
            m, _ = mean_std(b['sel'])
            ks.append(k); means.append(m)
        if ks:
            ax_time.plot(ks, means, marker='o', label=f'fraction={fraction}')

    ax_time.set_xlabel('Number of checkpoints used (K)')
    ax_time.set_ylabel('Selection wall-clock (s)')
    ax_time.set_title('Selection cost vs. trajectory length')
    ax_time.set_xticks(checkpoints)
    ax_time.legend()
    ax_time.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"Saved → {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='cifar100', choices=list(N_TRAIN))
    p.add_argument('--fractions', default=[0.1, 0.2, 0.4, 0.5], type=float, nargs='+')
    p.add_argument('--seeds', default=[42, 43, 44, 45, 46], type=int, nargs='+')
    p.add_argument('--checkpoints', default=[1, 3, 5, 10, 20], type=int, nargs='+',
                   help='--num_checkpoints values swept by checkpoint_ablation.sh')
    p.add_argument('--scratch', default=os.environ.get('SCRATCH', '/home/mila/a/ahmedm/scratch'))
    p.add_argument('--reference_timing', default=None,
                   help='train_time.json of the shared trajectory. Default: '
                        '$SCRATCH/data_selection_swap/<dataset>/checkpoints/train_time.json')
    args = p.parse_args()

    ref_path = args.reference_timing or os.path.join(
        args.scratch, 'data_selection_swap', args.dataset, 'checkpoints', 'train_time.json')

    results = collect(args.scratch, args.dataset, args.fractions, args.seeds,
                      args.checkpoints, ref_path)
    print_table(results, args.checkpoints)

    out_dir = os.path.join(args.scratch, 'data_selection_baselines', 'checkpoint_ablation')
    save_csv(results, args.checkpoints,
            os.path.join(out_dir, f'checkpoint_ablation_{args.dataset}.csv'))
    plot(results, args.checkpoints, args.fractions,
        os.path.join(out_dir, f'checkpoint_ablation_{args.dataset}.pdf'))


if __name__ == "__main__":
    main()