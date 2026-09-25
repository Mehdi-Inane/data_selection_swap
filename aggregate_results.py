#!/usr/bin/env python3
"""
aggregate_results.py

Reads per-seed metrics JSON files produced by all three training scripts and
prints a summary table of mean ± std test accuracy at the final epoch, for
every (method, fraction) combination.  Also writes JSON and CSV summaries.

Expected directory layout (produced by the modified training scripts):
    <BASE_DIR>/<budget>/seed_<seed>/<method>_<budget>_metrics.json

Usage:
    python aggregate_results.py
"""

import os
import json
import csv
import numpy as np

# ── Configuration — edit to match your runs ───────────────────────────────────
BASE_DIR   = "/home/mila/a/ahmedm/scratch/gradmatch_swap/cifar10"
FRACTIONS  = [0.1, 0.2, 0.4, 0.5]
N_TRAIN    = 45000          # 90 % of CIFAR-10 training set (50 000 × 0.9)
SEEDS      = [42, 43, 44, 45, 46]
METHODS    = ["random", "gradmatch", "kl_faithful"]
# ─────────────────────────────────────────────────────────────────────────────


def load_final_accuracy(metrics_path: str) -> float:
    """Return the test accuracy recorded at the highest epoch in a metrics file."""
    with open(metrics_path) as fh:
        metrics = json.load(fh)
    # JSON stores integer keys as strings; pick the highest epoch
    final_epoch = str(max(int(k) for k in metrics.keys()))
    return float(metrics[final_epoch])


def collect_results() -> dict:
    """
    Walk every (fraction, method, seed) combination and collect accuracies.

    Returns
    -------
    dict keyed as results[fraction][method] = {
        "mean"     : float,
        "std"      : float,
        "per_seed" : {seed: accuracy, ...},
    }
    """
    results = {}

    for fraction in FRACTIONS:
        budget = int(fraction * N_TRAIN)
        results[fraction] = {}

        for method in METHODS:
            per_seed = {}

            for seed in SEEDS:
                metrics_path = os.path.join(
                    BASE_DIR,
                    str(budget),
                    f"seed_{seed}",
                    f"{method}_{budget}_metrics.json",
                )
                if not os.path.exists(metrics_path):
                    print(f"[WARNING] Missing file: {metrics_path}")
                    continue

                per_seed[seed] = load_final_accuracy(metrics_path)

            if not per_seed:
                print(f"[WARNING] No data for fraction={fraction}, method={method}")
                continue

            acc_values = list(per_seed.values())
            results[fraction][method] = {
                "mean"     : float(np.mean(acc_values)),
                "std"      : float(np.std(acc_values)),
                "per_seed" : per_seed,
            }

    return results


def print_table(results: dict) -> None:
    """Pretty-print the results as a console table."""
    C = [10, 14, 10, 10, 45]          # column widths
    total_w = sum(C)
    sep = "-" * total_w

    header = (
        f"{'Fraction':<{C[0]}}"
        f"{'Method':<{C[1]}}"
        f"{'Mean %':<{C[2]}}"
        f"{'Std %':<{C[3]}}"
        f"{'Per-seed accuracies'}"
    )

    print()
    print("=" * total_w)
    print("RESULTS SUMMARY — Final-epoch test accuracy (mean ± std across 5 seeds)")
    print("=" * total_w)
    print(header)

    for fraction in FRACTIONS:
        print(sep)
        for method in METHODS:
            if method not in results.get(fraction, {}):
                print(f"{fraction:<{C[0]}}{method:<{C[1]}}{'N/A'}")
                continue

            r = results[fraction][method]
            per_seed_str = "  ".join(
                f"s{s}={r['per_seed'][s]:.2f}" for s in SEEDS if s in r["per_seed"]
            )
            print(
                f"{fraction:<{C[0]}}"
                f"{method:<{C[1]}}"
                f"{r['mean']:<{C[2]}.2f}"
                f"{r['std']:<{C[3]}.2f}"
                f"{per_seed_str}"
            )

    print("=" * total_w)


def save_json(results: dict, out_path: str) -> None:
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=4)
    print(f"\nSaved JSON summary  →  {out_path}")


def save_csv(results: dict, out_path: str) -> None:
    seed_cols = [f"seed_{s}" for s in SEEDS]
    fieldnames = ["fraction", "method", "mean_acc", "std_acc"] + seed_cols

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for fraction in FRACTIONS:
            for method in METHODS:
                if method not in results.get(fraction, {}):
                    continue
                r = results[fraction][method]
                row = {
                    "fraction" : fraction,
                    "method"   : method,
                    "mean_acc" : f"{r['mean']:.4f}",
                    "std_acc"  : f"{r['std']:.4f}",
                }
                for s in SEEDS:
                    col = f"seed_{s}"
                    row[col] = (
                        f"{r['per_seed'][s]:.4f}" if s in r["per_seed"] else "N/A"
                    )
                writer.writerow(row)

    print(f"Saved CSV summary   →  {out_path}")


def main() -> None:
    results = collect_results()
    print_table(results)
    save_json(results, os.path.join(BASE_DIR, "aggregated_results_cifar10.json"))
    save_csv(results,  os.path.join(BASE_DIR, "aggregated_results_cifar10.csv"))


if __name__ == "__main__":
    main()