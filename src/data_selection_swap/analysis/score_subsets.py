#!/usr/bin/env python3
"""Score every saved subset with the exact all-parameter objective
D(alpha), and join it against the test accuracy those same
runs already recorded. No training, no selection — it reads
`*_indices.pt` / `*_metrics.json` and answers one question:

    does the faithfulness objective rank subsets the way test accuracy does?

That is the question the whole framework rests on, and every ingredient for
it is already paid for: ~160 saved subsets (8 methods x 4 fractions x 5
seeds, plus kl_faithful's K variants) and their accuracies.

Differences from ``analysis.comparison``'s D_alpha column
───────────────────────────────────────────────────────────────────
* That column uses the last-layer factored kernel (linear.weight only);
  this uses the gradient of every trainable parameter, exactly.
* That column scores each K-tagged row on its *own* K-checkpoint
  trajectory. This scores everything on one common trajectory, because
  D values from different trajectories are not comparable to each other
  and the point here is a cross-method comparison.

Normalisation
─────────────
D shrinks with m mechanically — it is zero at I = [n] and falls like 1/m
for random subsets — so raw D cannot be compared across budgets. Fresh
random subsets are drawn at every (seed, budget) to estimate E_U[D], and
the reported D_rel = D / E_U[D] is "how much more faithful than random,
at this budget". D_rel is the column to compare across fractions; it also
gives the noise floor, since the spread of the random draws is the scale
below which differences between methods mean nothing.

Discovery caveat
────────────────
Runs are discovered through their `*_metrics.json`, the same way
``analysis.comparison`` does it, so a run made with
--selection_only (indices but no accuracy) is not picked up.

The objective being scored (--weighting, a.k.a. --optimizer) is the same
(w_k, M_k) pair the selection scripts optimise; ``trajectory`` documents
the choices. Scoring under adam needs a ``training.reference``
--optimizer adam trajectory, since vhat is not in a bare state_dict.

Usage:
    python scripts/analyze/score_subsets.py --dataset cifar100 --data_dir /path/to/cifar100_data
    python scripts/analyze/score_subsets.py --dataset cifar100 --data_dir ... \
        --num_checkpoints 10 --weighting sgd --n_random 5
    # pick the scoring checkpoints by hand:
    python scripts/analyze/score_subsets.py --dataset cifar100 --data_dir ... \
        --checkpoint_epochs 1 5 10 20 40 70 110 160 220 300
"""

import argparse
import csv
import logging
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

from .. import common
from .. import objective as d_alpha
from .. import trajectory as ref
from ..paths import tree
from .comparison import (N_TRAIN, RANDOM_METHOD, discover_methods,
                         discover_runs, indices_path_for, load_json, run_dir)
from .result_plots import locate_checkpoint_dir


# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────
def final_accuracy(metrics_path):
    metrics = load_json(metrics_path)
    if not metrics:
        return None
    last_epoch = str(max(int(ep) for ep in metrics))
    return float(metrics[last_epoch])


def collect_rows(scratch, dataset, architecture, fractions, seeds, methods, checkpoints,
                 variant=''):
    """One row per saved run: (fraction, method, K, seed, indices_path, acc)."""
    rows = []
    for fraction in fractions:
        budget = int(fraction * N_TRAIN[dataset])
        for method in methods:
            for seed in seeds:
                d = run_dir(
                    scratch, method, dataset, architecture, budget, seed,
                    variant=variant)
                for k, metrics_path, _ in discover_runs(d, method, budget):
                    if checkpoints is not None and k is not None and k not in checkpoints:
                        continue
                    indices_path = indices_path_for(metrics_path)
                    if indices_path is None:
                        continue
                    rows.append(dict(fraction=fraction, budget=budget, method=method,
                                     k=k, seed=seed, indices_path=indices_path,
                                     acc=final_accuracy(metrics_path)))
    return rows


def random_subsets(n_train, budget, seed, n_random):
    """Reproducible fresh random subsets, for the E_U[D] reference."""
    out = {}
    for r in range(n_random):
        g = torch.Generator().manual_seed(1000003 * seed + 31 * budget + r)
        out[r] = torch.randperm(n_train, generator=g)[:budget]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Correlation
#   D and accuracy both move strongly with the budget, and both carry a
#   per-seed offset (the seed fixes the split and the trajectory). Pooling
#   raw values across fractions or seeds would therefore report a
#   correlation that is mostly those two nuisance trends. So: correlate
#   within (fraction, seed) across methods, and for the pooled number,
#   centre both quantities within each (fraction, seed) group first.
# ─────────────────────────────────────────────────────────────────────────────
def _spearman(a, b):
    if len(a) < 3:
        return float('nan')
    rho = spearmanr(a, b).statistic
    return float(rho) if np.isfinite(rho) else float('nan')


def correlations(scored, fractions, seeds):
    per_fraction = {}
    for fraction in fractions:
        within, pooled_d, pooled_a = [], [], []
        for seed in seeds:
            group = [r for r in scored
                     if r['fraction'] == fraction and r['seed'] == seed
                     and r['method'] != RANDOM_METHOD and r['acc'] is not None]
            if len(group) < 3:
                continue
            ds = np.array([r['d_rel'] for r in group])
            accs = np.array([r['acc'] for r in group])
            within.append(_spearman(ds, accs))
            pooled_d.extend(ds - ds.mean())
            pooled_a.extend(accs - accs.mean())
        per_fraction[fraction] = dict(
            within_seed=(float(np.nanmean(within)),
                         float(np.nanstd(within, ddof=1 if len(within) > 1 else 0))) if within
            else (float('nan'), float('nan')),
            n_seeds=len(within),
            pooled_centred=_spearman(pooled_d, pooled_a),
            n_points=len(pooled_d),
        )
    return per_fraction


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
def aggregate(scored):
    """(fraction, method, K) -> mean/std over seeds of acc, D, D_rel."""
    buckets = {}
    for r in scored:
        key = (r['fraction'], r['method'], r['k'])
        b = buckets.setdefault(key, dict(acc=[], d=[], d_rel=[]))
        if r['acc'] is not None:
            b['acc'].append(r['acc'])
        b['d'].append(r['d'])
        b['d_rel'].append(r['d_rel'])

    out = []
    for (fraction, method, k), b in buckets.items():
        out.append(dict(fraction=fraction, method=method, k=k, n=len(b['d']),
                        acc=_mean_std(b['acc']), d=_mean_std(b['d']),
                        d_rel=_mean_std(b['d_rel'])))
    out.sort(key=lambda r: (r['fraction'], r['method'] == RANDOM_METHOD,
                            r['d_rel'][0] if np.isfinite(r['d_rel'][0]) else np.inf))
    return out


def _mean_std(values):
    if not values:
        return (float('nan'),) * 2
    return (float(np.mean(values)),
            float(np.std(values, ddof=1 if len(values) > 1 else 0)))


def print_table(agg, logger):
    header = (f"{'frac':<6}{'method':<34}{'K':<5}{'n':<4}"
              f"{'test acc %':<17}{'D(alpha)':<22}{'D / E_U[D]':<18}")
    logger.info("\n" + header + "\n" + "-" * len(header))
    last = None
    for r in agg:
        if last is not None and r['fraction'] != last:
            logger.info("-" * len(header))
        last = r['fraction']
        name = 'random (fresh draws)' if r['method'] == RANDOM_METHOD else r['method']
        k_disp = str(r['k']) if r['k'] is not None else '–'
        acc = (f"{r['acc'][0]:6.2f} ± {r['acc'][1]:<7.2f}"
               if np.isfinite(r['acc'][0]) else f"{'–':<14}")
        logger.info(f"{r['fraction']:<6}{name:<34}{k_disp:<5}{r['n']:<4}{acc:<17}"
                    f"{r['d'][0]:9.4g} ± {r['d'][1]:<9.3g}"
                    f"{r['d_rel'][0]:7.4f} ± {r['d_rel'][1]:<8.4f}")


def print_correlations(corr, logger):
    logger.info("\nSpearman(D / E_U[D], test accuracy) — negative means "
                "lower objective goes with higher accuracy, i.e. the "
                "objective is doing its job.\n")
    header = f"{'frac':<8}{'within-seed (mean ± std)':<30}{'seeds':<8}{'pooled, centred':<20}{'n':<5}"
    logger.info(header + "\n" + "-" * len(header))
    for fraction, c in sorted(corr.items()):
        m, s = c['within_seed']
        logger.info(f"{fraction:<8}{m:7.3f} ± {s:<20.3f}{c['n_seeds']:<8}"
                    f"{c['pooled_centred']:<20.3f}{c['n_points']:<5}")


def default_out_csv(scratch, dataset, architecture, fractions, variant=''):
    """Per-subset table path. A single-fraction run carries a fraction tag so
    that scoring 0.2 afterwards does not overwrite the 0.1 table."""
    tag = f'_f{fractions[0]:g}' if len(fractions) == 1 else ''
    return os.path.join(
        tree(scratch, 'data_selection_baselines', variant),
        f'd_exact_{dataset}_{architecture}{tag}.csv')


def write_csv(path, scored, dataset, architecture):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'architecture', 'fraction', 'budget', 'method',
                    'num_checkpoints_tag', 'seed', 'acc', 'd_alpha', 'd_rel',
                    'e_u_d', 'indices_path', 'scoring_trajectory'])
        for r in scored:
            w.writerow([dataset, architecture, r['fraction'], r['budget'], r['method'],
                        r['k'] if r['k'] is not None else '', r['seed'],
                        '' if r['acc'] is None else r['acc'],
                        r['d'], r['d_rel'], r['e_u_d'],
                        r.get('indices_path', ''), r.get('trajectory', '')])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='cifar100', choices=list(N_TRAIN))
    p.add_argument('--architecture', default=None, choices=list(common.ARCHITECTURES))
    p.add_argument('--data_dir', required=True,
                   help='Dataset root. Needs the real images: D is computed from gradients. '
                        'Use a persistent path, not $SLURM_TMPDIR from another job.')
    p.add_argument('--checkpoint_dir', default=None,
                   help='Reference trajectory directory or a template containing {seed}. '
                        'Default: $SCRATCH/data_selection_swap/<dataset>/<architecture>/'
                        'checkpoints/seed_{seed}. A distinct, manifest-validated trajectory '
                        'is required for every split seed.')
    p.add_argument('--scratch', default=os.environ.get('SCRATCH', '/home/mila/a/ahmedm/scratch'))
    p.add_argument('--variant', default='sgd', choices=['sgd', 'adam'],
                   help="Result tree. 'sgd' is the historical layout; 'adam' reads "
                        "and writes the adam/ subfolders.")
    p.add_argument('--fractions', default=[0.1, 0.2, 0.4, 0.5], type=float, nargs='+')
    p.add_argument('--seeds', default=[42, 43, 44, 45, 46], type=int, nargs='+')
    p.add_argument('--methods', default=None, nargs='+',
                   help='Default: kl_faithful + every method found on disk.')
    p.add_argument('--checkpoints', default=None, type=int, nargs='+',
                   help="Restrict to runs tagged with these K values (the runs' own tag, "
                        "not the scoring trajectory).")

    p.add_argument('--num_checkpoints', default=10, type=int,
                   help='Common scoring trajectory: K checkpoints (0 = all).')
    p.add_argument('--checkpoint_selection', default='spread', choices=['spread', 'prefix'],
                   help="'spread' spaces the K checkpoints over the whole run; 'prefix' "
                        'takes the first K, which is normally epochs 1-10.')
    p.add_argument('--checkpoint_weighting', default='point',
                   choices=list(ref.CHECKPOINT_WEIGHTINGS),
                   help="'point' uses eta_k at each checkpoint; 'window' integrates "
                        "eta over the optimizer steps represented by that checkpoint.")
    p.add_argument('--checkpoint_epochs', default=None, type=int, nargs='+',
                   help='Choose the scoring trajectory by epoch number instead, e.g. '
                        '1 5 10 20 40 70 110 160 220 300. Overrides the two above.')
    p.add_argument('--weighting', '--optimizer', dest='weighting', default=None,
                   choices=list(ref.OPTIMIZERS),
                   help='(w_t, M_t), shared with the selection scripts. See '
                        'trajectory.py. Default uniform, or adam when --variant adam. '
                        'The adam variants need a scripts/train/reference.py '
                        '--optimizer adam trajectory.')
    p.add_argument('--metric', default='auto', choices=list(ref.METRICS),
                   help='Override M_t independently of the step weight, e.g. '
                        '--weighting uniform --metric adam for a flat-eta prefix '
                        'trajectory.')
    p.add_argument('--momentum_response', default='persistent',
                   choices=list(ref.MOMENTUM_RESPONSES),
                   help="a_s in w_t; 'persistent' (a_s = 1) is the default.")
    p.add_argument('--base_lr', default=0.01, type=float,
                   help="Base LR of the reference trajectory, for the 'sgd' weighting on a "
                        'legacy checkpoint directory.')
    p.add_argument('--adam_eps', default=1e-8, type=float)
    p.add_argument('--beta1', default=0.9, type=float)
    p.add_argument('--scoring_transform', default='test', choices=['train', 'test'],
                   help="View the pool is scored through. 'test' (default here) is the clean "
                        "one; either way the view is materialised once, so D is reproducible.")
    p.add_argument('--n_random', default=3, type=int,
                   help='Fresh random subsets per (seed, budget) for the E_U[D] reference.')
    p.add_argument('--batch_size', default=256, type=int)
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--gbar_cache_dir', default=None,
                   help='Cache the full-pool mean gradients here (~90 MB per checkpoint per '
                        'seed). Default when unset: '
                        '$SCRATCH/data_selection_swap[/adam]/gbar_cache.')
    p.add_argument('--out_csv', default=None,
                   help='Default: $SCRATCH/data_selection_baselines/'
                        'd_exact_<dataset>_<architecture>[_f<fraction>].csv. The '
                        'fraction tag is added when exactly one --fractions value is '
                        'given, so single-budget tables do not overwrite each other.')
    p.add_argument('--download', action='store_true', default=False)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger('score_subsets')

    cfg = common.get_dataset_config(args.dataset, args.data_dir, download=args.download,
                                    architecture=args.architecture)
    architecture = cfg['architecture']
    if args.weighting is None:
        args.weighting = 'adam' if args.variant == 'adam' else 'uniform'
    if args.gbar_cache_dir is None:
        args.gbar_cache_dir = os.path.join(
            tree(args.scratch, 'data_selection_swap', args.variant), 'gbar_cache')
    methods = args.methods or discover_methods(
        args.scratch, args.dataset, architecture, variant=args.variant)

    rows = collect_rows(
        args.scratch, args.dataset, architecture, args.fractions, args.seeds,
        methods, args.checkpoints, variant=args.variant)
    if not rows:
        logger.error(f"No runs found under {args.scratch}. Nothing to score.")
        return
    logger.info(f"Found {len(rows)} saved subsets across {len(methods)} methods.")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_template = args.checkpoint_dir or os.path.join(
        tree(args.scratch, 'data_selection_swap', args.variant),
        args.dataset, cfg['architecture'], 'checkpoints', 'seed_{seed}')

    scored = []

    for seed in args.seeds:
        seed_rows = [r for r in rows if r['seed'] == seed]
        if not seed_rows:
            continue

        trainset, _, _, n_train = common.make_loaders(
            cfg, seed, args.batch_size, args.num_workers, args.scoring_transform)
        pool_indices_sha256 = common.ordered_indices_fingerprint(trainset.indices)
        checkpoint_dir = locate_checkpoint_dir(checkpoint_template, seed)
        checkpoints, _, description = ref.resolve(
            checkpoint_dir, optimizer=args.weighting,
            num_checkpoints=args.num_checkpoints, selection=args.checkpoint_selection,
            epochs=args.checkpoint_epochs, n_train=n_train,
            batch_size=args.batch_size, base_lr=args.base_lr,
            num_epochs=cfg['num_epochs'], beta1=args.beta1,
            momentum=args.momentum_response, metric=args.metric,
            checkpoint_weighting=args.checkpoint_weighting,
            expected_dataset=args.dataset, expected_seed=seed,
            expected_architecture=cfg.get('architecture'),
            expected_pool_indices_sha256=pool_indices_sha256)
        ckpt_paths = [c.path for c in checkpoints]
        weights = [c.weight for c in checkpoints]

        metrics = None
        if ref.resolve_metric(args.weighting, args.metric):
            named = list(common.build_model(
                cfg, cfg['num_classes'], 'cpu').named_parameters())
            metrics = [
                ref.adam_metric_flat(c, named, eps=args.adam_eps, power=2).to(device)
                for c in checkpoints]
            scale = max(float(M.mean()) for M in metrics)
            metrics = [M / scale for M in metrics]
            logger.info(f"Metric M_k = D_k^2 rescaled by 1/{scale:.4g} "
                        f"(global; D is in those units, so compare D_rel, not D)")

        trajectory = f"{description}, view={args.scoring_transform}, seed={seed}"
        logger.info(f"Scoring trajectory: {trajectory}")
        logger.info("\n" + ref.describe(checkpoints))

        scorer = d_alpha.TrajectoryScorer(
            cfg, trainset, ckpt_paths, weights, device,
            scoring_transform=args.scoring_transform, batch_size=args.batch_size,
            num_workers=args.num_workers, metrics=metrics,
            gbar_cache_dir=args.gbar_cache_dir,
            cache_tag=f"{args.dataset}_seed{seed}_{args.scoring_transform}", logger=logger)
        logger.info(f"seed {seed}: d={scorer.d:,} trainable params, n={scorer.n}, "
                    f"{len(seed_rows)} saved subsets + "
                    f"{args.n_random * len(args.fractions)} random references")

        subsets = {}
        for i, r in enumerate(seed_rows):
            idx = common.torch_load(r['indices_path'], map_location='cpu')
            idx = torch.as_tensor(idx, dtype=torch.long).reshape(-1)
            if idx.unique().numel() != idx.numel():
                logger.warning(f"{r['indices_path']} contains duplicate indices; "
                               f"m is taken as the stored count ({idx.numel()}).")
            subsets[('run', i)] = idx
        for fraction in args.fractions:
            budget = int(fraction * N_TRAIN[args.dataset])
            for rep, idx in random_subsets(n_train, budget, seed, args.n_random).items():
                subsets[('rand', fraction, rep)] = idx

        values = scorer.score(subsets)

        e_u_d = {}
        for fraction in args.fractions:
            draws = [values[k] for k in values
                     if k[0] == 'rand' and k[1] == fraction]
            e_u_d[fraction] = float(np.mean(draws)) if draws else float('nan')

        for i, r in enumerate(seed_rows):
            d = values[('run', i)]
            scored.append({**r, 'd': d, 'e_u_d': e_u_d[r['fraction']],
                           'trajectory': trajectory,
                           'd_rel': d / e_u_d[r['fraction']]})
        for key, value in values.items():
            if key[0] != 'rand':
                continue
            _, fraction, rep = key
            scored.append(dict(fraction=fraction,
                               budget=int(fraction * N_TRAIN[args.dataset]),
                               method=RANDOM_METHOD, k=None, seed=seed, acc=None,
                               d=value, e_u_d=e_u_d[fraction],
                               trajectory=trajectory,
                               d_rel=value / e_u_d[fraction]))

        del scorer
        if device == 'cuda':
            torch.cuda.empty_cache()

    print_table(aggregate(scored), logger)
    print_correlations(correlations(scored, args.fractions, args.seeds), logger)

    out_csv = args.out_csv or default_out_csv(
        args.scratch, args.dataset, architecture, args.fractions, args.variant)
    write_csv(out_csv, scored, args.dataset, architecture)
    logger.info(f"\nPer-subset results → {out_csv}")


if __name__ == "__main__":
    main()
