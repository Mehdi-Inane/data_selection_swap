#!/usr/bin/env python3
"""Does it matter that architectures select different examples?

``analysis.selection_overlap`` establishes *that* two architectures pick
different subsets. This answers what that costs: every architecture's saved
subset is scored against every architecture's trajectory kernel, giving a
source-by-target matrix of

    D_rel^{A->B} = D(alpha_A | trajectory_B) / E_U[D | trajectory_B]

one entry per (fraction, method, seed). The diagonal is the matched case each
architecture was selected for; an off-diagonal entry is what B would have got
had it imported A's subset. Their ratio,

    transfer penalty = D_rel^{A->B} / D_rel^{B->B},

is the whole point: 1.0 means the subsets are interchangeable as far as the
objective is concerned and the overlap difference is cosmetic, while a value
approaching ``1 / D_rel^{B->B}`` means the imported subset is no better than
drawing at random. No retraining is involved.

Why one architecture's indices are meaningful to another
────────────────────────────────────────────────────────
The pool is the whole training set (``common.make_train_set``): the same images
in the same order for every architecture and seed, so a saved index tensor is a
valid set of pool positions no matter which architecture produced it.

Relation to analysis.score_subsets
──────────────────────────────────
The same exact all-parameter objective, the same E_U[D] normalisation, and the
same trajectory knobs, so pass the values the scoring runs used. The diagonal
of this matrix recomputes what ``score_subsets`` already wrote for that
architecture, which makes it a free consistency check on the whole setup.

Costs one trajectory build per (target architecture, seed), and scores every
source architecture's subsets inside that one build — the checkpoint-outer loop
in ``objective.TrajectoryScorer`` shares each loaded checkpoint across all of
them, so adding source architectures is nearly free once the target is up.

Usage:
    python scripts/analyze/transfer_d.py --dataset cifar10 --data_dir /path/to/cifar10_data
    python scripts/analyze/transfer_d.py --dataset cifar10 --data_dir ... \\
        --target_architectures resnet18 --fractions 0.1 --weighting sgd
"""

import argparse
import csv
import logging
import os
import sys

import numpy as np
import torch

from .. import common
from .. import objective as d_alpha
from .. import trajectory as ref
from ..paths import default_results_root, tree
from .comparison import N_TRAIN, RANDOM_METHOD
from .score_subsets import random_subsets
from .selection_overlap import collect_subsets, run_label

FIELDS = [
    'dataset', 'fraction', 'budget', 'method', 'k', 'seed',
    'source_architecture', 'target_architecture',
    'd_alpha', 'e_u_d', 'd_rel', 'd_rel_matched', 'transfer_penalty',
    'indices_path', 'scoring_trajectory',
]


def resolve_checkpoint_dir(template, seed):
    """Trajectory directory for one seed, tolerating a seedless template.

    The launchers point at ``.../<architecture>/checkpoints``, while the
    per-seed trajectories live one level down; both spellings are accepted,
    and the per-seed directory wins over a stale manifest in the parent.
    """
    return ref.locate_seed_dir(template, seed)


def build_scorer(cfg, args, seed, target_architecture, logger):
    """Trajectory scorer for one (target architecture, seed), and its description."""
    trainset, _, _, n_train = common.make_loaders(
        cfg, seed, args.batch_size, args.num_workers, args.scoring_transform)
    pool_hash = common.ordered_indices_fingerprint(trainset.indices)

    template = args.checkpoint_dir or os.path.join(
        tree(args.scratch, 'data_selection_swap', args.variant),
        args.dataset, target_architecture, 'checkpoints', 'seed_{seed}')
    checkpoints, _, description = ref.resolve(
        resolve_checkpoint_dir(template, seed), optimizer=args.weighting,
        num_checkpoints=args.num_checkpoints, selection=args.checkpoint_selection,
        epochs=args.checkpoint_epochs, n_train=n_train,
        batch_size=args.reference_batch_size, base_lr=args.base_lr,
        num_epochs=cfg['num_epochs'], beta1=args.beta1,
        momentum=args.momentum_response, metric=args.metric,
        checkpoint_weighting=args.checkpoint_weighting,
        expected_dataset=args.dataset, expected_seed=seed,
        expected_architecture=cfg.get('architecture'),
        expected_pool_indices_sha256=pool_hash)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics = None
    if ref.resolve_metric(args.weighting, args.metric):
        named = list(common.build_model(cfg, cfg['num_classes'], 'cpu').named_parameters())
        metrics = [ref.adam_metric_flat(c, named, eps=args.adam_eps, power=2).to(device)
                   for c in checkpoints]
        scale = max(float(M.mean()) for M in metrics)
        metrics = [M / scale for M in metrics]
        logger.info(f"Metric M_k = D_k^2 rescaled by 1/{scale:.4g}; compare D_rel, not D.")

    trajectory = (f"{target_architecture}: {description}, "
                  f"view={args.scoring_transform}, seed={seed}")
    logger.info(f"Scoring trajectory: {trajectory}")

    scorer = d_alpha.TrajectoryScorer(
        cfg, trainset, [c.path for c in checkpoints], [c.weight for c in checkpoints],
        device, scoring_transform=args.scoring_transform, batch_size=args.batch_size,
        num_workers=args.num_workers, metrics=metrics,
        gbar_cache_dir=args.gbar_cache_dir,
        cache_tag=f"{args.dataset}_{target_architecture}_seed{seed}_{args.scoring_transform}",
        logger=logger)
    return scorer, trajectory, n_train


def score_target(records, cfg, args, seed, target_architecture, logger):
    """Score every source architecture's subsets on this target's trajectory."""
    seed_records = [record for record in records if record['seed'] == seed]
    if not seed_records:
        return []

    scorer, trajectory, n_train = build_scorer(
        cfg, args, seed, target_architecture, logger)
    logger.info(f"target={target_architecture} seed={seed}: scoring "
                f"{len(seed_records)} saved subsets from "
                f"{len({r['architecture'] for r in seed_records})} source "
                f"architecture(s) + {args.n_random * len(args.fractions)} random draws.")

    subsets = {('run', i): torch.as_tensor(record['indices'], dtype=torch.long)
               for i, record in enumerate(seed_records)}
    for fraction in args.fractions:
        budget = int(fraction * N_TRAIN[args.dataset])
        for rep, indices in random_subsets(n_train, budget, seed, args.n_random).items():
            subsets[('rand', fraction, rep)] = indices

    values = scorer.score(subsets)
    e_u_d = {}
    for fraction in args.fractions:
        draws = [values[key] for key in values
                 if key[0] == 'rand' and key[1] == fraction]
        e_u_d[fraction] = float(np.mean(draws)) if draws else float('nan')

    rows = []
    for i, record in enumerate(seed_records):
        value = values[('run', i)]
        rows.append(dict(
            dataset=args.dataset, fraction=record['fraction'], budget=record['budget'],
            method=record['method'], k='' if record['k'] is None else record['k'],
            seed=seed, source_architecture=record['architecture'],
            target_architecture=target_architecture,
            d_alpha=value, e_u_d=e_u_d[record['fraction']],
            d_rel=value / e_u_d[record['fraction']],
            indices_path=record['indices_path'], scoring_trajectory=trajectory))
    for key, value in values.items():
        if key[0] != 'rand':
            continue
        _, fraction, _ = key
        rows.append(dict(
            dataset=args.dataset, fraction=fraction,
            budget=int(fraction * N_TRAIN[args.dataset]), method=RANDOM_METHOD, k='',
            seed=seed, source_architecture=target_architecture,
            target_architecture=target_architecture,
            d_alpha=value, e_u_d=e_u_d[fraction], d_rel=value / e_u_d[fraction],
            indices_path='', scoring_trajectory=trajectory))

    del scorer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def attach_transfer_penalty(rows):
    """Divide each entry by the target architecture's own matched D_rel.

    Left blank where the target never ran that (method, K, seed, fraction)
    itself: without its own diagonal there is nothing to be penalised against.
    """
    matched = {}
    for row in rows:
        if (row['source_architecture'] == row['target_architecture']
                and row['method'] != RANDOM_METHOD):
            key = (row['target_architecture'], row['fraction'], row['method'],
                   row['k'], row['seed'])
            matched[key] = row['d_rel']
    for row in rows:
        key = (row['target_architecture'], row['fraction'], row['method'],
               row['k'], row['seed'])
        reference = matched.get(key)
        row['d_rel_matched'] = reference if reference is not None else float('nan')
        row['transfer_penalty'] = (row['d_rel'] / reference
                                   if reference not in (None, 0.0) else float('nan'))
    return rows


def transfer_matrix(rows, fraction, method):
    """``(sources, targets, mean D_rel, mean transfer penalty)`` for one cell."""
    selected = [row for row in rows if row['fraction'] == fraction
                and row['method'] == method and row['method'] != RANDOM_METHOD]
    if not selected:
        return [], [], np.zeros((0, 0)), np.zeros((0, 0))
    sources = sorted({row['source_architecture'] for row in selected})
    targets = sorted({row['target_architecture'] for row in selected})
    d_rel = np.full((len(sources), len(targets)), np.nan)
    penalty = np.full((len(sources), len(targets)), np.nan)
    for i, source in enumerate(sources):
        for j, target in enumerate(targets):
            cell = [row for row in selected if row['source_architecture'] == source
                    and row['target_architecture'] == target]
            if not cell:
                continue
            d_rel[i, j] = float(np.mean([row['d_rel'] for row in cell]))
            finite = [row['transfer_penalty'] for row in cell
                      if np.isfinite(row['transfer_penalty'])]
            if finite:
                penalty[i, j] = float(np.mean(finite))
    return sources, targets, d_rel, penalty


def print_matrices(rows, logger):
    methods = sorted({row['method'] for row in rows if row['method'] != RANDOM_METHOD})
    fractions = sorted({row['fraction'] for row in rows})
    logger.info(
        "\nD_rel of each source architecture's subset under each target's trajectory.\n"
        "D_rel = 1.0 is a random subset; lower is more faithful. The penalty in "
        "brackets is\nthe entry over the target's own matched subset, so 1.00 down "
        "the diagonal and\nhigher off it means the imported subset is worse for that "
        "target.\n")
    for fraction in fractions:
        for method in methods:
            sources, targets, d_rel, penalty = transfer_matrix(rows, fraction, method)
            if not sources:
                continue
            logger.info(f"fraction {fraction:g}, {method}")
            corner = 'selected by \\ scored on'
            header = f"{corner:<26}" + ''.join(f"{t:<22}" for t in targets)
            logger.info(header + "\n" + "-" * len(header))
            for i, source in enumerate(sources):
                cells = ''.join(
                    f"{d_rel[i, j]:.4f} ({penalty[i, j]:.2f})".ljust(22)
                    if np.isfinite(d_rel[i, j]) else f"{'–':<22}"
                    for j in range(len(targets)))
                logger.info(f"{source:<26}{cells}")
            logger.info("")


def plot_transfer_matrix(sources, targets, penalty, out_stem, title=None,
                         formats=('pdf',)):
    """Heatmap of the transfer penalty, 1.0 (interchangeable) as the neutral point."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from .selection_overlap import RC, WIDTHS_IN, _save

    finite = penalty[np.isfinite(penalty)]
    if not finite.size:
        return []
    spread = max(float(np.max(np.abs(finite - 1.0))), 0.02)

    with plt.rc_context(RC):
        size = WIDTHS_IN['onehalf']
        fig, ax = plt.subplots(figsize=(size, size * 0.92))
        image = ax.imshow(np.ma.masked_invalid(penalty), cmap='coolwarm',
                          vmin=1.0 - spread, vmax=1.0 + spread)
        image.cmap.set_bad('0.92')
        ax.set_xticks(range(len(targets)))
        ax.set_yticks(range(len(sources)))
        ax.set_xticklabels(targets, rotation=30, ha='right')
        ax.set_yticklabels(sources)
        ax.set_xlabel('trajectory scoring the subset')
        ax.set_ylabel('architecture that selected it')
        for i in range(len(sources)):
            for j in range(len(targets)):
                if np.isfinite(penalty[i, j]):
                    ax.text(j, i, f"{penalty[i, j]:.2f}", ha='center', va='center',
                            fontsize=7, color='black')
        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        bar.set_label(r'$D_{rel}^{A \to B} / D_{rel}^{B \to B}$', fontsize=8)
        if title:
            ax.set_title(title)
        fig.tight_layout()
        written = _save(fig, out_stem, formats)
        plt.close(fig)
    return written


def draw_all(rows, out_dir, dataset, title=True, formats=('pdf',), logger=None):
    logger = logger or logging.getLogger('transfer_d')
    written = []
    for fraction in sorted({row['fraction'] for row in rows}):
        for method in sorted({row['method'] for row in rows
                              if row['method'] != RANDOM_METHOD}):
            sources, targets, _, penalty = transfer_matrix(rows, fraction, method)
            if len(sources) < 2:
                continue
            written += plot_transfer_matrix(
                sources, targets, penalty,
                os.path.join(out_dir, f"transfer_d_{dataset}_{method}_f{fraction:g}"),
                f"{dataset} / {method}, fraction {fraction:g}" if title else None,
                formats)
    if written:
        logger.info(f"Wrote {len(written)} figure(s) -> {out_dir}")
    return written


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='cifar10', choices=list(N_TRAIN))
    p.add_argument('--source_architectures', default=['resnet18', 'vgg16', 'inception'],
                   nargs='+', help='Architectures whose saved subsets are scored.')
    p.add_argument('--target_architectures', default=None, nargs='+',
                   help='Architectures whose trajectories do the scoring. Default: the '
                        'source list, giving the full square matrix. Restrict it to one '
                        'target to split the work across array tasks.')
    p.add_argument('--data_dir', required=True,
                   help='Dataset root. D is computed from gradients, so this needs the '
                        'real images on a persistent path, not another job\'s $SLURM_TMPDIR.')
    p.add_argument('--checkpoint_dir', default=None,
                   help='Trajectory directory or a template containing {seed}. Default: '
                        '$SCRATCH/data_selection_swap/<dataset>/<target>/checkpoints/'
                        'seed_{seed}. The target architecture\'s own trajectory, since '
                        'that is what defines the objective being transferred to.')
    p.add_argument('--scratch', default=default_results_root())
    p.add_argument('--variant', default='sgd', choices=['sgd', 'adam'],
                   help="Result tree. 'sgd' is the historical layout; 'adam' uses "
                        "the adam/ subfolders.")
    p.add_argument('--fractions', default=[0.1], type=float, nargs='+')
    p.add_argument('--seeds', default=[42, 43, 44, 45, 46], type=int, nargs='+')
    p.add_argument('--methods', default=None, nargs='+',
                   help='Default: every method that saved indices on disk.')
    p.add_argument('--checkpoints', default=None, type=int, nargs='+',
                   help="Restrict to subsets tagged with these K values.")

    p.add_argument('--num_checkpoints', default=0, type=int,
                   help='Scoring trajectory: K checkpoints (0 = all).')
    p.add_argument('--checkpoint_selection', default='spread', choices=['spread', 'prefix'])
    p.add_argument('--checkpoint_weighting', default='point',
                   choices=list(ref.CHECKPOINT_WEIGHTINGS))
    p.add_argument('--checkpoint_epochs', default=None, type=int, nargs='+')
    p.add_argument('--weighting', '--optimizer', dest='weighting', default='sgd',
                   choices=list(ref.OPTIMIZERS))
    p.add_argument('--metric', default='none', choices=list(ref.METRICS))
    p.add_argument('--momentum_response', default='persistent',
                   choices=list(ref.MOMENTUM_RESPONSES))
    p.add_argument('--base_lr', default=0.01, type=float)
    p.add_argument('--adam_eps', default=1e-8, type=float)
    p.add_argument('--beta1', default=0.9, type=float)
    p.add_argument('--scoring_transform', default='test', choices=['train', 'test'])
    p.add_argument('--n_random', default=3, type=int,
                   help='Fresh random subsets per (seed, budget) for the E_U[D] '
                        'reference, redrawn under every target trajectory.')
    p.add_argument('--batch_size', default=256, type=int,
                   help='Batch size of the scoring sweeps only.')
    p.add_argument('--reference_batch_size', default=128, type=int,
                   help='Batch size the reference trajectories were TRAINED with '
                        '(scripts/slurm/run_full_data.sh uses 128). Independent of '
                        '--batch_size.')
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--gbar_cache_dir', default=None)
    p.add_argument('--out_csv', default=None,
                   help='Default: $SCRATCH/data_selection_baselines/'
                        'transfer_d_<dataset>[_f<fraction>][_to_<target>].csv')
    p.add_argument('--plot_dir', default=None)
    p.add_argument('--no_plots', action='store_true', default=False)
    p.add_argument('--formats', nargs='+', default=['pdf'])
    p.add_argument('--no_title', action='store_true', default=False)
    p.add_argument('--download', action='store_true', default=False)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger('transfer_d')

    targets = args.target_architectures or list(args.source_architectures)
    records = collect_subsets(args.scratch, args.dataset, args.source_architectures,
                              args.fractions, args.seeds, args.methods,
                              args.checkpoints, logger, variant=args.variant)
    if not records:
        logger.error(f"No saved subsets found under {args.scratch}. Nothing to score.")
        return []
    logger.info(f"Loaded {len(records)} saved subsets: "
                f"{sorted({r['architecture'] for r in records})} x "
                f"{sorted({run_label(r) for r in records})}.")

    rows = []
    for target in targets:
        cfg = common.get_dataset_config(args.dataset, args.data_dir,
                                        download=args.download, architecture=target)
        for seed in args.seeds:
            rows += score_target(records, cfg, args, seed, target, logger)

    attach_transfer_penalty(rows)
    print_matrices(rows, logger)

    tag = f"_f{args.fractions[0]:g}" if len(args.fractions) == 1 else ''
    if args.target_architectures and len(targets) == 1:
        tag += f"_to_{targets[0]}"
    out_csv = args.out_csv or os.path.join(
        tree(args.scratch, 'data_selection_baselines', args.variant),
        f'transfer_d_{args.dataset}{tag}.csv')
    logger.info(f"Per-subset transfer results -> {write_csv(out_csv, rows)}")

    if not args.no_plots:
        plot_dir = args.plot_dir or os.path.join(
            tree(args.scratch, 'data_selection_baselines', args.variant), 'plots')
        draw_all(rows, plot_dir, args.dataset, not args.no_title,
                 tuple(args.formats), logger)
    return rows


if __name__ == '__main__':
    main()
