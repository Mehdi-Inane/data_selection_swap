"""Store the full-network kernel once, then score every saved subset from it.

The objective on a subset ``I`` of size ``m`` is

    D(I) = sum_k w_k || u_I^k / m - gbar_k ||^2
         = (1/m^2) alpha^T P alpha + q(m)^T alpha + c

with ``alpha`` the 0/1 indicator of ``I``. ``P`` and ``c`` do not depend on
the subset or on ``m``. The linear term does:

    q(m) = - (2/m) r,    r_i = sum_k w_k <g_i^k, gbar_k>,
    c = sum_k w_k ||gbar_k||^2.

An exact ``P`` in parameter space is not something we can form: one
checkpoint of per-example full gradients is on the order of a terabyte.
What is stored instead is the same identity after a count sketch
``R: R^d -> R^p`` (the map used by ``selection.kl_full``):

    P = sum_k w_k (R G_k) (R G_k)^T,    r = sum_k w_k (R G_k) (R gbar_k).

``subset_d`` is then exactly ``sum_k w_k ||R(u_I^k / m - gbar_k)||^2``.
Rows of ``P`` are positions in the seed's training pool, which is also
the indexing of every ``*_indices.pt``.

The kernel depends on the seed. Checkpoints live in ``seed_<seed>`` (the pool
itself is the full training set for every seed), so one
``(dataset, architecture)`` job stores one file per seed.
"""

import argparse
import json
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..paths import default_results_root, tree
from .comparison import N_TRAIN
from .result_plots import (
    KL_DISPLAY_NAME,
    attach_cached_d,
    baseline_comparison_rows,
    collect_runs,
    comparison_accuracy_series,
    emphasizes_k1,
    group_by_budget,
    group_error_vs_d,
    is_kl_method,
    k_accuracy_series,
    latex_path_for,
    locate_checkpoint_dir,
    plot_accuracy_vs_fraction,
    plot_error_vs_d,
    plot_vs_budget,
    scoring_id,
    with_accuracy,
    write_latex,
    write_table,
)

EXPERIMENT_DATASETS = ('cifar10', 'cifar100')
EXPERIMENT_ARCHITECTURES = ('resnet18', 'vgg16', 'inception')
EXPERIMENT_SEEDS = (42, 43, 44, 45, 46)
EXPERIMENT_FRACTIONS = (0.1, 0.2, 0.4, 0.5)


def kernel_scoring_id(dataset, architecture, weighting, metric, num_checkpoints,
                      checkpoint_selection, checkpoint_weighting,
                      scoring_transform, checkpoint_epochs, proj_dim, proj_seed):
    base = scoring_id(
        dataset, architecture, weighting, metric, num_checkpoints,
        checkpoint_selection, checkpoint_weighting, scoring_transform,
        checkpoint_epochs)
    return f"{base}|proj={int(proj_dim)}|proj_seed={int(proj_seed)}"


def kernel_path(scratch, dataset, architecture, seed, proj_dim, variant=''):
    return os.path.join(
        tree(scratch, 'data_selection_swap', variant), 'kernels', dataset, architecture,
        f'seed_{seed}', f'kernel_p{int(proj_dim)}.pt')


def budget_q(r, m):
    """``q(m) = -(2/m) r``. ``r`` is what the kernel file stores."""
    m = int(m)
    if m <= 0:
        raise ValueError(f"budget m must be positive, got {m}")
    return (-2.0 / float(m)) * r


def subset_d(P, r, c, indices):
    """``D = alpha^T P alpha / m^2 + q(m)^T alpha + c`` for a 0/1 subset."""
    idx = torch.as_tensor(indices, dtype=torch.long).reshape(-1).cpu()
    m = int(idx.numel())
    if m == 0:
        raise ValueError("subset is empty")
    if int(idx.min()) < 0 or int(idx.max()) >= P.shape[0]:
        raise IndexError(
            f"subset index outside 0..{P.shape[0] - 1} "
            f"(min {int(idx.min())}, max {int(idx.max())})")
    quad = P.index_select(0, idx).index_select(1, idx).double().sum()
    linear = budget_q(r, m).index_select(0, idx).double().sum()
    return float(quad / (m * m) + linear + float(c))


def accumulate_gram(P, r, c, sketches, weight, row_block=2048):
    """Add ``weight * S S^T`` and ``weight * S sbar`` into ``P``, ``r``, ``c``.

    ``sketches`` is ``[n, p]``. ``P`` is float32 ``[n, n]`` and ``r`` is
    float64 ``[n]``, both on CPU. Returns the updated scalar ``c``.
    """
    if sketches.dim() != 2 or sketches.shape[0] != P.shape[0]:
        raise ValueError(
            f"sketches must be [{P.shape[0]}, p], got {tuple(sketches.shape)}")
    weight = float(weight)
    sbar = sketches.mean(dim=0)
    linear = (sketches.double() @ sbar.double()).detach().cpu()
    r += weight * linear
    c = float(c) + weight * float(sbar.double().dot(sbar.double()))
    transposed = sketches.T.contiguous()
    n = sketches.shape[0]
    for start in range(0, n, row_block):
        gram = (sketches[start:start + row_block] @ transposed).float()
        P[start:start + row_block] += (weight * gram).detach().cpu()
    return c


def _disable_inplace(model):
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


def example_sketches(model, state_dict, X, Y, sketch, batch_size):
    """Count-sketch of every per-example gradient. Returns ``[n, p]`` float32."""
    from ..selection.kl_full import split_state

    model.eval()
    _disable_inplace(model)
    device = X.device
    params, buffers = split_state(model, state_dict, device)
    order = [name for name, param in model.named_parameters() if param.requires_grad]
    sketch.buckets = sketch.buckets.to(device)
    sketch.signs = sketch.signs.to(device)

    def loss_one(p, x, y):
        logits = torch.func.functional_call(
            model, {**p, **buffers}, (x.unsqueeze(0),))
        return F.cross_entropy(logits.reshape(1, -1), y.reshape(1))

    per_example = torch.func.vmap(
        torch.func.grad(loss_one), in_dims=(None, 0, 0), randomness='error')

    n = int(X.shape[0])
    out = torch.empty(n, sketch.proj_dim, dtype=torch.float32, device=device)
    for start in range(0, n, batch_size):
        xb = X[start:start + batch_size]
        yb = Y[start:start + batch_size]
        grads = per_example(params, xb, yb)
        flat = torch.cat(
            [grads[name].reshape(xb.shape[0], -1) for name in order], dim=1)
        out[start:start + xb.shape[0]] = sketch.apply_batch(flat.float())
        del grads, flat
    return out


def _identity(args, seed, pool_hash, checkpoints, description):
    return {
        'dataset': args.dataset,
        'architecture': args.architecture,
        'seed': int(seed),
        'proj_dim': int(args.proj_dim),
        'proj_seed': int(args.proj_seed),
        'weighting': args.weighting,
        'metric': args.metric,
        'num_checkpoints': int(args.num_checkpoints),
        'checkpoint_selection': args.checkpoint_selection,
        'checkpoint_weighting': args.checkpoint_weighting,
        'scoring_transform': args.scoring_transform,
        'pool_indices_sha256': pool_hash,
        'description': description,
        'checkpoints': [
            {'path': os.path.realpath(c.path), 'epoch': int(c.epoch),
             'weight': float(c.weight)}
            for c in checkpoints
        ],
    }


def _save_payload(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _load_payload(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def precompute_seed(args, seed, logger):
    """Build ``P``, ``r``, and ``c`` for one seed and write ``kernel_p{p}.pt``."""
    from .. import common
    from .. import objective as d_alpha
    from .. import trajectory as ref
    from ..selection.kl_full import CountSketch

    if int(args.proj_dim) <= 0:
        raise ValueError(
            "proj_dim must be positive. The stored P is the Gram of "
            "count-sketched full gradients; an exact n-by-n Gram of the "
            "unsketched gradients is not formed.")
    if ref.resolve_metric(args.weighting, args.metric):
        raise ValueError(
            "This kernel is the Euclidean objective (M = I). "
            "Pass --metric none, or use a weighting whose default metric is none.")

    cfg = common.get_dataset_config(
        args.dataset, args.data_dir, download=args.download,
        architecture=args.architecture)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainset, _, _, n_train = common.make_loaders(
        cfg, seed, args.batch_size, args.num_workers, args.scoring_transform)
    pool_hash = common.ordered_indices_fingerprint(trainset.indices)
    template = args.checkpoint_dir or os.path.join(
        tree(args.scratch, 'data_selection_swap', args.variant),
        args.dataset, cfg['architecture'], 'checkpoints')
    checkpoint_dir = locate_checkpoint_dir(template, seed)
    checkpoints, _, description = ref.resolve(
        checkpoint_dir, optimizer=args.weighting,
        num_checkpoints=args.num_checkpoints,
        selection=args.checkpoint_selection,
        epochs=args.checkpoint_epochs, n_train=n_train,
        batch_size=args.batch_size, base_lr=args.base_lr,
        num_epochs=cfg['num_epochs'], beta1=args.beta1,
        momentum=args.momentum_response, metric=args.metric,
        checkpoint_weighting=args.checkpoint_weighting,
        expected_dataset=args.dataset, expected_seed=seed,
        expected_architecture=cfg.get('architecture'),
        expected_pool_indices_sha256=pool_hash)
    identity = _identity(args, seed, pool_hash, checkpoints, description)
    path = args.kernel_out or kernel_path(
        args.scratch, args.dataset, cfg['architecture'], seed, args.proj_dim,
        args.variant)
    partial = path + '.partial'
    logger.info(
        f"Seed {seed}: {len(checkpoints)} checkpoints ({description}), "
        f"count sketch width {args.proj_dim}, writing {path}")

    pool = common.scoring_subset(cfg, trainset, args.scoring_transform)
    X, Y = d_alpha.materialise(pool, device, args.batch_size, args.num_workers)
    n = int(Y.shape[0])
    model = common.build_model(cfg, cfg['num_classes'], 'cpu').eval()
    _disable_inplace(model)
    param_dim = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sketch = CountSketch(param_dim, int(args.proj_dim), int(args.proj_seed))

    completed = 0
    P = torch.zeros(n, n, dtype=torch.float32)
    r = torch.zeros(n, dtype=torch.float64)
    c = 0.0
    if os.path.isfile(partial) and not args.restart:
        previous = _load_payload(partial)
        previous_id = {key: previous['meta'].get(key) for key in identity}
        if previous_id == identity and tuple(previous['P'].shape) == (n, n):
            P = previous['P'].float().contiguous()
            r = previous['r'].double().contiguous()
            c = float(previous['c'])
            completed = int(previous['meta'].get('completed', 0))
            logger.info(f"Resuming at checkpoint {completed + 1}/{len(checkpoints)}")
        else:
            logger.info("Partial kernel does not match this trajectory; starting over.")
        del previous

    for t, checkpoint in enumerate(checkpoints):
        if t < completed:
            continue
        state = ref.load_weights(checkpoint.path)
        sketches = example_sketches(
            model, state, X, Y, sketch, args.example_batch_size)
        c = accumulate_gram(P, r, c, sketches, checkpoint.weight)
        del sketches, state
        if device == 'cuda':
            torch.cuda.empty_cache()
        done = t + 1
        logger.info(
            f"  checkpoint {done}/{len(checkpoints)} "
            f"(epoch {checkpoint.epoch}, w={checkpoint.weight:.6g})")
        if done == len(checkpoints) or done % int(args.save_every) == 0:
            meta = dict(identity)
            meta['completed'] = done
            meta['n'] = n
            meta['c'] = float(c)
            _save_payload(partial, {'P': P, 'r': r, 'c': float(c), 'meta': meta})

    meta = dict(identity)
    meta['n'] = n
    meta['c'] = float(c)
    meta['q'] = 'q(m) = -2 * r / m; r and c are stored, q is not (it depends on m)'
    _save_payload(path, {'P': P, 'r': r, 'c': float(c), 'meta': meta})
    sidecar = os.path.splitext(path)[0] + '.json'
    with open(sidecar, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2)
        fh.write('\n')
    if os.path.isfile(partial):
        os.remove(partial)
    logger.info(f"Wrote kernel -> {path}")
    return path


def _check_kernel(payload, args, seed):
    meta = payload.get('meta') or {}
    expected = {
        'dataset': args.dataset,
        'architecture': args.architecture,
        'seed': int(seed),
        'proj_dim': int(args.proj_dim),
        'proj_seed': int(args.proj_seed),
        'weighting': args.weighting,
        'metric': args.metric,
        'num_checkpoints': int(args.num_checkpoints),
        'checkpoint_selection': args.checkpoint_selection,
        'checkpoint_weighting': args.checkpoint_weighting,
        'scoring_transform': args.scoring_transform,
    }
    mismatches = [
        f"{key}: file={meta.get(key)!r}, requested={value!r}"
        for key, value in expected.items()
        if str(meta.get(key)) != str(value)
    ]
    if mismatches:
        raise ValueError("Kernel file does not match this score request: "
                         + "; ".join(mismatches))
    return meta


def score_seed_rows(rows, path, args, seed, logger):
    from .. import common

    payload = _load_payload(path)
    _check_kernel(payload, args, seed)
    P, r, c = payload['P'], payload['r'], float(payload['c'])
    for row in rows:
        idx = common.torch_load(row['indices_path'], map_location='cpu')
        row['d'] = subset_d(P, r, c, idx)
    logger.info(f"Scored {len(rows)} subsets for seed {seed} from {path}")
    del payload, P, r


def _add_trajectory_args(p):
    p.add_argument('--variant', default='sgd', choices=['sgd', 'adam'],
                   help="Result tree. 'sgd' is the historical layout; 'adam' uses "
                        "the adam/ subfolders.")
    p.add_argument('--num_checkpoints', default=0, type=int,
                   help='0 = every saved checkpoint.')
    p.add_argument('--checkpoint_selection', default='spread',
                   choices=['spread', 'prefix'])
    p.add_argument('--checkpoint_weighting', default='point',
                   choices=['point', 'window'])
    p.add_argument('--checkpoint_epochs', default=None, type=int, nargs='+')
    p.add_argument('--weighting', '--optimizer', dest='weighting', default='sgd',
                   choices=['uniform', 'sgd', 'adam', 'adam_update'])
    p.add_argument('--metric', default='auto', choices=['auto', 'none', 'adam'])
    p.add_argument('--momentum_response', default='persistent',
                   choices=['persistent', 'isolated'])
    p.add_argument('--base_lr', default=0.01, type=float)
    p.add_argument('--beta1', default=0.9, type=float)
    p.add_argument('--scoring_transform', default='test', choices=['train', 'test'])
    p.add_argument('--proj_dim', default=4096, type=int,
                   help='Count-sketch width. P is the Gram of these sketches.')
    p.add_argument('--proj_seed', default=0, type=int)


def precompute_main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True, choices=list(N_TRAIN))
    p.add_argument('--architecture', required=True,
                   choices=list(EXPERIMENT_ARCHITECTURES))
    p.add_argument('--seed', default=None, type=int,
                   help='One split. Omit to loop --seeds. The kernel is per seed.')
    p.add_argument('--seeds', default=list(EXPERIMENT_SEEDS), type=int, nargs='+')
    p.add_argument('--data_dir', required=True)
    p.add_argument('--scratch', default=default_results_root())
    p.add_argument('--checkpoint_dir', default=None,
                   help='Default: $SCRATCH/data_selection_swap/<dataset>/<architecture>/checkpoints '
                        'with a seed_<seed> child. "{seed}" is substituted.')
    p.add_argument('--kernel_out', default=None,
                   help='Override the output file. Only valid with a single --seed.')
    p.add_argument('--example_batch_size', default=8, type=int,
                   help='Examples per per-example backward. Lower this if the GPU runs out of memory.')
    p.add_argument('--batch_size', default=128, type=int)
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--save_every', default=10, type=int,
                   help='Write a resumable partial kernel every this many checkpoints.')
    p.add_argument('--restart', action='store_true', default=False,
                   help='Ignore a partial kernel and start from the first checkpoint.')
    p.add_argument('--download', action='store_true', default=False)
    _add_trajectory_args(p)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger('precompute_kernel')
    seeds = [args.seed] if args.seed is not None else list(args.seeds)
    if args.kernel_out and len(seeds) != 1:
        raise ValueError("--kernel_out applies to one seed; pass --seed.")
    for seed in seeds:
        precompute_seed(args, seed, logger)


def score_main(argv=None):
    """Fill D for every saved subset from the stored kernel, then write tables and plots."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True, choices=list(N_TRAIN))
    p.add_argument('--architecture', required=True,
                   choices=list(EXPERIMENT_ARCHITECTURES))
    p.add_argument('--fractions', default=list(EXPERIMENT_FRACTIONS), type=float, nargs='+')
    p.add_argument('--seeds', default=list(EXPERIMENT_SEEDS), type=int, nargs='+')
    p.add_argument('--methods', default=None, nargs='+')
    p.add_argument('--scratch', default=default_results_root())
    p.add_argument('--out_dir', default=None,
                   help='Default: $SCRATCH/data_selection_baselines/plots')
    p.add_argument('--cache_csv', default=None)
    _add_trajectory_args(p)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger('score_kernel')

    out_dir = args.out_dir or os.path.join(
        tree(args.scratch, 'data_selection_baselines', args.variant), 'plots')
    cache_path = args.cache_csv or os.path.join(
        out_dir, f"d_kernel_{args.dataset}_{args.architecture}_p{int(args.proj_dim)}.csv")
    scoring = kernel_scoring_id(
        args.dataset, args.architecture, args.weighting, args.metric,
        args.num_checkpoints, args.checkpoint_selection, args.checkpoint_weighting,
        args.scoring_transform, args.checkpoint_epochs, args.proj_dim, args.proj_seed)

    rows = collect_runs(
        args.scratch, args.dataset, args.architecture, args.fractions,
        args.seeds, args.methods, variant=args.variant)
    if not rows:
        logger.error(f"No runs found under {args.scratch} for "
                     f"{args.dataset}/{args.architecture}.")
        return
    logger.info(f"Found {len(rows)} saved runs.")
    attach_cached_d(rows, cache_path, scoring)

    by_seed = {}
    for row in rows:
        if row.get('indices_path') and row.get('d') is None:
            by_seed.setdefault(row['seed'], []).append(row)
    for seed, seed_rows in sorted(by_seed.items()):
        path = kernel_path(
            args.scratch, args.dataset, args.architecture, seed, args.proj_dim,
            args.variant)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"No kernel for seed {seed}: {path}. "
                f"Run scripts/analyze/precompute_kernel.py for this "
                f"dataset, architecture, and seed first.")
        score_seed_rows(seed_rows, path, args, seed, logger)
        write_table(cache_path, rows, scoring, args.dataset, args.architecture)
        logger.info(f"Wrote D cache -> {cache_path}")

    write_table(cache_path, rows, scoring, args.dataset, args.architecture)
    logger.info(f"Per-subset table -> {cache_path}")
    logger.info(f"LaTeX table -> {latex_path_for(cache_path)}")

    prefix = f"{args.dataset} / {args.architecture}"
    d_label = (f"Sketched full-network D (p={int(args.proj_dim)}, "
               f"all checkpoints)")
    kl_rows = [row for row in rows if is_kl_method(row['method'])]
    if kl_rows:
        kl_tex = latex_path_for(cache_path, '_kl')
        write_latex(kl_tex, kl_rows, args.dataset, args.architecture, kl_only=True)
        logger.info(f"KL-only LaTeX table -> {kl_tex}")
        plot_accuracy_vs_fraction(
            k_accuracy_series(with_accuracy(kl_rows)),
            os.path.join(out_dir, f"{args.dataset}_{args.architecture}_kl_accuracy_vs_fraction.pdf"),
            emphasize=emphasizes_k1)

    plot_accuracy_vs_fraction(
        comparison_accuracy_series(baseline_comparison_rows(with_accuracy(rows))),
        os.path.join(out_dir, f"{args.dataset}_{args.architecture}_accuracy_vs_fraction.pdf"),
        emphasize=KL_DISPLAY_NAME,
        exclude_from_limits=('el2n',))
    plot_error_vs_d(
        group_error_vs_d(rows),
        f"{prefix}: test error vs sketched D",
        os.path.join(out_dir,
                     f"{args.dataset}_{args.architecture}_error_vs_d_p{int(args.proj_dim)}.png"),
        xlabel=d_label)
    plot_vs_budget(
        group_by_budget(rows, 'selection_s'),
        f"{prefix}: selection time vs budget",
        'Budget (selected points)', 'Selection time (s)',
        os.path.join(out_dir, f"{args.dataset}_{args.architecture}_time_vs_budget.png"))


if __name__ == '__main__':
    precompute_main()
