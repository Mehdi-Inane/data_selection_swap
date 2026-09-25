"""
train_submodular_selection.py
GraphCut and Facility Location subset selection with apricot
(Schreiber, Bilmes & Noble, JMLR 2020;
https://apricot-select.readthedocs.io/en/stable/functions/graphCut.html).

Objectives (per class c, similarity S between samples of class c)
───────────────────────────────────────────────────────────────────
    Facility Location   f(X) = Σ_{v in V_c} max_{x in X} S(x, v)
    Graph Cut           f(X) = λ Σ_{v in V_c} Σ_{x in X} S(x, v)  -  Σ_{x, y in X} S(x, y)

Both are maximised with apricot's lazy greedy under |X| = quota_c.
Running per class keeps S at n_c × n_c (≈450² on CIFAR-100, ≈1300² on
ImageNet) instead of n_train², and matches the class-wise protocol of
DeepCore / CRAIG-style submodular baselines.

S = max(D) - D with D the Euclidean distance, which is apricot's own
default (metric='euclidean'). With --kernel grad / grad_trajectory, D is
the distance in the last-layer-gradient RKHS instead — the latter is the
kernel KL-Faithful optimises over.
"""

import json

import numpy as np
import torch
import apricot.functions.facilityLocation as _apricot_fl
import apricot.functions.graphCut as _apricot_gc
from apricot import FacilityLocationSelection, GraphCutSelection

import selection_common as common


# apricot eagerly re-JITs its numba kernels inside every selector's
# _initialize (~0.2 s GraphCut, ~0.45 s FacilityLocation per class — more
# than the greedy itself). Compile once per process so the timed
# 'selection' phase measures the algorithm, not the compiler.
def _compile_once(factory):
    cache = {}

    def wrapped(*args):
        key = repr(args)
        if key not in cache:
            cache[key] = factory(*args)
        return cache[key]
    return wrapped


for _mod, _names in ((_apricot_gc, ['calculate_gains_sieve']),
                     (_apricot_fl, ['calculate_gains', 'calculate_gains_sparse',
                                    'calculate_gains_sieve'])):
    for _name in _names:
        setattr(_mod, _name, _compile_once(getattr(_mod, _name)))


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm: per-class submodular maximisation
# ─────────────────────────────────────────────────────────────────────────────
def submodular_select(function, kernels, class_idx, quota, gc_lambda, optimizer, seed):
    selected_indices = []
    for c, idx in enumerate(class_idx):
        q = min(int(quota[c].item()), idx.numel())
        if q == 0:
            continue
        if q == idx.numel():
            selected_indices.extend(idx.tolist())
            continue

        S = common.kernel_to_similarity(kernels[c].double()).numpy()
        if function == 'graphcut':
            selector = GraphCutSelection(q, metric='precomputed', alpha=gc_lambda,
                                         optimizer=optimizer, random_state=seed)
        else:
            selector = FacilityLocationSelection(q, metric='precomputed',
                                                 optimizer=optimizer, random_state=seed)
        selector.fit(S)
        selected_indices.extend(idx[torch.from_numpy(np.asarray(selector.ranking))].tolist())

    return torch.tensor(selected_indices, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = common.base_argparser("GraphCut / Facility Location subset selection")
    p.add_argument('--function', required=True, choices=['graphcut', 'facility_location'])
    p.add_argument('--embedding_checkpoint', default=None, type=str,
                    help='Checkpoint to embed with. Default: last checkpoint '
                         'in --checkpoint_dir (the converged reference model).')
    p.add_argument('--kernel', default='features',
                   choices=['features', 'grad', 'grad_trajectory'])
    p.add_argument('--gc_lambda', default=1.0, type=float,
                   help='Graph-cut λ (apricot `alpha`, default 1).')
    p.add_argument('--optimizer', default='lazy', type=str,
                   help='apricot optimizer (lazy = exact greedy, accelerated).')
    args = p.parse_args()

    common.set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── 1. Dataset ───────────────────────────────────────────────────────────
    cfg = common.get_dataset_config(args.dataset, args.data_dir, download=args.download)
    NUM_CLASSES = cfg['num_classes']

    trainset, eval_loader, testloader, n_train = common.make_loaders(
        cfg, args.seed, args.batch_size, args.num_workers, args.scoring_transform)
    budget = int(args.fraction * n_train)

    method = args.function if args.kernel == 'features' else f'{args.function}_{args.kernel}'
    paths = common.get_save_paths(method, args.dataset, budget, args.seed)
    logger = common.get_logger(__name__, paths['log'])
    logger.info(f"Dataset={args.dataset}  budget={budget}  seed={args.seed}  "
                f"function={args.function}  kernel={args.kernel}  device={device}")

    timer = common.SelectionTimer(device)
    ref_model = common.build_model(cfg, NUM_CLASSES, device)

    # ── 2. Reference embedding → per-class kernels ───────────────────────────
    with timer.phase('scoring'):
        kernels, class_idx, ckpts = common.build_class_kernels(
            args.kernel, ref_model, eval_loader, args.checkpoint_dir,
            args.embedding_checkpoint, NUM_CLASSES, device)
    logger.info(f"Built per-class '{args.kernel}' kernels from {len(ckpts)} checkpoint(s).")

    # ── 3. Submodular selection ──────────────────────────────────────────────
    logger.info(f"Running {args.function} selection (budget={budget}/{n_train}) ...")
    targets = torch.empty(n_train, dtype=torch.long)
    for c, idx in enumerate(class_idx):
        targets[idx] = c
    with timer.phase('selection'):
        quota = common.per_class_budget(targets, NUM_CLASSES, budget)
        selected_indices = submodular_select(args.function, kernels, class_idx, quota,
                                             args.gc_lambda, args.optimizer, args.seed)
    logger.info(f"Selected {len(selected_indices)} samples across {NUM_CLASSES} classes.")

    common.save_timing(paths['timing'], timer, logger, method=method,
                       reference_epochs_used=common.reference_epochs_used(ckpts))
    with open(paths['scores'], 'w') as fh:
        json.dump({"method": method, "function": args.function, "kernel": args.kernel,
                    "gc_lambda": args.gc_lambda, "checkpoints": ckpts,
                    "n_selected": len(selected_indices)}, fh, indent=4)

    # ── 4. Retrain on selected subset (identical protocol to every baseline) ─
    common.finish(cfg, trainset, selected_indices, device, args, logger, paths)


if __name__ == "__main__":
    main()
