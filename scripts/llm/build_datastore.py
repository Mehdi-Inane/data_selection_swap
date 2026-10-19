#!/usr/bin/env python3
"""Build a gradient-feature store for smoke tests or from real LESS gradient dumps.

Two modes:

* default — a deterministic **synthetic** store, so the whole selection pipeline
  runs without GPU or dataset access (smoke tests).
* ``--from-less`` — ingest LESS Step-2 (train) and Step-3 (validation) projected
  gradient dumps into the same ``FeatureStore`` layout. It reads LESS's
  ``.../dim{D}/{all_orig|all_unormalized|grads-*}.pt`` outputs and reshapes the
  flattened validation gradients into subtasks, so nothing downstream changes.

Templates follow LESS ``matching.sh``; run with ``--help`` for the arguments.
"""

import argparse
import json

from data_selection_swap.llm import datastore


def _synthetic(args) -> datastore.FeatureStore:
    return datastore.make_synthetic_store(
        n=args.n, d=args.d, checkpoints=args.checkpoints, tasks=tuple(args.tasks),
        subtasks=args.subtasks, normalize=args.normalize, seed=args.seed,
        pool=not args.no_pool)


def _from_less(args) -> datastore.FeatureStore:
    n_subtasks = json.loads(args.n_subtasks) if args.n_subtasks else None
    pool_files = {}
    for pair in args.pool_files or []:
        name, _, path = pair.partition("=")
        pool_files[name] = path
    return datastore.load_from_less(
        dim=args.dim, ckpts=args.ckpts, checkpoint_weights=args.checkpoint_weights,
        train_file_names=args.train_file_names, train_grad_template=args.train_grad_template,
        tasks=args.tasks, val_grad_template=args.val_grad_template,
        n_subtasks=n_subtasks, pool_files=pool_files or None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--from-less", action="store_true",
                   help="ingest LESS gradient dumps instead of a synthetic store")

    # synthetic mode
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--d", type=int, default=8192)
    p.add_argument("--checkpoints", type=int, default=4)
    p.add_argument("--subtasks", type=int, default=2)
    p.add_argument("--normalize", choices=["l2", "none"], default="l2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-pool", action="store_true")

    # LESS ingest mode
    p.add_argument("--dim", type=int, default=8192)
    p.add_argument("--ckpts", type=int, nargs="+", default=None)
    p.add_argument("--checkpoint-weights", type=float, nargs="+", default=None)
    p.add_argument("--train-file-names", nargs="+", default=None)
    p.add_argument("--train-grad-template", default=None,
                   help="e.g. 'GRADS/{name}-ckpt{ckpt}-adam/dim{dim}'")
    p.add_argument("--val-grad-template", default=None,
                   help="e.g. 'GRADS/{task}-ckpt{ckpt}-sgd/dim{dim}'")
    p.add_argument("--tasks", nargs="+", default=["mmlu", "tydiqa", "bbh"])
    p.add_argument("--n-subtasks", default=None,
                   help="JSON map task->count (defaults to LESS mmlu:57 bbh:27 tydiqa:9)")
    p.add_argument("--pool-files", nargs="*", default=None, metavar="NAME=PATH",
                   help="original train JSONL per train-file name, for selected-data output")
    args = p.parse_args()

    if args.from_less:
        missing = [name for name, val in [
            ("--ckpts", args.ckpts), ("--checkpoint-weights", args.checkpoint_weights),
            ("--train-file-names", args.train_file_names),
            ("--train-grad-template", args.train_grad_template),
            ("--val-grad-template", args.val_grad_template)] if not val]
        if missing:
            p.error(f"--from-less requires: {', '.join(missing)}")
        store = _from_less(args)
    else:
        store = _synthetic(args)

    prov = store.save(args.out)
    print(f"store written to {args.out}: n={store.n}, checkpoints={len(store.weights)}, "
          f"tasks={sorted(store.validation)}, normalize={store.normalize}")
    print(f"provenance={prov}")


if __name__ == "__main__":
    main()
