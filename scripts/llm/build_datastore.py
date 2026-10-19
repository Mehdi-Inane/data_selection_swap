#!/usr/bin/env python3
"""Build a gradient-feature store for smoke tests and (Phase 0) real LESS runs.

This stub materializes a deterministic *synthetic* store so the whole selection
pipeline can be exercised without GPU or dataset access. In Phase 0 this same
command is extended to ingest LESS's Step-2/Step-3 gradient dumps into the
identical ``FeatureStore`` layout (``features/<ckpt>.pt`` and
``val/<task>/<subtask>/<ckpt>.pt`` plus a ``manifest.json``); nothing downstream
changes because everything consumes the store, not LESS directly.
"""

import argparse

from data_selection_swap.llm import datastore


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--d", type=int, default=8192)
    p.add_argument("--checkpoints", type=int, default=4)
    p.add_argument("--tasks", nargs="+", default=["mmlu", "tydiqa", "bbh"])
    p.add_argument("--subtasks", type=int, default=2)
    p.add_argument("--normalize", choices=["l2", "none"], default="l2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-pool", action="store_true")
    args = p.parse_args()

    store = datastore.make_synthetic_store(
        n=args.n, d=args.d, checkpoints=args.checkpoints, tasks=tuple(args.tasks),
        subtasks=args.subtasks, normalize=args.normalize, seed=args.seed,
        pool=not args.no_pool)
    prov = store.save(args.out)
    print(f"wrote synthetic store to {args.out} (n={args.n}, d={args.d}, "
          f"checkpoints={args.checkpoints}, tasks={args.tasks})")
    print(f"provenance={prov}")


if __name__ == "__main__":
    main()
