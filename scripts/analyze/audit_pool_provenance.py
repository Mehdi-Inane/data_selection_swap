#!/usr/bin/env python3
"""Audit which selection pool each saved artifact was produced on.  Read-only.

Two pools exist in the wild:

  identity    Subset(full_train, range(n))                (current branch code)
  legacy(s)   random_split(full_train, [n, 0],
                           generator=Generator().manual_seed(s))
              i.e. a seed-dependent permutation of all n   (how the existing
              reference trajectories were trained)

They contain the same images but in a different order. Saved selector indices
are *pool positions*, so an index file only means something together with the
pool it was produced on. Scoring legacy-pool indices against the identity pool
(or vice versa) gives wrong D values with no error.

For every seed this script
  1. regenerates the legacy permutation and checks that it reproduces the
     pool_indices_sha256 recorded in the trajectory manifest;
  2. reads <prefix>_run.json of every KL run (these record the pool hash) and
     labels it identity / legacy / other;
  3. counts baseline runs (random, herding, gradmatch, ...), which record NO
     pool hash, and shows the mtime range so you can judge them against the
     date you switched code.

Usage:
  python scripts/analyze/audit_pool_provenance.py --scratch $SCRATCH \
      --dataset cifar10 --architecture resnet18 --seeds 42 43 44 45 46
"""

import argparse
import datetime
import glob
import json
import os
import re
from collections import Counter, defaultdict


def identity_pool(n):
    return list(range(n))


def legacy_pool(n, seed):
    """Exactly what random_split(ds, [n, 0], generator=manual_seed(seed)) used."""
    import torch
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()


def legacy_split_pool(n, n_val, seed):
    """Training half of the OLD 90/10 split: random_split(ds, [n-n_val, n_val], gen(seed))."""
    return legacy_pool(n, seed)[:n - n_val]


def fingerprint(indices):
    from data_selection_swap import common
    return common.ordered_indices_fingerprint(indices)


def classify(recorded, identity_hash, legacy_hash):
    if recorded is None:
        return 'unrecorded'
    if recorded == identity_hash:
        return 'identity'
    if recorded == legacy_hash:
        return 'legacy'
    return 'other'


def _mtime(path):
    return datetime.datetime.fromtimestamp(os.path.getmtime(path))


def audit_trajectories(ckpt_template, seeds, identity_hash, legacy_hashes):
    ok = True
    print("== Reference trajectories ==")
    for seed in seeds:
        path = os.path.join(ckpt_template.format(seed=seed), 'trajectory.json')
        if not os.path.exists(path):
            print(f"  seed {seed}: MISSING {path}")
            ok = False
            continue
        manifest = json.load(open(path))
        rec = manifest.get('pool_indices_sha256')
        label = classify(rec, identity_hash, legacy_hashes[seed])
        print(f"  seed {seed}: n_train={manifest.get('n_train')} "
              f"pool={label:<10} recorded={str(rec)[:12]}  "
              f"identity={identity_hash[:12]} legacy({seed})={legacy_hashes[seed][:12]}  "
              f"mtime={_mtime(path):%Y-%m-%d %H:%M}")
        if label != 'legacy':
            ok = False
    return ok


def scored_budgets(fractions, n_train):
    return {int(f * n_train) for f in fractions}


def audit_kl_runs(scratch, dataset, architecture, seeds, identity_hash, legacy_hashes,
                  old45k_hashes, budgets):
    root = os.path.join(scratch, 'gradmatch_swap', dataset, architecture)
    print(f"\n== KL runs (pool hash recorded in *_run.json) ==")
    print(f"   scored budgets (fraction x n_train): {sorted(budgets)}")
    by_budget = defaultdict(Counter)       # (budget) -> Counter(label)
    detail = defaultdict(Counter)          # (label, seed) -> Counter(trajectory dir)
    ntrain = defaultdict(Counter)          # label -> Counter(n_train)
    for seed in seeds:
        pattern = os.path.join(root, '*', f'seed_{seed}', '*_run.json')
        for path in glob.glob(pattern):
            budget = os.path.basename(os.path.dirname(os.path.dirname(path)))
            resolved = json.load(open(path)).get('config', {}).get('resolved', {})
            rec = resolved.get('pool_indices_sha256')
            if rec == old45k_hashes[seed]:
                label = 'legacy-45k(old 90/10 split)'
            else:
                label = classify(rec, identity_hash, legacy_hashes[seed])
            by_budget[budget][label] += 1
            ntrain[label][resolved.get('n_train')] += 1
            ckpts = resolved.get('checkpoints') or []
            if ckpts:
                detail[(label, seed)][os.path.dirname(ckpts[0].get('path', '?'))] += 1
    if not by_budget:
        print(f"  no *_run.json found under {root}")
        return
    print("  label counts per budget directory:")
    for budget in sorted(by_budget, key=lambda b: (not b.isdigit(), int(b) if b.isdigit() else 0)):
        scored = ' <- scored' if budget.isdigit() and int(budget) in budgets else ''
        print(f"    budget {budget:>6}: {dict(by_budget[budget])}{scored}")
    print("  n_train recorded, by label:")
    for label, c in ntrain.items():
        print(f"    {label}: {dict(c)}")
    print("  reference-trajectory directory each run actually used:")
    for (label, seed), c in sorted(detail.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        for d, n in c.items():
            print(f"    {label:<28} seed {seed}: {n:>3} runs  {d}")


def audit_baselines(scratch, dataset, architecture, seeds, budgets):
    root = os.path.join(scratch, 'data_selection_baselines')
    print("\n== Baseline runs at scored budgets (no pool hash recorded; judge by date) ==")
    pat = os.path.join(root, '*', dataset, architecture, '*', 'seed_*', '*_indices.pt')
    per_method = defaultdict(list)
    skipped = Counter()
    for path in glob.glob(pat):
        m = re.search(r'seed_(\d+)', path)
        if not m or int(m.group(1)) not in seeds:
            continue
        budget = os.path.basename(os.path.dirname(os.path.dirname(path)))
        method = os.path.relpath(path, root).split(os.sep)[0]
        if not (budget.isdigit() and int(budget) in budgets):
            skipped[budget] += 1
            continue
        per_method[method].append(_mtime(path))
    if not per_method:
        print(f"  none found at scored budgets under {root}")
    for method, times in sorted(per_method.items()):
        print(f"  {method:<28} n={len(times):<4} "
              f"{min(times):%Y-%m-%d %H:%M} .. {max(times):%Y-%m-%d %H:%M}")
    if skipped:
        print(f"  (ignored, not scored: budget dirs {dict(skipped)})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scratch', default=os.environ.get('SCRATCH'))
    p.add_argument('--dataset', required=True)
    p.add_argument('--architecture', required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    p.add_argument('--n_train', type=int, default=50000)
    p.add_argument('--fractions', type=float, nargs='+', default=[0.1, 0.2, 0.4, 0.5])
    p.add_argument('--checkpoint_template', default=None,
                   help="default: <scratch>/data_selection_swap/<dataset>/<arch>/"
                        "checkpoints/seed_{seed}")
    args = p.parse_args()

    template = args.checkpoint_template or os.path.join(
        args.scratch, 'data_selection_swap', args.dataset, args.architecture,
        'checkpoints', 'seed_{seed}')

    identity_hash = fingerprint(identity_pool(args.n_train))
    legacy_hashes = {s: fingerprint(legacy_pool(args.n_train, s)) for s in args.seeds}

    old45k = {s: fingerprint(legacy_split_pool(args.n_train, args.n_train // 10, s))
              for s in args.seeds}
    budgets = scored_budgets(args.fractions, args.n_train)

    traj_ok = audit_trajectories(template, args.seeds, identity_hash, legacy_hashes)
    audit_kl_runs(args.scratch, args.dataset, args.architecture, args.seeds,
                  identity_hash, legacy_hashes, old45k, budgets)
    audit_baselines(args.scratch, args.dataset, args.architecture, set(args.seeds),
                    budgets)

    print("\n== Reading the result ==")
    if not traj_ok:
        print("  Some trajectories are NOT reproduced by the legacy permutation "
              "(check the seeding convention in the code that trained them).")
    print("  KL runs labelled 'legacy' hold indices in the seed's permuted order; "
          "'identity' hold dataset indices. Mixed labels within a dataset/"
          "architecture mean subsets are in different coordinate systems.")


if __name__ == '__main__':
    main()