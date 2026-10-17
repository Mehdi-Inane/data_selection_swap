#!/usr/bin/env python3
"""Relabel legacy-pool reference trajectories to the identity pool.  Mutates
scratch manifests only with --apply; the default run is a dry verification.

Background
----------
The Oct-10 full-data baselines were trained through the pre-split-removal
code path (``random_split(full_train, [n, 0], generator=manual_seed(seed))``).
The pool was the whole training set, but its ``.indices`` were a seed-dependent
*permutation* of 0..n-1, so the trajectory manifest recorded
``pool_indices_sha256 = hash(perm)``.  Current code (``common.make_train_set``)
defines the pool as the *identity* order ``range(n)`` and every consumer
(``kl_full``, ``score_subsets``, ``comparison``, ``kernel``, ``transfer_d``)
refuses a trajectory whose recorded hash differs.

The two pools contain the same images; the permutation only relabels pool
positions, and the training loader shuffled each epoch anyway, so the recorded
weights are exactly as valid under the identity pool.  This script therefore
rewrites the fingerprint instead of retraining, but only after *proving* per
manifest that the recorded hash is reproduced by the legacy permutation for
that (n, seed) -- a manifest that matches neither pool order is left untouched
and reported.

For each verified manifest, with --apply:
  * back up the original once as ``trajectory.json.pre-pool-migration``;
  * set ``pool_indices_sha256`` (top level and inside ``immutable_config``)
    to ``hash(range(n))``;
  * recompute ``run_id = common.config_fingerprint(immutable_config)`` so the
    manifest stays internally consistent;
  * record a ``pool_provenance_migration`` block with the old values.

Checkpoint ``.pth`` files are never touched.

Usage:
  python scripts/analyze/migrate_pool_provenance.py --scratch $SCRATCH            # dry run
  python scripts/analyze/migrate_pool_provenance.py --scratch $SCRATCH --apply
  python scripts/analyze/migrate_pool_provenance.py --datasets cifar10 \
      --architectures resnet18 --seeds 42 43
"""

import argparse
import datetime
import json
import os
import shutil
import sys

_REPO_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src')
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

from data_selection_swap import common  # noqa: E402
from data_selection_swap.trajectory import MANIFEST_NAME  # noqa: E402

BACKUP_NAME = MANIFEST_NAME + '.pre-pool-migration'


def identity_hash(n):
    return common.ordered_indices_fingerprint(range(n))


def legacy_hash(n, seed):
    """Hash of the pool random_split(full, [n, 0], manual_seed(seed)) produced."""
    import torch
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return common.ordered_indices_fingerprint(perm)


def classify_manifest(manifest, n, seed):
    """Classify the recorded pool hash: 'identity', 'legacy', or 'other'."""
    recorded = manifest.get('pool_indices_sha256')
    if recorded == identity_hash(n):
        return 'identity'
    if recorded == legacy_hash(n, seed):
        return 'legacy'
    return 'other'


def _run_id_consistent(manifest):
    """True when immutable_config reproduces run_id (or there is nothing to check)."""
    immutable = manifest.get('immutable_config')
    recorded = manifest.get('run_id')
    if not isinstance(immutable, dict) or recorded is None:
        return True
    return common.config_fingerprint(immutable) == recorded


def migrate_manifest(path, *, apply=False):
    """Verify one trajectory manifest and, if it carries the legacy pool hash,
    relabel it to the identity pool.

    Returns (status, message) where status is one of:
      'verified-legacy'   -- would be / was migrated (apply decides which);
      'already-migrated'  -- already on the identity pool, nothing to do;
      'unverifiable'      -- hash matches neither pool, or the manifest is not
                             internally consistent; never touched.
    """
    with open(path) as fh:
        manifest = json.load(fh)

    n = manifest.get('n_train')
    seed = manifest.get('seed')
    if not isinstance(n, int) or seed is None:
        return 'unverifiable', 'manifest is missing n_train or seed'

    kind = classify_manifest(manifest, n, int(seed))
    if kind == 'identity':
        return 'already-migrated', (
            f"n={n} already on the identity pool hash {identity_hash(n)[:12]}")
    if kind == 'other':
        return 'unverifiable', (
            f"n={n} seed={seed} recorded={manifest.get('pool_indices_sha256')!r} "
            f"matches neither identity={identity_hash(n)[:12]} "
            f"nor legacy({seed})={legacy_hash(n, int(seed))[:12]}")
    if not _run_id_consistent(manifest):
        return 'unverifiable', (
            'run_id does not equal config_fingerprint(immutable_config); '
            'the manifest was edited after training -- refusing to migrate')

    # Sanity: the legacy permutation is a bijection over the whole pool, i.e.
    # relabeling to identity preserves membership exactly.
    import torch
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(int(seed))).tolist()
    if sorted(perm) != list(range(n)):
        return 'unverifiable', f'legacy permutation for seed {seed} is not a cover'

    if not apply:
        return 'verified-legacy', (
            f"n={n} seed={seed}: legacy hash {legacy_hash(n, int(seed))[:12]} "
            f"-> identity hash {identity_hash(n)[:12]} (dry run)")

    old_hash = manifest['pool_indices_sha256']
    old_run_id = manifest.get('run_id')

    backup = os.path.join(os.path.dirname(os.path.abspath(path)), BACKUP_NAME)
    if not os.path.exists(backup):
        shutil.copy2(path, backup)

    new_hash = identity_hash(n)
    manifest['pool_indices_sha256'] = new_hash
    immutable = manifest.get('immutable_config')
    if isinstance(immutable, dict) and 'pool_indices_sha256' in immutable:
        immutable['pool_indices_sha256'] = new_hash
        new_run_id = common.config_fingerprint(immutable)
        manifest['run_id'] = new_run_id
    else:
        new_run_id = old_run_id

    manifest['pool_provenance_migration'] = {
        'migrated_at': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        'script': os.path.relpath(os.path.abspath(__file__), _REPO_SRC),
        'old_pool_indices_sha256': old_hash,
        'old_run_id': old_run_id,
        'reason': (
            'Trajectory was trained on the full pool through '
            'random_split(full, [n, 0], manual_seed(seed)); its .indices were '
            'recorded as the seed permutation. Relabeled to the identity pool '
            'range(n): same membership, and the training loader shuffled each '
            'epoch, so the weights are pool-order independent.'),
    }
    common.atomic_json_dump(manifest, path)
    return 'verified-legacy', (
        f"n={n} seed={seed}: migrated {old_hash[:12]} -> {new_hash[:12]} "
        f"(run_id {old_run_id} -> {new_run_id}, backup {BACKUP_NAME})")


def seed_dirs(scratch, variant, datasets, architectures, seeds):
    root = os.path.join(scratch, 'data_selection_swap', variant) if variant \
        else os.path.join(scratch, 'data_selection_swap')
    for dataset in datasets:
        for architecture in architectures:
            for seed in seeds:
                yield (dataset, architecture, seed, os.path.join(
                    root, dataset, architecture, 'checkpoints', f'seed_{seed}'))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scratch', default=os.environ.get('SCRATCH', '/network/scratch/a/ahmedm'))
    p.add_argument('--variant', default='',
                   help="result-tree variant subfolder, e.g. 'adam'; '' is the sgd tree")
    p.add_argument('--datasets', nargs='+', default=['cifar10', 'cifar100'])
    p.add_argument('--architectures', nargs='+', default=['resnet18', 'vgg16', 'inception'])
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    p.add_argument('--apply', action='store_true',
                   help='rewrite manifests (default: verify only)')
    args = p.parse_args()

    mode = 'APPLY' if args.apply else 'DRY RUN'
    print(f"Pool-provenance migration [{mode}] under "
          f"{os.path.join(args.scratch, 'data_selection_swap', args.variant) if args.variant else os.path.join(args.scratch, 'data_selection_swap')}")
    counts = {'verified-legacy': 0, 'already-migrated': 0,
              'unverifiable': 0, 'missing': 0}
    problems = []
    for dataset, architecture, seed, directory in seed_dirs(
            args.scratch, args.variant, args.datasets, args.architectures, args.seeds):
        path = os.path.join(directory, MANIFEST_NAME)
        tag = f"{dataset}/{architecture}/seed_{seed}"
        if not os.path.isfile(path):
            counts['missing'] += 1
            problems.append(f"  MISSING  {tag}: no {MANIFEST_NAME} in {directory}")
            print(f"  MISSING       {tag}")
            continue
        status, message = migrate_manifest(path, apply=args.apply)
        counts[status] += 1
        if status == 'unverifiable':
            problems.append(f"  REFUSED  {tag}: {message}")
        print(f"  {status:<17} {tag}: {message}")

    print(f"\nSummary: {counts}")
    if problems:
        print("\nDirectories needing attention:")
        print('\n'.join(problems))
        print("Unverifiable/missing manifests were left untouched. Narrow the "
              "--datasets/--architectures/--seeds or investigate before --apply.")
        return 1
    if not args.apply and counts['verified-legacy']:
        print(f"\n{counts['verified-legacy']} manifests verified; re-run with --apply.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
