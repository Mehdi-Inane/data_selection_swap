#!/usr/bin/env python3
"""Do different architectures — and different methods — select the same examples?

Reads the saved ``*_indices.pt`` and answers that with one number per pair of
subsets: the chance-corrected agreement

    kappa = (o - E[o]) / (max possible o - E[o]),    E[o] = m_A m_B / n

with ``o = |A ∩ B|``. For two subsets of the same size m at fraction f = m/n
this is ``(o/m - f) / (1 - f)``: 0 means the two selections are no more alike
than independent draws, 1 means identical. Raw overlap is reported alongside
it but must not be compared across budgets — two independent draws at f = 0.5
already share half their examples. Jaccard is a monotone function of the same
o, so it ranks pairs identically and is carried only as a familiar number.

Why the pool-relative indices can be intersected directly
─────────────────────────────────────────────────────────
``common.make_loaders`` draws the train/val split from a generator seeded only
by ``seed``; the architecture never enters. So at a fixed seed, index i means
the same image for every architecture and every method, and the saved index
tensors are directly comparable. Across *seeds* both the pool ordering and the
pool membership change, so that contrast maps back to absolute dataset indices
(``pool_indices``) and conditions on the two pools' intersection.

Three contrasts, all on the same kappa scale
────────────────────────────────────────────
  architecture : same seed, same method, different architecture
  method       : same seed, same architecture, different method
  seed         : same architecture, same method, different seed

The third is the one that makes the first readable. It measures how
reproducible selection is at all, so it is the ceiling any cross-architecture
kappa should be judged against: if the two are comparable the architecture is
doing no more than reseeding, and if the architecture kappa is far below it
the architecture genuinely decides which examples are picked.

No significance test is reported, on purpose. Overlap under independence is
Hypergeometric with a standard deviation of ~19 examples at n = 50000,
m = 5000, so every pair clears any p-value threshold by a wide margin while
saying nothing about effect size. The hypergeometric sd is written to the CSV
as the measurement-noise scale instead.

Class composition versus individual examples
────────────────────────────────────────────
Several selectors are class-balanced by construction (``common.per_class_topk``)
and the rest still concentrate on hard classes, so agreement is split in two
when ``--data_dir`` supplies the labels: ``kappa_stratified`` uses the
within-class null ``E[o] = sum_c m_c^A m_c^B / n_c``, and ``class_tv`` is the
total-variation distance between the two class histograms. Small class_tv with
near-zero kappa_stratified says the architectures agree on which classes are
hard and disagree on which examples.

Usage:
    python scripts/analyze/selection_overlap.py --dataset cifar10
    python scripts/analyze/selection_overlap.py --dataset cifar10 --fractions 0.1 0.2 \\
        --data_dir /path/to/cifar10_data
    python scripts/analyze/selection_overlap.py --dataset cifar10 \\
        --scored_input $SCRATCH/data_selection_baselines
"""

import argparse
import csv
import logging
import os
import re
import sys

import numpy as np
import torch

from ..paths import tree
from .comparison import N_TRAIN, load_json, run_dir
from .d_accuracy_scatter import expand_inputs
from .result_plots import is_kl_method

# Size of the training set. The selection pool is the whole training set, so
# this must equal N_TRAIN; the two are cross-checked in pool_indices so a wrong
# entry here cannot silently shift every mapping.
N_TOTAL = {
    'cifar10': 50000,
    'cifar100': 50000,
    'mnist': 60000,
    'imagenet': 1281167,
}

ARCHITECTURE_CONTRAST = 'architecture'
METHOD_CONTRAST = 'method'
SEED_CONTRAST = 'seed'

PAIR_FIELDS = [
    'contrast', 'dataset', 'fraction', 'budget', 'n_pool',
    'architecture_a', 'method_a', 'k_a', 'seed_a', 'size_a',
    'architecture_b', 'method_b', 'k_b', 'seed_b', 'size_b',
    'overlap', 'overlap_frac', 'jaccard', 'kappa',
    'kappa_stratified', 'class_tv', 'chance_overlap', 'hypergeom_sd',
    'k_mismatch',
]

CONSENSUS_FIELDS = [
    'dataset', 'architecture_group', 'fraction', 'budget', 'method', 'k', 'seed',
    'n_pool', 'n_groups', 'selected_by', 'count', 'expected_count', 'ratio',
    'union_size', 'union_over_budget', 'expected_union_over_budget',
]

CLOSEST_FIELDS = [
    'dataset', 'architecture', 'fraction', 'reference', 'method', 'n_pairs',
    'kappa_mean', 'kappa_std', 'seed_floor', 'kappa_over_floor',
    'overlap_frac_mean', 'class_tv_mean',
    'd_rel_reference', 'd_rel_method', 'd_rel_gap',
    'acc_reference', 'acc_method', 'acc_gap',
]


# ─────────────────────────────────────────────────────────────────────────────
# Pool mapping
# ─────────────────────────────────────────────────────────────────────────────
def pool_indices(dataset, seed):
    """Absolute dataset indices behind pool-relative position 0..n_train-1.

    The pool is the full training set in its natural order
    (``common.make_train_set``), so position i is dataset index i whatever the
    seed. ``seed`` is kept so call sites need no change.
    """
    if dataset not in N_TOTAL:
        raise KeyError(f"No pool size recorded for dataset {dataset!r}; "
                       f"known: {sorted(N_TOTAL)}")
    n_total = N_TOTAL[dataset]
    n_train = n_total
    if n_train != N_TRAIN[dataset]:
        raise AssertionError(
            f"N_TOTAL[{dataset!r}] = {n_total} implies a pool of {n_train}, but "
            f"analysis.comparison.N_TRAIN says {N_TRAIN[dataset]}. One of the two "
            f"is wrong, and every cross-seed index mapping depends on it.")
    return np.arange(n_total)


def pool_labels(dataset, data_dir, seed):
    """Class label of every pool position, in pool order.

    Needs the dataset only for its ``targets`` array: no transforms, no images
    decoded, no GPU. torchvision is imported lazily so the default run of this
    module does not pay for it.
    """
    import torchvision.datasets as datasets

    builders = {
        'cifar10': datasets.CIFAR10,
        'cifar100': datasets.CIFAR100,
        'mnist': datasets.MNIST,
    }
    if dataset not in builders:
        raise ValueError(
            f"Class-stratified metrics are not available for {dataset!r} "
            f"(supported: {sorted(builders)}). Drop --data_dir to skip them.")
    full = builders[dataset](root=data_dir, train=True, download=False)
    targets = np.asarray(full.targets, dtype=np.int64)
    if targets.size != N_TOTAL[dataset]:
        raise ValueError(
            f"{dataset} at {data_dir!r} has {targets.size} training examples, "
            f"expected {N_TOTAL[dataset]}.")
    return targets[pool_indices(dataset, seed)]


# ─────────────────────────────────────────────────────────────────────────────
# Discovery
#   Keyed on *_indices.pt rather than *_metrics.json. analysis.comparison keys
#   on the metrics file because it reports accuracy, which makes a
#   --selection_only run invisible to it. Such a run has exactly what this
#   analysis needs, so it is discovered here.
# ─────────────────────────────────────────────────────────────────────────────
_KL_INDICES = re.compile(
    r'^(kl_faithful(?:_[a-zA-Z0-9]+)*)_\d+_K\d+(?:_cfg[0-9a-f]+)?_indices\.pt$')


def _indices_pattern(method, budget):
    return re.compile(
        rf'^{re.escape(method)}_{budget}(?:_K(?P<k>\d+))?'
        rf'(?:_cfg(?P<run_id>[0-9a-f]+))?_indices\.pt$')


def discover_methods(scratch, dataset, architecture, variant=''):
    """Every KL variant and baseline that saved indices for this architecture."""
    kl_methods = set()
    kl_root = os.path.join(
        tree(scratch, 'gradmatch_swap', variant), dataset, architecture)
    if os.path.isdir(kl_root):
        for _, _, files in os.walk(kl_root):
            for fname in files:
                match = _KL_INDICES.match(fname)
                if match:
                    kl_methods.add(match.group(1))

    root = tree(scratch, 'data_selection_baselines', variant)
    baselines = sorted(
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name, dataset, architecture))
    ) if os.path.isdir(root) else []
    ordered_kl = sorted(kl_methods, key=lambda name: (name != 'kl_faithful', name))
    return ordered_kl + baselines


def _run_description(indices_path, filename_run_id):
    """Human-readable identity for a discovered subset, for conflict reports."""
    prefix = indices_path[:-len('_indices.pt')]
    manifest_path = f'{prefix}_run.json'
    manifest = load_json(manifest_path)
    manifest_run_id = manifest.get('run_id') if isinstance(manifest, dict) else None
    run_id = manifest_run_id or filename_run_id or 'legacy/no-run-id'
    fields = [f"run_id={run_id!r}", f"indices={indices_path!r}"]
    if isinstance(manifest, dict):
        config = manifest.get('config')
        if isinstance(config, dict) and config.get('selector') is not None:
            fields.append(f"selector={config['selector']!r}")
    fields.append(
        f"manifest={manifest_path!r}" if os.path.exists(manifest_path)
        else "manifest=<missing>")
    return "{" + ", ".join(fields) + "}"


def discover_subsets(directory, method, budget):
    """``(K, indices_path)`` for every run of this method/budget in ``directory``.

    K is None for the baselines, which do not tag a checkpoint count. As in
    ``analysis.comparison.discover_runs``, two configuration fingerprints in one
    (method, budget, K, seed) cell are refused rather than treated as replicate
    seeds — here they would otherwise contribute two near-duplicate points to
    every pairwise agreement.
    """
    if not os.path.isdir(directory):
        return []
    pattern = _indices_pattern(method, budget)
    found = []
    for fname in sorted(os.listdir(directory)):
        match = pattern.match(fname)
        if not match:
            continue
        found.append({
            'k': int(match.group('k')) if match.group('k') is not None else None,
            'run_id': match.group('run_id'),
            'indices': os.path.join(directory, fname),
        })

    by_k = {}
    for candidate in found:
        by_k.setdefault(candidate['k'], []).append(candidate)
    conflicts = {k: values for k, values in by_k.items() if len(values) > 1}
    if conflicts:
        details = []
        for k, values in sorted(conflicts.items(), key=lambda item: str(item[0])):
            runs = "; ".join(
                _run_description(value['indices'], value['run_id']) for value in values)
            details.append(f"K={k!r}: {runs}")
        raise RuntimeError(
            "Multiple selector configurations occupy the same cell "
            f"(directory={directory!r}, method={method!r}, budget={budget}). Refusing to "
            "treat configuration reruns as replicate seeds. Keep exactly one run per "
            "K or compare explicit run_id groups separately. Conflicts: "
            + " | ".join(details))

    return [(candidate['k'], candidate['indices']) for candidate in found]


def load_indices(path):
    """Saved pool-relative indices as a sorted, deduplicated int64 array.

    Deliberately not ``common.torch_load``: importing ``common`` drags in
    torchvision and the training stack to read one index tensor, which would
    cost this CPU-only analysis its light dependency footprint.
    """
    try:
        raw = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError as exc:
        # Older torch rejects the kwarg, or accepts it and fails in Unpickler.
        if 'weights_only' not in str(exc):
            raise
        raw = torch.load(path, map_location='cpu')
    values = torch.as_tensor(raw, dtype=torch.long).reshape(-1).numpy()
    unique = np.unique(values)
    if unique.size != values.size:
        raise ValueError(f"{path} contains {values.size - unique.size} duplicate indices; "
                         f"a subset must be a set.")
    return unique


def collect_subsets(scratch, dataset, architectures, fractions, seeds,
                    methods=None, checkpoints=None, logger=None, variant=''):
    """One record per saved subset, with its indices already loaded."""
    logger = logger or logging.getLogger('selection_overlap')
    records = []
    for architecture in architectures:
        names = methods or discover_methods(
            scratch, dataset, architecture, variant=variant)
        if not names:
            logger.warning(f"No saved subsets found for architecture {architecture}.")
            continue
        for fraction in fractions:
            budget = int(fraction * N_TRAIN[dataset])
            for method in names:
                for seed in seeds:
                    directory = run_dir(scratch, method, dataset, architecture,
                                        budget, seed, variant=variant)
                    for k, path in discover_subsets(directory, method, budget):
                        if checkpoints is not None and k is not None and k not in checkpoints:
                            continue
                        records.append(dict(
                            dataset=dataset, architecture=architecture,
                            fraction=fraction, budget=budget, method=method,
                            k=k, seed=seed, indices_path=path,
                            indices=load_indices(path),
                        ))
    return records


def run_label(record):
    """Method name, carrying the checkpoint count when the method tags one."""
    return record['method'] if record['k'] is None else f"{record['method']}_K{record['k']}"


# ─────────────────────────────────────────────────────────────────────────────
# Agreement metrics
# ─────────────────────────────────────────────────────────────────────────────
def kappa(size_a, size_b, overlap, n):
    """Overlap rescaled so 0 is chance and 1 is as equal as the sizes allow.

    ``min(size_a, size_b)`` is the largest overlap two sets of these sizes can
    reach, so unequal budgets (class quotas sometimes fall a few examples short)
    still land on a comparable scale.
    """
    if n <= 0 or size_a <= 0 or size_b <= 0:
        return float('nan')
    expected = size_a * size_b / n
    ceiling = min(size_a, size_b)
    if ceiling <= expected:
        return float('nan')
    return (overlap - expected) / (ceiling - expected)


def jaccard(size_a, size_b, overlap):
    union = size_a + size_b - overlap
    return overlap / union if union else float('nan')


def hypergeometric_sd(size_a, size_b, n):
    """Standard deviation of the overlap of two independent uniform subsets."""
    if n < 2 or size_a <= 0 or size_b <= 0:
        return float('nan')
    p = size_b / n
    return float(np.sqrt(size_a * p * (1.0 - p) * (n - size_a) / (n - 1)))


def class_counts(indices, labels, num_classes):
    return np.bincount(labels[indices], minlength=num_classes).astype(np.float64)


def stratified_kappa(indices_a, indices_b, overlap, labels, num_classes):
    """Agreement measured against a null that already matches class composition.

    The unstratified kappa cannot tell "these two picked the same hard classes"
    apart from "these two picked the same hard examples". This holds each
    method's class histogram fixed, so what is left is agreement within classes.
    """
    pool = np.bincount(labels, minlength=num_classes).astype(np.float64)
    count_a = class_counts(indices_a, labels, num_classes)
    count_b = class_counts(indices_b, labels, num_classes)
    present = pool > 0
    expected = float(np.sum(count_a[present] * count_b[present] / pool[present]))
    ceiling = float(np.sum(np.minimum(count_a, count_b)))
    if ceiling <= expected:
        return float('nan')
    return (overlap - expected) / (ceiling - expected)


def class_histogram_tv(indices_a, indices_b, labels, num_classes):
    """Total-variation distance between the two selections' class proportions."""
    count_a = class_counts(indices_a, labels, num_classes)
    count_b = class_counts(indices_b, labels, num_classes)
    if count_a.sum() == 0 or count_b.sum() == 0:
        return float('nan')
    return float(0.5 * np.abs(count_a / count_a.sum() - count_b / count_b.sum()).sum())


def pair_metrics(indices_a, indices_b, n, labels=None, num_classes=None):
    """Every agreement number for one pair of subsets over a shared pool of size n."""
    size_a, size_b = int(indices_a.size), int(indices_b.size)
    overlap = int(np.intersect1d(indices_a, indices_b, assume_unique=True).size)
    row = dict(
        size_a=size_a, size_b=size_b, overlap=overlap,
        overlap_frac=overlap / min(size_a, size_b) if min(size_a, size_b) else float('nan'),
        jaccard=jaccard(size_a, size_b, overlap),
        kappa=kappa(size_a, size_b, overlap, n),
        chance_overlap=size_a * size_b / n if n else float('nan'),
        hypergeom_sd=hypergeometric_sd(size_a, size_b, n),
        kappa_stratified=float('nan'), class_tv=float('nan'),
    )
    if labels is not None:
        row['kappa_stratified'] = stratified_kappa(
            indices_a, indices_b, overlap, labels, num_classes)
        row['class_tv'] = class_histogram_tv(indices_a, indices_b, labels, num_classes)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Contrasts
# ─────────────────────────────────────────────────────────────────────────────
def _pair_row(contrast, record_a, record_b, metrics, n_pool, k_mismatch=False):
    row = dict(
        contrast=contrast,
        dataset=record_a['dataset'],
        fraction=record_a['fraction'],
        budget=record_a['budget'],
        n_pool=n_pool,
        architecture_a=record_a['architecture'], method_a=record_a['method'],
        k_a='' if record_a['k'] is None else record_a['k'], seed_a=record_a['seed'],
        architecture_b=record_b['architecture'], method_b=record_b['method'],
        k_b='' if record_b['k'] is None else record_b['k'], seed_b=record_b['seed'],
        k_mismatch=int(bool(k_mismatch)),
    )
    row.update(metrics)
    return row


def _match_by_method(left, right, logger):
    """Pair runs of the same method across two groups.

    Exact (method, K) matches first. A method whose K differs but which has
    exactly one run on each side is still paired — the KL runs tag the number of
    checkpoints actually available, which can differ between two architectures'
    trajectories — and the pair is flagged so the CSV keeps the distinction.
    """
    pairs = []
    by_key_left = {}
    for record in left:
        by_key_left.setdefault((record['method'], record['k']), []).append(record)
    by_key_right = {}
    for record in right:
        by_key_right.setdefault((record['method'], record['k']), []).append(record)

    matched_left, matched_right = set(), set()
    for key, records_a in by_key_left.items():
        for record_a in records_a:
            for record_b in by_key_right.get(key, []):
                pairs.append((record_a, record_b, False))
                matched_left.add(id(record_a))
                matched_right.add(id(record_b))

    leftover_left, leftover_right = {}, {}
    for record in left:
        if id(record) not in matched_left:
            leftover_left.setdefault(record['method'], []).append(record)
    for record in right:
        if id(record) not in matched_right:
            leftover_right.setdefault(record['method'], []).append(record)
    for method, records_a in leftover_left.items():
        records_b = leftover_right.get(method, [])
        if len(records_a) == 1 and len(records_b) == 1:
            logger.debug(
                f"Pairing {method} across K={records_a[0]['k']} and K={records_b[0]['k']}: "
                f"one run on each side.")
            pairs.append((records_a[0], records_b[0], True))
    return pairs


def architecture_contrast(records, labels_by_seed=None, num_classes=None, logger=None):
    """Same seed, same method, different architecture."""
    logger = logger or logging.getLogger('selection_overlap')
    groups = {}
    for record in records:
        groups.setdefault((record['fraction'], record['seed']), []).append(record)

    rows = []
    for (fraction, seed), group in sorted(groups.items()):
        n_pool = N_TRAIN[group[0]['dataset']]
        labels = (labels_by_seed or {}).get(seed)
        by_architecture = {}
        for record in group:
            by_architecture.setdefault(record['architecture'], []).append(record)
        names = sorted(by_architecture)
        for i, arch_a in enumerate(names):
            for arch_b in names[i + 1:]:
                for record_a, record_b, mismatch in _match_by_method(
                        by_architecture[arch_a], by_architecture[arch_b], logger):
                    metrics = pair_metrics(record_a['indices'], record_b['indices'],
                                           n_pool, labels, num_classes)
                    rows.append(_pair_row(ARCHITECTURE_CONTRAST, record_a, record_b,
                                          metrics, n_pool, mismatch))
    return rows


def method_contrast(records, labels_by_seed=None, num_classes=None):
    """Same seed, same architecture, different method."""
    groups = {}
    for record in records:
        key = (record['fraction'], record['seed'], record['architecture'])
        groups.setdefault(key, []).append(record)

    rows = []
    for (fraction, seed, architecture), group in sorted(groups.items()):
        n_pool = N_TRAIN[group[0]['dataset']]
        labels = (labels_by_seed or {}).get(seed)
        ordered = sorted(group, key=run_label)
        for i, record_a in enumerate(ordered):
            for record_b in ordered[i + 1:]:
                metrics = pair_metrics(record_a['indices'], record_b['indices'],
                                       n_pool, labels, num_classes)
                rows.append(_pair_row(METHOD_CONTRAST, record_a, record_b,
                                      metrics, n_pool))
    return rows


def seed_contrast(records, logger=None):
    """Same architecture, same method, different seed — the reproducibility ceiling.

    All seeds share one pool (the full training set), so selections are
    directly comparable. They are still mapped back to absolute dataset indices
    and restricted to the shared part of the two pools, which is now the whole
    pool.
    """
    logger = logger or logging.getLogger('selection_overlap')
    pools = {}

    def pool_for(dataset, seed):
        if (dataset, seed) not in pools:
            pools[(dataset, seed)] = pool_indices(dataset, seed)
        return pools[(dataset, seed)]

    groups = {}
    for record in records:
        key = (record['fraction'], record['architecture'])
        groups.setdefault(key, []).append(record)

    rows = []
    for (fraction, architecture), group in sorted(groups.items()):
        by_seed = {}
        for record in group:
            by_seed.setdefault(record['seed'], []).append(record)
        seeds = sorted(by_seed)
        for i, seed_a in enumerate(seeds):
            for seed_b in seeds[i + 1:]:
                for record_a, record_b, mismatch in _match_by_method(
                        by_seed[seed_a], by_seed[seed_b], logger):
                    dataset = record_a['dataset']
                    pool_a = pool_for(dataset, seed_a)
                    pool_b = pool_for(dataset, seed_b)
                    shared = np.intersect1d(pool_a, pool_b)
                    absolute_a = np.intersect1d(pool_a[record_a['indices']], shared)
                    absolute_b = np.intersect1d(pool_b[record_b['indices']], shared)
                    metrics = pair_metrics(absolute_a, absolute_b, int(shared.size))
                    rows.append(_pair_row(SEED_CONTRAST, record_a, record_b,
                                          metrics, int(shared.size), mismatch))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Consensus across architectures
# ─────────────────────────────────────────────────────────────────────────────
def poisson_binomial(probabilities):
    """Distribution of the number of successes among independent, unequal trials.

    Exact, by convolution: the groups here number three or so, and the equal-p
    binomial shortcut would hide the small budget differences class quotas
    introduce.
    """
    distribution = np.zeros(len(probabilities) + 1, dtype=np.float64)
    distribution[0] = 1.0
    for j, p in enumerate(probabilities):
        distribution[1:j + 2] = distribution[1:j + 2] * (1 - p) + distribution[:j + 1] * p
        distribution[0] *= (1 - p)
    return distribution


def consensus_rows(records):
    """How many architectures picked each example, against the independent null."""
    groups = {}
    for record in records:
        key = (record['fraction'], record['seed'], record['method'], record['k'])
        groups.setdefault(key, []).append(record)

    rows = []
    for (fraction, seed, method, k), group in sorted(
            groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        if len({record['architecture'] for record in group}) < 2:
            continue
        group = sorted(group, key=lambda record: record['architecture'])
        dataset = group[0]['dataset']
        n_pool = N_TRAIN[dataset]
        multiplicity = np.zeros(n_pool, dtype=np.int32)
        for record in group:
            multiplicity[record['indices']] += 1
        observed = np.bincount(multiplicity, minlength=len(group) + 1)

        expected = n_pool * poisson_binomial(
            [record['indices'].size / n_pool for record in group])
        union = int(np.count_nonzero(multiplicity))
        budget = group[0]['budget']
        architectures = '+'.join(record['architecture'] for record in group)
        for selected_by in range(len(group) + 1):
            rows.append(dict(
                dataset=dataset, architecture_group=architectures, fraction=fraction,
                budget=budget, method=method, k='' if k is None else k, seed=seed,
                n_pool=n_pool, n_groups=len(group), selected_by=selected_by,
                count=int(observed[selected_by]),
                expected_count=float(expected[selected_by]),
                ratio=(float(observed[selected_by] / expected[selected_by])
                       if expected[selected_by] > 0 else float('nan')),
                union_size=union,
                union_over_budget=union / budget if budget else float('nan'),
                expected_union_over_budget=(
                    float(n_pool * (1.0 - expected[0] / n_pool) / budget)
                    if budget else float('nan')),
            ))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────
def mean_std(values):
    finite = [value for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return float('nan'), float('nan'), 0
    return (float(np.mean(finite)),
            float(np.std(finite, ddof=1 if len(finite) > 1 else 0)),
            len(finite))


def seed_floors(rows):
    """``(fraction, architecture, method) -> (mean seed-to-seed kappa, its sem)``."""
    buckets = {}
    for row in rows:
        if row['contrast'] != SEED_CONTRAST:
            continue
        key = (row['fraction'], row['architecture_a'], row['method_a'])
        buckets.setdefault(key, []).append(row['kappa'])
    floors = {}
    for key, values in buckets.items():
        mean, std, count = mean_std(values)
        floors[key] = (mean, std / np.sqrt(count) if count else float('nan'))
    return floors


def floor_ratio(value, floor):
    """``kappa / floor``, but only where the floor is itself above chance.

    A method whose two seeds already disagree has a floor sitting on zero, and
    dividing by it manufactures ratios in the tens that say nothing. The ratio
    is reported only when the floor clears two standard errors above chance;
    otherwise the raw kappa and the raw floor are left to speak for themselves.
    """
    if floor is None:
        return float('nan')
    mean, sem = floor
    if not np.isfinite(mean) or not np.isfinite(value):
        return float('nan')
    if mean <= 2.0 * (sem if np.isfinite(sem) else 0.0) or mean <= 0.0:
        return float('nan')
    return value / mean


def architecture_summary(rows, floors):
    """Per (fraction, method, architecture pair): kappa against the seed floor."""
    buckets = {}
    for row in rows:
        if row['contrast'] != ARCHITECTURE_CONTRAST:
            continue
        key = (row['fraction'], row['method_a'],
               row['architecture_a'], row['architecture_b'])
        buckets.setdefault(key, []).append(row)

    summary = []
    for (fraction, method, arch_a, arch_b), group in sorted(buckets.items()):
        mean, std, count = mean_std([row['kappa'] for row in group])
        # Both architectures' own reproducibility, pooled; the wider of the two
        # standard errors keeps the ratio's gate on the conservative side.
        pairs = [floors[(fraction, arch, method)] for arch in (arch_a, arch_b)
                 if (fraction, arch, method) in floors]
        floor = (mean_std([pair[0] for pair in pairs])[0],
                 max([pair[1] for pair in pairs], default=float('nan'))) if pairs else None
        summary.append(dict(
            fraction=fraction, method=method, architecture_a=arch_a,
            architecture_b=arch_b, n=count, kappa=mean, kappa_std=std,
            seed_floor=floor[0] if floor else float('nan'),
            kappa_over_floor=floor_ratio(mean, floor),
            class_tv=mean_std([row['class_tv'] for row in group])[0],
            kappa_stratified=mean_std([row['kappa_stratified'] for row in group])[0],
        ))
    return summary


def kappa_matrix(rows, fraction, architecture):
    """Symmetric method-by-method mean kappa for one (fraction, architecture)."""
    buckets = {}
    for row in rows:
        if (row['contrast'] != METHOD_CONTRAST or row['fraction'] != fraction
                or row['architecture_a'] != architecture):
            continue
        label_a = row['method_a'] if row['k_a'] == '' else f"{row['method_a']}_K{row['k_a']}"
        label_b = row['method_b'] if row['k_b'] == '' else f"{row['method_b']}_K{row['k_b']}"
        buckets.setdefault((label_a, label_b), []).append(row['kappa'])
    if not buckets:
        return [], np.zeros((0, 0))

    labels = sorted({label for pair in buckets for label in pair},
                    key=lambda name: (not is_kl_method(name), name))
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.full((len(labels), len(labels)), np.nan)
    np.fill_diagonal(matrix, 1.0)
    for (label_a, label_b), values in buckets.items():
        value = mean_std(values)[0]
        matrix[index[label_a], index[label_b]] = value
        matrix[index[label_b], index[label_a]] = value
    return labels, matrix


# ─────────────────────────────────────────────────────────────────────────────
# Closest method: set agreement joined to objective and accuracy
# ─────────────────────────────────────────────────────────────────────────────
def _float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def read_scored_tables(paths):
    """``(dataset, architecture, fraction, method, seed) -> (d_rel, acc)``.

    Reads the per-subset tables ``analysis.score_subsets`` writes. Only the two
    columns needed here are taken; rows without a usable D_rel are skipped.
    """
    scored = {}
    for path in expand_inputs(paths):
        with open(path, newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            if not {'d_rel', 'method', 'seed'} <= fields:
                raise ValueError(
                    f"{path} is not a per-subset table from "
                    f"scripts/analyze/score_subsets.py: it has no 'd_rel' column.")
            for record in reader:
                d_rel = _float(record.get('d_rel'))
                fraction = _float(record.get('fraction'))
                if d_rel is None or fraction is None:
                    continue
                key = (record.get('dataset', ''), record.get('architecture', ''),
                       fraction, record.get('method', ''), str(record.get('seed', '')))
                scored[key] = (d_rel, _float(record.get('acc')))
    return scored


def closest_methods(rows, floors, references=None, scored=None):
    """Rank every method by how close its subsets are to the KL reference's.

    Three senses of close, side by side, because they can disagree: the same
    examples (kappa), the same faithfulness (D_rel), and the same outcome
    (accuracy). A method far away in set space but matching on D_rel would say
    the objective admits many near-optimal subsets rather than one.
    """
    scored = scored or {}
    buckets = {}
    for row in rows:
        if row['contrast'] != METHOD_CONTRAST:
            continue
        for first, second in ((row['method_a'], row['method_b']),
                              (row['method_b'], row['method_a'])):
            if not is_kl_method(first) or first == second:
                continue
            if references is not None and first not in references:
                continue
            key = (row['dataset'], row['architecture_a'], row['fraction'], first, second)
            buckets.setdefault(key, []).append(row)

    def scored_means(dataset, architecture, fraction, name, seeds):
        entries = [scored.get((dataset, architecture, fraction, name, seed), (None, None))
                   for seed in seeds]
        return (mean_std([entry[0] for entry in entries])[0],
                mean_std([entry[1] for entry in entries])[0])

    def gap(a, b):
        return abs(a - b) if np.isfinite(a) and np.isfinite(b) else float('nan')

    results = []
    for (dataset, architecture, fraction, reference, method), group in buckets.items():
        mean, std, count = mean_std([row['kappa'] for row in group])
        floor = floors.get((fraction, architecture, reference))
        seeds = {str(row['seed_a']) for row in group}
        d_reference, acc_reference = scored_means(
            dataset, architecture, fraction, reference, seeds)
        d_method, acc_method = scored_means(
            dataset, architecture, fraction, method, seeds)
        results.append(dict(
            dataset=dataset, architecture=architecture, fraction=fraction,
            reference=reference, method=method, n_pairs=count,
            kappa_mean=mean, kappa_std=std,
            seed_floor=floor[0] if floor else float('nan'),
            kappa_over_floor=floor_ratio(mean, floor),
            overlap_frac_mean=mean_std([row['overlap_frac'] for row in group])[0],
            class_tv_mean=mean_std([row['class_tv'] for row in group])[0],
            d_rel_reference=d_reference, d_rel_method=d_method,
            d_rel_gap=gap(d_method, d_reference),
            acc_reference=acc_reference, acc_method=acc_method,
            acc_gap=gap(acc_method, acc_reference),
        ))
    results.sort(key=lambda row: (row['dataset'], row['architecture'], row['fraction'],
                                  row['reference'],
                                  -row['kappa_mean'] if np.isfinite(row['kappa_mean'])
                                  else np.inf))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
def write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _value(number, spec='7.3f'):
    return format(number, spec) if number is not None and np.isfinite(number) else '    –  '


def print_architecture_table(summary, logger):
    if not summary:
        logger.info("No cross-architecture pairs found.")
        return
    logger.info(
        "\nCross-architecture agreement. kappa = 0 is chance, 1 is identical. "
        "The seed floor is\nthe same method's agreement with itself across seeds; "
        "kappa/floor near 1 means the\narchitecture does no more than reseeding, "
        "near 0 means it decides the selection.\nThe ratio is blank where the floor "
        "is itself within noise of chance.\n")
    header = (f"{'frac':<7}{'method':<30}{'pair':<24}{'n':<4}"
              f"{'kappa':<18}{'floor':<9}{'kappa/floor':<13}{'class TV':<10}")
    logger.info(header + "\n" + "-" * len(header))
    last = None
    for row in summary:
        if last is not None and row['fraction'] != last:
            logger.info("-" * len(header))
        last = row['fraction']
        pair = f"{row['architecture_a']} vs {row['architecture_b']}"
        logger.info(
            f"{row['fraction']:<7g}{row['method']:<30}{pair:<24}{row['n']:<4}"
            f"{_value(row['kappa'], '+.3f')} ± {_value(row['kappa_std'], '.3f')}   "
            f"{_value(row['seed_floor'], '+.3f'):<9}"
            f"{_value(row['kappa_over_floor'], '.2f'):<13}"
            f"{_value(row['class_tv'], '.4f'):<10}")


def print_closest_table(results, logger):
    if not results:
        logger.info("No KL reference runs to rank the other methods against.")
        return
    logger.info(
        "\nClosest method to the KL reference, by how often they pick the same "
        "examples.\nD_rel gap and accuracy gap come from the score_subsets tables "
        "when they were supplied:\na method can be far in set space yet match on "
        "D_rel, which would say the objective\nadmits many near-optimal subsets.\n")
    header = (f"{'arch':<12}{'frac':<7}{'reference':<22}{'method':<28}{'n':<4}"
              f"{'kappa':<18}{'/floor':<9}{'overlap':<9}{'dD_rel':<10}{'dacc':<8}")
    logger.info(header + "\n" + "-" * len(header))
    last = None
    for row in results:
        key = (row['architecture'], row['fraction'], row['reference'])
        if last is not None and key != last:
            logger.info("-" * len(header))
        last = key
        logger.info(
            f"{row['architecture']:<12}{row['fraction']:<7g}{row['reference']:<22}"
            f"{row['method']:<28}{row['n_pairs']:<4}"
            f"{_value(row['kappa_mean'], '+.3f')} ± {_value(row['kappa_std'], '.3f')}   "
            f"{_value(row['kappa_over_floor'], '.2f'):<9}"
            f"{_value(row['overlap_frac_mean'], '.3f'):<9}"
            f"{_value(row['d_rel_gap'], '.4f'):<10}"
            f"{_value(row['acc_gap'], '.2f'):<8}")


def print_consensus_table(rows, logger):
    if not rows:
        logger.info("No architecture group large enough for a consensus count.")
        return
    logger.info(
        "\nHow many architectures picked each example, against the independent-draw "
        "null.\nratio > 1 at the top row means a shared core the architectures agree on.\n")
    header = (f"{'frac':<7}{'method':<30}{'seed':<7}{'picked by':<11}"
              f"{'count':<10}{'expected':<12}{'ratio':<9}{'|union|/m':<10}")
    logger.info(header + "\n" + "-" * len(header))
    for row in rows:
        if row['selected_by'] == 0:
            continue
        logger.info(
            f"{row['fraction']:<7g}{row['method']:<30}{row['seed']:<7}"
            f"{row['selected_by']}/{row['n_groups']:<9}"
            f"{row['count']:<10}{row['expected_count']:<12.1f}"
            f"{_value(row['ratio'], '.2f'):<9}"
            f"{_value(row['union_over_budget'], '.2f'):<10}")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
WIDTHS_IN = {'single': 3.4, 'onehalf': 5.2, 'double': 7.2}

RC = {
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'axes.linewidth': 0.7,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}


def _save(fig, out_stem, formats):
    os.makedirs(os.path.dirname(out_stem) or '.', exist_ok=True)
    written = []
    for fmt in formats:
        path = f"{out_stem}.{fmt}"
        fig.savefig(path, dpi=600 if fmt != 'pdf' else None, bbox_inches='tight')
        written.append(path)
    return written


def plot_kappa_heatmap(labels, matrix, out_stem, title=None, width='onehalf',
                       formats=('pdf',)):
    """Method-by-method mean kappa, annotated.

    Diverging around zero, because that is where the quantity's meaning sits:
    white is chance, red is agreement, blue is the two methods avoiding each
    other's picks. The diagonal is left blank — a method agrees with itself by
    construction, and colouring it only compresses the scale everything else
    has to share.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    shown = np.array(matrix, dtype=np.float64, copy=True)
    np.fill_diagonal(shown, np.nan)
    finite = shown[np.isfinite(shown)]
    limit = max(float(np.max(np.abs(finite))), 0.05) if finite.size else 1.0

    with plt.rc_context(RC):
        size = WIDTHS_IN[width]
        fig, ax = plt.subplots(figsize=(size, size * 0.92))
        image = ax.imshow(np.ma.masked_invalid(shown), cmap='coolwarm',
                          vmin=-limit, vmax=limit)
        image.cmap.set_bad('0.92')
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i == j or not np.isfinite(shown[i, j]):
                    continue
                ax.text(j, i, f"{shown[i, j]:.2f}", ha='center', va='center',
                        fontsize=6.5, color='black')
        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        bar.set_label(r'chance-corrected agreement $\kappa$', fontsize=8)
        if title:
            ax.set_title(title)
        fig.tight_layout()
        written = _save(fig, out_stem, formats)
        plt.close(fig)
    return written


def plot_method_dendrogram(labels, matrix, out_stem, title=None, width='onehalf',
                           formats=('pdf',)):
    """Average-linkage tree on ``1 - kappa``: which selectors behave alike."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    distance = 1.0 - matrix
    np.fill_diagonal(distance, 0.0)
    if not np.isfinite(distance).all():
        return []
    distance = np.clip((distance + distance.T) / 2.0, 0.0, None)
    np.fill_diagonal(distance, 0.0)

    with plt.rc_context(RC):
        size = WIDTHS_IN[width]
        fig, ax = plt.subplots(figsize=(size, size * 0.72))
        dendrogram(linkage(squareform(distance, checks=False), method='average'),
                   labels=labels, ax=ax, color_threshold=None,
                   above_threshold_color='#2B6CB0')
        ax.set_ylabel(r'$1 - \kappa$')
        ax.tick_params(axis='x', rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('right')
        if title:
            ax.set_title(title)
        fig.tight_layout()
        written = _save(fig, out_stem, formats)
        plt.close(fig)
    return written


def plot_architecture_bars(summary, fraction, out_stem, title=None, width='double',
                           formats=('pdf',)):
    """Cross-architecture kappa per method, with each method's seed floor marked."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = [row for row in summary if row['fraction'] == fraction
            and np.isfinite(row['kappa'])]
    if not rows:
        return []
    methods = sorted({row['method'] for row in rows},
                     key=lambda name: (not is_kl_method(name), name))
    pairs = sorted({(row['architecture_a'], row['architecture_b']) for row in rows})
    lookup = {(row['method'], row['architecture_a'], row['architecture_b']): row
              for row in rows}

    with plt.rc_context(RC):
        size = WIDTHS_IN[width]
        fig, ax = plt.subplots(figsize=(size, size * 0.46))
        positions = np.arange(len(methods), dtype=np.float64)
        bar_width = 0.8 / max(len(pairs), 1)
        for index, pair in enumerate(pairs):
            values = [lookup.get((method, *pair), {}).get('kappa', np.nan)
                      for method in methods]
            errors = [lookup.get((method, *pair), {}).get('kappa_std', np.nan)
                      for method in methods]
            offset = (index - (len(pairs) - 1) / 2.0) * bar_width
            ax.bar(positions + offset, values, bar_width, yerr=errors, capsize=2,
                   label=f"{pair[0]} vs {pair[1]}", edgecolor='black', linewidth=0.4)

        for index, method in enumerate(methods):
            floors = [lookup[key]['seed_floor'] for key in lookup
                      if key[0] == method and np.isfinite(lookup[key]['seed_floor'])]
            if not floors:
                continue
            ax.hlines(float(np.mean(floors)), index - 0.45, index + 0.45,
                      color='black', lw=1.1, ls='--', zorder=4)

        ax.set_xticks(positions)
        ax.set_xticklabels(methods, rotation=30, ha='right')
        ax.set_ylabel(r'$\kappa$ across architectures')
        ax.axhline(0.0, color='0.4', lw=0.7)
        ax.grid(axis='y', alpha=0.25, lw=0.4)
        ax.legend(fontsize=6.5, frameon=False, ncol=max(len(pairs), 1),
                  loc='upper center', bbox_to_anchor=(0.5, -0.32))
        if title:
            ax.set_title(title)
        fig.tight_layout()
        written = _save(fig, out_stem, formats)
        plt.close(fig)
    return written


def plot_consensus(rows, fraction, method, out_stem, title=None, width='single',
                   formats=('pdf',)):
    """Observed versus null count of examples picked by j of the architectures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    selected = [row for row in rows
                if row['fraction'] == fraction and row['method'] == method
                and row['selected_by'] > 0]
    if not selected:
        return []
    groups = max(row['n_groups'] for row in selected)
    observed, expected = [], []
    for j in range(1, groups + 1):
        values = [row for row in selected if row['selected_by'] == j]
        observed.append(float(np.mean([row['count'] for row in values])) if values else 0.0)
        expected.append(float(np.mean([row['expected_count'] for row in values]))
                        if values else 0.0)

    with plt.rc_context(RC):
        size = WIDTHS_IN[width]
        fig, ax = plt.subplots(figsize=(size, size * 0.78))
        positions = np.arange(1, groups + 1, dtype=np.float64)
        ax.bar(positions - 0.19, observed, 0.38, label='observed',
               color='#2B6CB0', edgecolor='black', linewidth=0.4)
        ax.bar(positions + 0.19, expected, 0.38, label='independent draws',
               color='0.75', edgecolor='black', linewidth=0.4)
        ax.set_yscale('log')
        ax.set_xticks(positions)
        ax.set_xlabel('architectures selecting the example')
        ax.set_ylabel('examples')
        ax.legend(fontsize=7, frameon=False)
        ax.grid(axis='y', alpha=0.25, lw=0.4)
        if title:
            ax.set_title(title, fontsize=8)
        fig.tight_layout()
        written = _save(fig, out_stem, formats)
        plt.close(fig)
    return written


def draw_all(pairs, consensus, summary, out_dir, dataset, architectures, fractions,
             width='onehalf', formats=('pdf',), title=True, logger=None):
    logger = logger or logging.getLogger('selection_overlap')
    written = []
    for fraction in fractions:
        for architecture in architectures:
            labels, matrix = kappa_matrix(pairs, fraction, architecture)
            if len(labels) < 2:
                continue
            stem = os.path.join(
                out_dir, f"selection_kappa_{dataset}_{architecture}_f{fraction:g}")
            heading = (f"{dataset} / {architecture}, fraction {fraction:g}"
                       if title else None)
            written += plot_kappa_heatmap(labels, matrix, stem, heading, width, formats)
            written += plot_method_dendrogram(
                labels, matrix,
                os.path.join(out_dir,
                             f"selection_dendrogram_{dataset}_{architecture}_f{fraction:g}"),
                heading, width, formats)

        written += plot_architecture_bars(
            summary, fraction,
            os.path.join(out_dir, f"selection_architecture_{dataset}_f{fraction:g}"),
            f"{dataset}, fraction {fraction:g}" if title else None, 'double', formats)

        for method in sorted({row['method'] for row in consensus
                              if row['fraction'] == fraction}):
            written += plot_consensus(
                consensus, fraction, method,
                os.path.join(out_dir,
                             f"selection_consensus_{dataset}_{method}_f{fraction:g}"),
                f"{dataset} / {method}, fraction {fraction:g}" if title else None,
                'single', formats)
    if written:
        logger.info(f"Wrote {len(written)} figure(s) -> {out_dir}")
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='cifar10', choices=list(N_TRAIN))
    p.add_argument('--architectures', default=['resnet18', 'vgg16', 'inception'],
                   nargs='+')
    p.add_argument('--fractions', default=[0.1], type=float, nargs='+')
    p.add_argument('--seeds', default=[42, 43, 44, 45, 46], type=int, nargs='+')
    p.add_argument('--methods', default=None, nargs='+',
                   help='Default: every method that saved indices on disk.')
    p.add_argument('--checkpoints', default=None, type=int, nargs='+',
                   help="Restrict to runs tagged with these K values.")
    p.add_argument('--scratch', default=os.environ.get('SCRATCH', '/home/mila/a/ahmedm/scratch'))
    p.add_argument('--variant', default='sgd', choices=['sgd', 'adam'],
                   help="Result tree. 'sgd' is the historical layout; 'adam' reads "
                        "the adam/ subfolders.")
    p.add_argument('--data_dir', default=None,
                   help='Dataset root. Only the labels are read (no images decoded, no '
                        'GPU); it turns on kappa_stratified and the class-histogram '
                        'distance. Omit to skip those two columns.')
    p.add_argument('--scored_input', default=None, nargs='+',
                   help='d_exact_*.csv tables from scripts/analyze/score_subsets.py, to '
                        'add the D_rel and accuracy gaps to the closest-method ranking. '
                        'Files, directories, or globs.')
    p.add_argument('--references', default=None, nargs='+',
                   help='Methods to rank the others against. Default: every kl_faithful* '
                        'variant found on disk.')
    p.add_argument('--out_dir', default=None,
                   help='Default: $SCRATCH/data_selection_baselines')
    p.add_argument('--plot_dir', default=None,
                   help='Default: $SCRATCH/data_selection_baselines/plots')
    p.add_argument('--no_plots', action='store_true', default=False)
    p.add_argument('--width', choices=list(WIDTHS_IN), default='onehalf')
    p.add_argument('--formats', nargs='+', default=['pdf'])
    p.add_argument('--no_title', action='store_true', default=False)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger('selection_overlap')

    out_dir = args.out_dir or tree(
        args.scratch, 'data_selection_baselines', args.variant)
    plot_dir = args.plot_dir or os.path.join(out_dir, 'plots')

    records = collect_subsets(args.scratch, args.dataset, args.architectures,
                              args.fractions, args.seeds, args.methods,
                              args.checkpoints, logger, variant=args.variant)
    if not records:
        logger.error(
            "No saved subsets found. Expected *_indices.pt under "
            f"{os.path.join(tree(args.scratch, 'gradmatch_swap', args.variant), args.dataset)} and "
            f"{tree(args.scratch, 'data_selection_baselines', args.variant)}.")
        return {}
    logger.info(f"Loaded {len(records)} saved subsets across "
                f"{len({r['architecture'] for r in records})} architecture(s), "
                f"{len({run_label(r) for r in records})} method run(s), "
                f"{len({r['seed'] for r in records})} seed(s).")

    labels_by_seed, num_classes = None, None
    if args.data_dir:
        labels_by_seed = {seed: pool_labels(args.dataset, args.data_dir, seed)
                          for seed in sorted({r['seed'] for r in records})}
        num_classes = int(max(array.max() for array in labels_by_seed.values()) + 1)
        logger.info(f"Class-stratified metrics on: {num_classes} classes from "
                    f"{args.data_dir}.")

    pairs = (architecture_contrast(records, labels_by_seed, num_classes, logger)
             + method_contrast(records, labels_by_seed, num_classes)
             + seed_contrast(records, logger))
    consensus = consensus_rows(records)
    floors = seed_floors(pairs)
    summary = architecture_summary(pairs, floors)
    scored = read_scored_tables(args.scored_input) if args.scored_input else {}
    if args.scored_input:
        logger.info(f"Joined {len(scored)} scored subset(s) for the D_rel and "
                    f"accuracy gaps.")
    closest = closest_methods(pairs, floors, args.references, scored)

    print_architecture_table(summary, logger)
    print_consensus_table(consensus, logger)
    print_closest_table(closest, logger)

    tag = (f"f{args.fractions[0]:g}" if len(args.fractions) == 1
           else f"f{min(args.fractions):g}-{max(args.fractions):g}")
    written = [
        write_csv(os.path.join(out_dir, f'selection_overlap_{args.dataset}_{tag}.csv'),
                  PAIR_FIELDS, pairs),
        write_csv(os.path.join(out_dir, f'selection_consensus_{args.dataset}_{tag}.csv'),
                  CONSENSUS_FIELDS, consensus),
        write_csv(os.path.join(out_dir, f'selection_closest_{args.dataset}_{tag}.csv'),
                  CLOSEST_FIELDS, closest),
    ]
    for path in written:
        logger.info(f"Saved -> {path}")

    if not args.no_plots:
        draw_all(pairs, consensus, summary, plot_dir, args.dataset,
                 args.architectures, args.fractions, args.width,
                 tuple(args.formats), not args.no_title, logger)

    return dict(records=records, pairs=pairs, consensus=consensus,
                summary=summary, closest=closest, floors=floors)


if __name__ == '__main__':
    main()
