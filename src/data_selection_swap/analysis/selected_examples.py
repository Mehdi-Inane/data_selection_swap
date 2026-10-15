#!/usr/bin/env python3
"""Draw a grid of training images chosen by each selection method.

One figure per dataset and architecture. Eight rows, top to bottom:

  * Gradient Swap, full-network gradients, K=1
    (``kl_faithful_full_<architecture>_sgd``)
  * random, herding, moderate, el2n, moso, graphcut, facility_location

Each row shows six examples drawn from that method's saved subset. The draw
uses ``--example_seed`` so the same files produce the same panel. Indices on
disk are pool-relative; ``selection_overlap.pool_indices`` maps them back to
the original training image. Images are loaded with no crop, flip, or
normalization.

A missing indices file, or two configurations in the same seed cell, stops
the run before any figure is written. The second case is the error raised by
``selection_overlap.discover_subsets``.

Usage:
    python scripts/plot/selected_examples.py --dataset cifar10 \\
        --architecture resnet18 --data_dir /path/to/cifar10
"""

import argparse
import os

import numpy as np

from ..paths import tree
from .checkpoint_ablation import ARCHITECTURE_NAMES, full_method
from .comparison import N_TRAIN, run_dir
from .selection_overlap import (
    N_TOTAL, discover_subsets, load_indices, pool_indices,
)

OURS_K = 1
N_EXAMPLES = 6
BASELINES = (
    'random', 'herding', 'moderate', 'el2n', 'moso', 'graphcut',
    'facility_location',
)
ROW_LABELS = {
    'random': 'Random',
    'herding': 'Herding',
    'moderate': 'Moderate',
    'el2n': 'EL2N',
    'moso': 'MoSo',
    'graphcut': 'GraphCut',
    'facility_location': 'Facility Location',
}
OURS_LABEL = 'Gradient Swap'
DATASET_NAMES = {
    'cifar10': 'CIFAR-10',
    'cifar100': 'CIFAR-100',
    'mnist': 'MNIST',
    'imagenet': 'ImageNet',
}


def row_specs(architecture, variant='sgd'):
    """``(method, k, label)`` in figure order. Baselines have no K tag."""
    ours = full_method(architecture, variant)
    rows = [(ours, OURS_K, OURS_LABEL)]
    rows.extend((name, None, ROW_LABELS[name]) for name in BASELINES)
    return rows


def indices_path_for_row(scratch, dataset, architecture, budget, seed, method, k,
                         variant=''):
    """The one indices file for this row. Missing files and K conflicts fail here."""
    directory = run_dir(
        scratch, method, dataset, architecture, budget, seed, variant=variant)
    found = discover_subsets(directory, method, budget)
    if k is None:
        matched = [path for found_k, path in found if found_k is None]
        wanted = 'untagged'
    else:
        matched = [path for found_k, path in found if found_k == int(k)]
        wanted = f'K={int(k)}'
    if len(matched) != 1:
        raise FileNotFoundError(
            f"Expected one {wanted} indices file for method {method!r} "
            f"in {directory}, found {len(matched)}.")
    return matched[0]


def sample_indices(indices, n_examples, rng):
    """``n_examples`` distinct members of ``indices``, in draw order."""
    if indices.size < n_examples:
        raise ValueError(
            f"Subset has {indices.size} indices, which is fewer than the "
            f"{n_examples} examples the figure asks for.")
    return np.asarray(rng.choice(indices, size=n_examples, replace=False))


def collect_rows(scratch, dataset, architecture, fraction, seed, n_examples,
                 example_seed, variant=''):
    """One record per row, with pool and absolute indices already sampled.

    Every indices path is resolved before any subset is sampled, so a missing
    method raises before a later row is drawn.
    """
    budget = int(fraction * N_TRAIN[dataset])
    specs = row_specs(architecture, variant)
    paths = [
        indices_path_for_row(
            scratch, dataset, architecture, budget, seed, method, k, variant)
        for method, k, _label in specs
    ]
    pool = pool_indices(dataset, seed)
    rng = np.random.default_rng(example_seed)
    rows = []
    for (method, k, label), path in zip(specs, paths):
        saved = load_indices(path)
        chosen = sample_indices(saved, n_examples, rng)
        if np.any(chosen < 0) or np.any(chosen >= pool.size):
            raise ValueError(
                f"{path} has pool indices outside 0..{pool.size - 1}.")
        absolute = pool[chosen]
        rows.append(dict(
            method=method, k=k, label=label, indices_path=path,
            pool_index=chosen, absolute=absolute,
        ))
    return rows


def open_raw_dataset(dataset, data_dir, download=False):
    """Training split with ``transform=None``: the original image, not a crop."""
    import torchvision.datasets as datasets

    if dataset == 'imagenet':
        return datasets.ImageFolder(os.path.join(data_dir, 'train'))
    builders = {
        'cifar10': datasets.CIFAR10,
        'cifar100': datasets.CIFAR100,
        'mnist': datasets.MNIST,
    }
    if dataset not in builders:
        raise ValueError(
            f"No raw image loader for {dataset!r}. "
            f"Supported: {sorted(builders) + ['imagenet']}.")
    return builders[dataset](root=data_dir, train=True, download=download)


def class_name(dataset_obj, label):
    classes = getattr(dataset_obj, 'classes', None)
    if classes is None:
        return str(int(label))
    return str(classes[int(label)]).replace('_', ' ')


def read_examples(dataset_obj, absolute):
    """``(image array, class name)`` for each absolute training index."""
    panels = []
    n_total = len(dataset_obj)
    for index in absolute:
        index = int(index)
        if index < 0 or index >= n_total:
            raise ValueError(
                f"Absolute index {index} is outside the training set "
                f"(size {n_total}).")
        image, label = dataset_obj[index]
        panels.append((np.asarray(image), class_name(dataset_obj, label)))
    return panels


def figure_stem(out_dir, dataset, architecture, fraction, seed):
    return os.path.join(
        out_dir,
        f"selected_examples_{dataset}_{architecture}_f{fraction:g}_seed{seed}")


def draw_selected_examples(rows, panels, title, out_stem):
    """Write ``out_stem``.png and ``out_stem``.pdf. ``panels[i]`` is row ``i``."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_rows = len(rows)
    n_cols = len(panels[0]) if panels else 0
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(1.7 * n_cols, 2.0 * n_rows), squeeze=False)
    for i, (row, images) in enumerate(zip(rows, panels)):
        for j, (image, name) in enumerate(images):
            ax = axes[i, j]
            if image.ndim == 2:
                ax.imshow(image, cmap='gray', interpolation='nearest')
            else:
                ax.imshow(image, interpolation='nearest')
            ax.set_xlabel(name, fontsize=7, labelpad=2)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[i, 0].set_ylabel(row['label'], fontsize=10)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_stem) or '.', exist_ok=True)
    written = []
    for ext in ('png', 'pdf'):
        path = f"{out_stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        written.append(path)
    plt.close(fig)
    return written


def figure_title(dataset, architecture, fraction, seed):
    data_name = DATASET_NAMES.get(dataset, dataset)
    arch_name = ARCHITECTURE_NAMES.get(architecture, architecture)
    return f"{data_name} / {arch_name}, {100 * fraction:g}% subset, seed {seed}"


def plot_selected_examples(scratch, dataset, architecture, out_dir,
                           fraction=0.1, seed=42, n_examples=N_EXAMPLES,
                           example_seed=0, variant='sgd', data_dir=None,
                           download=False, reader=None):
    """Resolve every row, then load images and write the grid.

    ``reader(absolute_indices) -> [(image, class name), ...]`` replaces the
    dataset load. Tests use it so the figure does not need CIFAR on disk.
    """
    rows = collect_rows(
        scratch, dataset, architecture, fraction, seed, n_examples,
        example_seed, variant=variant)
    if reader is None:
        if not data_dir:
            raise ValueError("--data_dir is required to load the training images.")
        dataset_obj = open_raw_dataset(dataset, data_dir, download=download)
        if len(dataset_obj) != N_TOTAL[dataset]:
            raise ValueError(
                f"{dataset} at {data_dir!r} has {len(dataset_obj)} training "
                f"examples, expected {N_TOTAL[dataset]}.")

        def reader(absolute):
            return read_examples(dataset_obj, absolute)
    panels = [reader(row['absolute']) for row in rows]
    for row, panel in zip(rows, panels):
        if len(panel) != n_examples:
            raise ValueError(
                f"{row['label']} returned {len(panel)} images, expected {n_examples}.")
    written = draw_selected_examples(
        rows, panels, figure_title(dataset, architecture, fraction, seed),
        figure_stem(out_dir, dataset, architecture, fraction, seed))
    for path in written:
        print(f"Saved -> {path}")
    return written


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='cifar10', choices=list(N_TRAIN))
    p.add_argument('--architecture', default='resnet18',
                   choices=['resnet18', 'vgg16', 'inception'])
    p.add_argument('--fraction', default=0.1, type=float)
    p.add_argument('--seed', default=42, type=int,
                   help='Selection seed whose subset is shown.')
    p.add_argument('--n_examples', default=N_EXAMPLES, type=int)
    p.add_argument('--example_seed', default=0, type=int,
                   help='RNG seed for which selected examples appear.')
    p.add_argument('--data_dir', required=True,
                   help='Dataset root, usually $SLURM_TMPDIR after staging.')
    p.add_argument('--scratch', default=os.environ.get(
        'SCRATCH', '/home/mila/a/ahmedm/scratch'))
    p.add_argument('--variant', default='sgd', choices=['sgd', 'adam'],
                   help="Result tree. 'sgd' is the historical layout; 'adam' "
                        "reads the adam/ subfolders.")
    p.add_argument('--out_dir', default=None,
                   help='Default: $SCRATCH/data_selection_baselines[/adam]/plots')
    p.add_argument('--download', action='store_true', default=False)
    args = p.parse_args(argv)

    out_dir = args.out_dir or os.path.join(
        tree(args.scratch, 'data_selection_baselines', args.variant), 'plots')
    plot_selected_examples(
        args.scratch, args.dataset, args.architecture, out_dir,
        fraction=args.fraction, seed=args.seed, n_examples=args.n_examples,
        example_seed=args.example_seed, variant=args.variant,
        data_dir=args.data_dir, download=args.download)


if __name__ == '__main__':
    main()
