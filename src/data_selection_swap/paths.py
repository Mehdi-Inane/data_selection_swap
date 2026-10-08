"""On-disk layout for the SGD results and the Adam rerun.

SGD keeps the historical directories. Adam and AdamW write under an ``adam``
subfolder of each root, so a new optimizer cannot overwrite an existing run:

    <results>/data_selection_swap/<dataset>/...
    <results>/data_selection_swap/adam/<dataset>/...
    <results>/data_selection_baselines/<method>/...
    <results>/data_selection_baselines/adam/<method>/...
    <results>/gradmatch_swap/<dataset>/...
    <results>/gradmatch_swap/adam/<dataset>/...

``<results>`` is ``RESULTS_ROOT``, then ``SCRATCH``, then ``./results``.
"""

import os

ADAM_OPTIMIZERS = ('adam', 'adamw', 'adam_update')


def default_results_root() -> str:
    """Directory that holds ``data_selection_swap``, baselines, and KL runs.

    ``RESULTS_ROOT`` wins, then ``SCRATCH`` (set on many clusters), then a
    ``results`` folder in the current working directory.
    """
    for name in ('RESULTS_ROOT', 'SCRATCH'):
        value = os.environ.get(name)
        if value:
            return value
    return os.path.join(os.getcwd(), 'results')


def storage_variant(*names) -> str:
    """``adam`` when any name is an Adam-family optimizer, otherwise ``''``."""
    for name in names:
        if name in ADAM_OPTIMIZERS:
            return 'adam'
    return ''


def normalize_variant(variant) -> str:
    """Map a CLI value to the on-disk subfolder: ``''`` (SGD) or ``adam``."""
    if variant in (None, '', 'sgd'):
        return ''
    if variant in ADAM_OPTIMIZERS or variant == 'adam':
        return 'adam'
    raise ValueError(f"Unknown result variant {variant!r}. Choose 'sgd' or 'adam'.")


def tree(scratch: str, root_name: str, variant='') -> str:
    """``$SCRATCH/<root>`` or ``$SCRATCH/<root>/adam``."""
    sub = normalize_variant(variant)
    if sub:
        return os.path.join(scratch, root_name, sub)
    return os.path.join(scratch, root_name)
