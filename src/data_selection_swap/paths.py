"""On-disk layout for the SGD results and the Adam rerun.

SGD keeps the historical directories. Adam and AdamW write under an ``adam``
subfolder of each root, so a new optimizer cannot overwrite an existing run:

    $SCRATCH/data_selection_swap/<dataset>/...
    $SCRATCH/data_selection_swap/adam/<dataset>/...
    $SCRATCH/data_selection_baselines/<method>/...
    $SCRATCH/data_selection_baselines/adam/<method>/...
    $SCRATCH/gradmatch_swap/<dataset>/...
    $SCRATCH/gradmatch_swap/adam/<dataset>/...
"""

import os

ADAM_OPTIMIZERS = ('adam', 'adamw', 'adam_update')


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
