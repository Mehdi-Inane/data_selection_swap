"""Plain dataclass configuration for the LLM selector.

Keeping configuration here (rather than argparse in the package) lets both the
``scripts/llm/*`` entry points and the tests build the same objects, and keeps
this subpackage free of CV experiment conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Normalization applied to gradient features before scoring. ``l2`` reproduces
#: LESS's cosine similarity (the sequence-length correction); ``none`` keeps the
#: raw projected gradients.
NORM_MODES = ("l2", "none")

#: How the per-subtask validation influences are pooled into one per-example
#: score. ``max`` mirrors LESS (Eq. 2 uses ``max_j``); ``mean`` is an ablation.
AGG_MODES = ("max", "mean")

#: Which selection setting the linear score encodes.
#: ``faithful`` -> target = pool mean gradient (Setting A, validation-free).
#: ``targeted`` -> target = validation gradient (Setting B, LESS-style).
SETTINGS = ("faithful", "targeted")

#: Swap search rule. ``ranked`` is the cheap Algorithm-3-style sweep; ``best``
#: is the exact best single swap (Algorithm-2-style) per iteration.
RULES = ("ranked", "best")


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration for a single selection run.

    Parameters
    ----------
    setting:
        ``"faithful"`` (Setting A, pool-mean target) or ``"targeted"``
        (Setting B, validation target).
    fraction:
        Fraction of the pool to select; the subset size is
        ``max(1, round(fraction * n))``.
    normalize:
        ``"l2"`` (default, LESS cosine) or ``"none"``.
    aggregation:
        ``"max"`` (default, LESS) or ``"mean"`` for pooling subtask influences.
    rule:
        ``"ranked"`` (default) or ``"best"``.
    max_iters:
        Maximum number of accepted swaps.
    block_size:
        Number of candidate "add" examples evaluated per chunk. Bounds the
        dense ``|I| x block`` cross-kernel block so the full ``n x n`` Gram is
        never materialized.
    tolerance:
        A swap is accepted only when its exact float64 change is below
        ``-tolerance`` (guards float rounding, mirroring the CV selector).
    seed:
        RNG seed for the random initialization of the swap search.
    """

    setting: str = "targeted"
    fraction: float = 0.05
    normalize: str = "l2"
    aggregation: str = "max"
    rule: str = "ranked"
    max_iters: int = 2000
    block_size: int = 4096
    tolerance: float = 1e-9
    seed: int = 0

    def __post_init__(self) -> None:
        if self.setting not in SETTINGS:
            raise ValueError(f"setting must be one of {SETTINGS}, got {self.setting!r}")
        if self.normalize not in NORM_MODES:
            raise ValueError(f"normalize must be one of {NORM_MODES}, got {self.normalize!r}")
        if self.aggregation not in AGG_MODES:
            raise ValueError(f"aggregation must be one of {AGG_MODES}, got {self.aggregation!r}")
        if self.rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}, got {self.rule!r}")
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {self.fraction!r}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size!r}")
        if self.max_iters < 0:
            raise ValueError(f"max_iters must be >= 0, got {self.max_iters!r}")

    def subset_size(self, n: int) -> int:
        """Fixed subset size ``m`` for a pool of ``n`` examples."""
        return max(1, min(n, int(round(self.fraction * n))))
