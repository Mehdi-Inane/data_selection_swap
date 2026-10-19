"""LLM instruction-tuning data selection (LESS-compatible feature store).

This subpackage ports the trajectory/faithfulness selector of
:mod:`data_selection_swap` to the instruction-tuning setting of
LESS (Xia et al., ICML 2024). It is intentionally **additive and isolated**:

* it depends only on ``numpy`` and ``torch`` (no torchvision / CV packages);
* it never imports the CV experiment modules
  (``common``, ``paths``, ``objective``, ``trajectory``, ``selection``);
* it reads LESS gradient features from disk and writes LESS-compatible
  selected-data files, so LESS stays an external checkout, not a dependency.

The two selection settings share one quadratic set objective
(:mod:`~data_selection_swap.llm.objective`) and one swap solver
(:mod:`~data_selection_swap.llm.swap`); only the linear score vector ``c``
changes:

* **Setting A (faithful / validation-free):** ``c`` is built from the pool mean
  gradient, so the objective is exactly the manuscript's trajectory
  discrepancy ``D(I)`` up to an additive constant.
* **Setting B (targeted / LESS-style):** ``c`` is the LESS influence score
  (cosine similarity to the few-shot validation gradient), so ``top-k`` on
  ``c`` reproduces LESS and the swap adds a gradient-redundancy penalty.
"""

__all__ = [
    "config",
    "features",
    "targets",
    "objective",
    "swap",
    "scoring",
    "datastore",
]
