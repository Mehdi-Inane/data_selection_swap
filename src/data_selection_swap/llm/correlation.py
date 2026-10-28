"""Score-vs-accuracy correlation (LLM port of the CV D(α)-correlation study).

Given runs that each have a selection *score* (the swap objective value, the
residual trajectory discrepancy, or LESS's influence) and a measured downstream
*accuracy*, report the Spearman rank correlation — per method and overall. A
method whose selection score predicts accuracy better is the one whose ranking is
doing the real work. Rows are dicts (e.g. the merged run rows from
:func:`data_selection_swap.llm.plotting.read_run_rows`).
"""

from __future__ import annotations

from typing import Dict, Hashable, Sequence, Tuple

from .scoring import spearman


def score_vs_accuracy(rows: Sequence[dict], score_key: str = "value",
                      acc_key: str = "accuracy",
                      group_keys: Tuple[str, ...] = ("method",)) -> Dict[Hashable, dict]:
    """Spearman(score, accuracy) overall (key ``"all"``) and per ``group_keys`` value."""
    buckets: Dict[Hashable, Tuple[list, list]] = {"all": ([], [])}
    for r in rows:
        score, acc = r.get(score_key), r.get(acc_key)
        if score is None or acc is None:
            continue
        score, acc = float(score), float(acc)
        buckets["all"][0].append(score)
        buckets["all"][1].append(acc)
        key = tuple(str(r.get(k)) for k in group_keys)
        b = buckets.setdefault(key, ([], []))
        b[0].append(score)
        b[1].append(acc)

    out: Dict[Hashable, dict] = {}
    for key, (xs, ys) in buckets.items():
        out[key] = {"n": len(xs), "spearman": spearman(xs, ys) if len(xs) >= 2 else float("nan")}
    return out
