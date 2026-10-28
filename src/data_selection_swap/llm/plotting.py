"""Budget-efficiency curves and (optional) plots.

Data prep is pure and dependency-free so it is unit-testable everywhere;
``save_budget_curve`` imports matplotlib lazily and is a no-op (returns ``None``)
when matplotlib is unavailable, so analysis code never hard-depends on it.

A "row" is one selection run with keys ``fraction``, a value (accuracy or
objective value), and a task/label. Downstream accuracy is read from a per-run
``accuracy.json`` ({"accuracy": x}) written by the eval step; objective values
come from ``diagnostics.json``.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence


def row_label(row: dict) -> str:
    """Stable series label for a run row (baselines by method, ours by setting/rule)."""
    if row.get("method"):
        return str(row["method"])
    setting = row.get("setting") or "selection"
    rule = row.get("rule")
    return f"{setting}-{rule}" if rule else str(setting)


def budget_curve(rows: Sequence[dict], value_key: str = "accuracy") -> Dict[str, Dict[str, List]]:
    """Group rows into ``{task: {label: [(fraction, value), ...]}}`` (fraction-sorted).

    ``value_key`` selects the metric column (e.g. ``"accuracy"`` for downstream
    performance, ``"value"``/``"objective"`` for the selection objective).
    """
    curve: Dict[str, Dict[str, List]] = {}
    for r in rows:
        if value_key not in r or r.get(value_key) is None or r.get("fraction") is None:
            continue
        task = str(r.get("task") or "all")
        label = row_label(r)
        curve.setdefault(task, {}).setdefault(label, []).append(
            (float(r["fraction"]), float(r[value_key])))
    for task in curve:
        for label in curve[task]:
            curve[task][label].sort(key=lambda t: t[0])
    return curve


def read_run_rows(root: str) -> List[dict]:
    """Merge every ``diagnostics.json`` (+ optional ``accuracy.json``) under ``root``."""
    rows: List[dict] = []
    for dirpath, _, files in os.walk(root):
        if "diagnostics.json" not in files:
            continue
        with open(os.path.join(dirpath, "diagnostics.json")) as fh:
            row = json.load(fh)
        row.setdefault("run", os.path.relpath(dirpath))
        swap = row.get("swap") or {}
        row.setdefault("value", swap.get("value"))
        if "accuracy.json" in files:
            with open(os.path.join(dirpath, "accuracy.json")) as fh:
                row["accuracy"] = json.load(fh).get("accuracy")
        rows.append(row)
    rows.sort(key=lambda r: (str(r.get("task")), row_label(r), float(r.get("fraction") or 0)))
    return rows


def save_budget_curve(rows: Sequence[dict], out_path: str, value_key: str = "accuracy",
                      title: Optional[str] = None) -> Optional[str]:
    """Render one panel per task; returns the path, or ``None`` if matplotlib is absent."""
    curve = budget_curve(rows, value_key=value_key)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    tasks = sorted(curve)
    ncols = max(1, len(tasks))
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for label, pts in sorted(curve[task].items()):
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", label=label)
        ax.set_xlabel("selection fraction")
        ax.set_title(task)
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel(value_key)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6))
    fig.suptitle(title or f"budget-efficiency curve ({value_key})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
