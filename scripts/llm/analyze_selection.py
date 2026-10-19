#!/usr/bin/env python3
"""Aggregate ``diagnostics.json`` files from many selection runs into one table.

Point it at the parent of a set of run directories (each produced by
``select_swap.py`` / ``select_baselines.py``) and it emits a flat summary — one
row per run — as JSON on stdout and (optionally) CSV, so budget-efficiency curves
and the swap-vs-LESS comparison can be plotted without re-running selection.
"""

import argparse
import csv
import json
import os
import sys


_FIELDS = ["run", "setting", "method", "task", "fraction", "normalize",
           "aggregation", "rule", "seed", "subset_size", "swap_value",
           "swap_redundancy", "less_value", "less_redundancy", "jaccard_vs_swap"]


def _flatten(path: str, doc: dict) -> dict:
    swap = doc.get("swap", {}) or {}
    less = doc.get("less_topk", {}) or {}
    return {
        "run": os.path.relpath(path),
        "setting": doc.get("setting"), "method": doc.get("method"),
        "task": doc.get("task"), "fraction": doc.get("fraction"),
        "normalize": doc.get("normalize"), "aggregation": doc.get("aggregation"),
        "rule": doc.get("rule"), "seed": doc.get("seed"),
        "subset_size": doc.get("subset_size"),
        "swap_value": swap.get("value"), "swap_redundancy": swap.get("redundancy"),
        "less_value": less.get("value"), "less_redundancy": less.get("redundancy"),
        "jaccard_vs_swap": less.get("jaccard_vs_swap"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="directory tree containing diagnostics.json files")
    p.add_argument("--csv", default=None, help="optional CSV output path")
    args = p.parse_args()

    rows = []
    for dirpath, _, files in os.walk(args.root):
        if "diagnostics.json" in files:
            with open(os.path.join(dirpath, "diagnostics.json")) as fh:
                rows.append(_flatten(dirpath, json.load(fh)))
    rows.sort(key=lambda r: (str(r["setting"]), str(r["task"]),
                             r["fraction"] or 0, str(r["method"]), r["seed"] or 0))

    json.dump(rows, sys.stdout, indent=2)
    print()
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_FIELDS)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
